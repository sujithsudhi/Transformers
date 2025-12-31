from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import GPT2TokenizerFast

# Ensure repository root is importable when executing as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.train_encoder import DecoderLanguageModel  # noqa: E402
from data.tinystory import DataPrep, DataStreamer  # noqa: E402
from tool.utils import _to_serializable, load_config_target  # noqa: E402
from training import evaluate, load_training_config  # noqa: E402
from training.trainer_utils import build_cross_entropy_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on cached TinyStories validation tokens."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs.tinystories:TinyStoriesConfig",
        help="Python path to the configuration object.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load (defaults to config checkpoint_path).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=("train", "validation"),
        help="Cached token split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for evaluation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of batches to evaluate.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip loss/perplexity evaluation.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to generate from.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (0 to disable).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for generation.",
    )
    return parser.parse_args()


def _apply_top_k_top_p(logits: Tensor, top_k: int, top_p: float) -> Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, -float("inf"))
        restored = torch.full_like(logits, -float("inf"))
        logits = restored.scatter(-1, sorted_indices, sorted_logits)

    return logits


def _generate_text(model: torch.nn.Module,
                    tokenizer: GPT2TokenizerFast,
                    prompt: str,
                    max_new_tokens: int,
                    temperature: float,
                    top_k: int,
                    top_p: float,
                    device: torch.device,
                    max_length: int,
                ) -> str:
    model.eval()
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)

    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, -max_length:]

    if input_ids.numel() == 0:
        eos = tokenizer.eos_token_id or 0
        input_ids = torch.tensor([[eos]], device=device)

    with torch.no_grad():
        logits, past_kvs = model(input_ids, use_cache=True)

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled = next_logits / temperature
            filtered = _apply_top_k_top_p(scaled, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]

        with torch.no_grad():
            logits, past_kvs = model(next_token, past_kvs=past_kvs, use_cache=True)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    app_config = load_config_target(args.config)
    data_cfg = app_config.data
    dataloader_cfg = getattr(app_config, "dataloader", None)

    dataset_name = getattr(data_cfg, "dataset_name", None) or "roneneldan/TinyStories"
    if "/" not in dataset_name:
        dataset_name = "roneneldan/TinyStories"

    tokenizer_name = getattr(app_config, "tokenizer_name", "gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    model_config = replace(app_config.model,
                           vocab_size=vocab_size,
                           max_length=data_cfg.max_tokens)
    model = DecoderLanguageModel(model_config)

    checkpoint_path = Path(
        args.checkpoint or getattr(app_config, "checkpoint_path", "results/tiny_stories_transformer.pt")
    ).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(state_dict)

    training_config = load_training_config(_to_serializable(app_config.training))
    if args.device:
        training_config.device = args.device

    batch_size = args.batch_size or getattr(dataloader_cfg, "batch_size", 32)
    device = torch.device(training_config.device)
    model = model.to(device)

    if not args.no_eval:
        data_prep = DataPrep(dataset=dataset_name,
                             block_size=data_cfg.max_tokens,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=getattr(dataloader_cfg, "num_workers", 0),
                             pin_memory=getattr(dataloader_cfg, "pin_memory", True),
                             cache_dir=str(getattr(data_cfg, "cache_dir", "data/cache/tinystories")),
                             tokenizer_name=tokenizer_name,
                             stride=getattr(data_cfg, "stride", None),
                            )
        cache_path = data_prep._cache_path(args.split, tokenizer_name)
        if cache_path is None or not cache_path.exists():
            raise FileNotFoundError(
                f"Cached tokens not found at {cache_path}. Run training or data prep first."
            )

        tokens = torch.load(cache_path)
        dataset = DataStreamer(tokens=tokens,
                               blockSize=data_cfg.max_tokens,
                               stride=data_prep.stride)
        
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=getattr(dataloader_cfg, "num_workers", 0),
                                                 pin_memory=getattr(dataloader_cfg, "pin_memory", True),
                                                )

        if args.max_batches is not None:
            import itertools
            dataloader = itertools.islice(dataloader, args.max_batches)

        ce_loss = build_cross_entropy_loss()

        def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
            vocab = logits.size(-1)
            return ce_loss(logits.view(-1, vocab), targets.view(-1))

        metrics = evaluate(model=model,
                            dataloader=dataloader,
                            loss_fn=loss_fn,
                            device=device,
                            non_blocking=training_config.non_blocking,
                            progress_desc=f"{args.split.title()} Inference",
                        )

        loss = metrics.get("loss")
        if loss is not None and math.isfinite(loss):
            metrics["perplexity"] = math.exp(loss)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Split: {args.split}")
        print(metrics)

    if args.prompt:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        generated = _generate_text(model=model,
                                   tokenizer=tokenizer,
                                   prompt=args.prompt,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature,
                                   top_k=args.top_k,
                                   top_p=args.top_p,
                                   device=device,
                                   max_length=data_cfg.max_tokens,
                                  )
        print("\n=== Generated ===")
        print(generated)


if __name__ == "__main__":
    main()
