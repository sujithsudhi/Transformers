from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from transformers import GPT2TokenizerFast

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.tinystory import DataPrep
from models.core import PositionalEncoding, TransformerDecoderLayer
from tool.utils import _to_serializable, load_config_target
from training import (
    Trainer,
    build_optimizer,
    evaluate,
    init_wandb_run,
    load_training_config,
    maybe_plot_history,
    maybe_save_history,
)
from training.trainer_utils import build_cross_entropy_loss


class DecoderLanguageModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if config.vocab_size is None or config.vocab_size <= 0:
            raise ValueError("vocab_size must be set for language modeling.")

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size,
                                            config.embed_dim,
                                            padding_idx=0)
        self.position = PositionalEncoding(max_len=config.max_length,
                                           embed_dim=config.embed_dim,
                                           dropout=config.dropout,
                                           method="trainable")
        self.decoder = nn.ModuleList(
            TransformerDecoderLayer(embedDim=config.embed_dim,
                                    numHeads=config.num_heads,
                                    mlpRatio=config.mlp_ratio,
                                    dropout=config.dropout,
                                    attentionDropout=config.attention_dropout)
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        if inputs.dtype != torch.long:
            inputs = inputs.long()

        x = self.token_embedding(inputs)
        x = self.position(x)

        for layer in self.decoder:
            x = layer(x, mask=attention_mask)

        x = self.norm(x)
        return self.lm_head(x)


def main() -> None:
    torch.manual_seed(42)

    app_config = load_config_target("configs.tinystories:TinyStoriesConfig")

    wandb_api_key = getattr(app_config, "wandb_api_key", None)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = str(wandb_api_key)

    if getattr(app_config, "wandb_disabled", False):
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)

    wandb_run, wandb_logger = init_wandb_run(app_config)

    if not hasattr(app_config, "data") or not hasattr(app_config, "model"):
        raise TypeError("Configuration object must expose 'data', 'model', and 'training'.")

    data_cfg = app_config.data
    dataloader_cfg = getattr(app_config, "dataloader", None)

    dataset_name = getattr(data_cfg, "dataset_name", None) or "roneneldan/TinyStories"
    if "/" not in dataset_name:
        dataset_name = "roneneldan/TinyStories"

    data_prep = DataPrep(dataset=dataset_name,
                         block_size=data_cfg.max_tokens,
                         batch_size=getattr(dataloader_cfg, "batch_size", 32),
                         shuffle=True,
                         num_workers=getattr(dataloader_cfg, "num_workers", 0),
                         pin_memory=getattr(dataloader_cfg, "pin_memory", True),
                         cache_dir=str(getattr(data_cfg, "cache_dir", "data/cache")),
                        )

    train_loader, val_loader = data_prep.prep()

    tokenizer_name = getattr(app_config, "tokenizer_name", "gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    vocab_size = tokenizer.vocab_size

    model_config = replace(app_config.model,
                           vocab_size=vocab_size,
                           max_length=data_cfg.max_tokens)
    model = DecoderLanguageModel(model_config)

    training_cfg = app_config.training
    optimizer = build_optimizer(model, lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)
    ce_loss = build_cross_entropy_loss()

    def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
        vocab = logits.size(-1)
        return ce_loss(logits.view(-1, vocab), targets.view(-1))

    training_config = load_training_config({
        "epochs": training_cfg.epochs,
        "device": training_cfg.device,
        "gradient_clip_norm": training_cfg.gradient_clip_norm,
        "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        "use_amp": training_cfg.use_amp,
        "log_interval": training_cfg.log_interval,
        "non_blocking": training_cfg.non_blocking,
        "early_stopping_patience": training_cfg.early_stopping_patience,
        "lr_reduction_patience": training_cfg.lr_reduction_patience,
        "lr_reduction_factor": training_cfg.lr_reduction_factor,
    })

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      train_loader=train_loader,
                      config=training_config,
                      val_loader=val_loader,
                      logger=wandb_logger,
                     )
    history = trainer.fit()

    test_metrics = evaluate(trainer.model,
                            val_loader,
                            loss_fn,
                            torch.device(training_config.device),
                            training_config.non_blocking,
                            progress_desc="Validation",
                           )

    history_path = getattr(app_config, "history_path", None)
    plot_path = getattr(app_config, "plot_path", None)
    checkpoint_path = Path(getattr(app_config, "checkpoint_path", Path("results/tinystories_encoder.pt")))

    maybe_save_history(history, history_path)
    maybe_plot_history(history, plot_path)

    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_state = trainer.best_model_state_dict()
    torch.save(
        {
            "model_state_dict": best_state,
            "config": {
                "model": _to_serializable(model_config),
                "training": _to_serializable(app_config.training),
                "data": _to_serializable(app_config.data),
            },
        },
        checkpoint_path,
    )

    summary: Dict[str, Any] = {
        "train_history": history,
        "val_metrics": test_metrics,
        "checkpoint_path": checkpoint_path.as_posix(),
    }
    print(json.dumps(summary, indent=2))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
