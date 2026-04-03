from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import TinyStoriesDataPrep
from tool.utils import _to_serializable, load_config_target
from training import (Trainer,
                      build_loss,
                      build_optimizer,
                      evaluate,
                      init_wandb_run,
                      load_training_config,
                      maybe_plot_history,
                      maybe_save_history,
                     )
from transformer_core import PositionalEncoding, TransformerDecoderLayer

try:
    import wandb
except ImportError:  # wandb is optional; keep training usable without it.
    wandb = None

CONFIG_TARGET = "configs.tinystories:TinyStoriesConfig"


class DecoderLanguageModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if config.vocab_size is None or config.vocab_size <= 0:
            raise ValueError("vocab_size must be set for language modeling.")

        self.config = config
        self.use_rope = bool(getattr(config, "use_rope", True))
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=0,
        )

        self.position: Optional[PositionalEncoding] = None
        if not self.use_rope:
            self.position = PositionalEncoding(
                max_len=config.max_length,
                embed_dim=config.embed_dim,
                dropout=config.dropout,
                method="trainable",
            )

        self.decoder = nn.ModuleList(
            TransformerDecoderLayer(config=config)
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

    def forward(
        self,
        inputs: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_kvs: Optional[list[tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if inputs.dtype != torch.long:
            inputs = inputs.long()

        hidden_states = self.token_embedding(inputs)
        if self.position is not None:
            position_offset = 0
            if past_kvs is not None and past_kvs:
                position_offset = past_kvs[0][0].size(2)
            hidden_states = self.position(hidden_states, offset=position_offset)

        presents: list[tuple[Tensor, Tensor]] = []
        for idx, layer in enumerate(self.decoder):
            past_kv = past_kvs[idx] if past_kvs is not None else None
            if use_cache:
                hidden_states, present = layer(
                    hidden_states,
                    mask=attention_mask,
                    past_kv=past_kv,
                    use_cache=True,
                )
                presents.append(present)
            else:
                hidden_states = layer(hidden_states, mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if use_cache:
            return logits, presents
        return logits


def _build_training_config(training_cfg: Any):
    return load_training_config(
        {
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
            "warmup_epochs": training_cfg.warmup_epochs,
            "warmup_start_factor": training_cfg.warmup_start_factor,
            "use_cosine_decay": training_cfg.use_cosine_decay,
            "min_lr": training_cfg.min_lr,
        }
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Any):
    if not training_cfg.use_cosine_decay and training_cfg.warmup_epochs <= 0:
        return None

    total_epochs = max(1, int(training_cfg.epochs))
    warmup_epochs = max(0, int(training_cfg.warmup_epochs))
    warmup_epochs = min(warmup_epochs, max(0, total_epochs - 1))
    cosine_epochs = max(1, total_epochs - warmup_epochs)

    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=float(training_cfg.warmup_start_factor),
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=float(training_cfg.min_lr),
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    return CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=float(training_cfg.min_lr),
    )


def _build_checkpoint_config(app_config: Any, model_config: Any) -> Dict[str, Any]:
    checkpoint_config = {
        "model": _to_serializable(model_config),
        "training": _to_serializable(app_config.training),
        "data": _to_serializable(app_config.data),
    }
    for section_name in ("dataloader", "optimizer", "loss"):
        section_value = getattr(app_config, section_name, None)
        if section_value is not None:
            checkpoint_config[section_name] = _to_serializable(section_value)
    return checkpoint_config


def main() -> None:
    app_config = load_config_target(CONFIG_TARGET)
    if not hasattr(app_config, "data") or not hasattr(app_config, "model") or not hasattr(app_config, "training"):
        raise TypeError("Configuration object must expose 'data', 'model', and 'training'.")

    torch.manual_seed(int(getattr(app_config.training, "seed", 42)))
    wandb_run, wandb_logger = init_wandb_run(app_config)

    data_cfg = app_config.data
    dataloader_cfg = getattr(app_config, "dataloader", None)

    dataset_name = getattr(data_cfg, "dataset_name", None) or "roneneldan/TinyStories"
    if "/" not in dataset_name:
        dataset_name = "roneneldan/TinyStories"

    tokenizer_name = getattr(app_config, "tokenizer_name", "gpt2")
    data_prep = TinyStoriesDataPrep(
        dataset=dataset_name,
        block_size=data_cfg.max_tokens,
        batch_size=getattr(dataloader_cfg, "batch_size", 32),
        shuffle=True,
        num_workers=getattr(dataloader_cfg, "num_workers", 0),
        pin_memory=getattr(dataloader_cfg, "pin_memory", True),
        cache_dir=str(getattr(data_cfg, "cache_dir", "data/cache/tinystories")),
        tokenizer_name=tokenizer_name,
        stride=getattr(data_cfg, "stride", None),
        use_map=getattr(data_cfg, "use_map", False),
        map_num_proc=getattr(data_cfg, "map_num_proc", 8),
        map_batch_size=getattr(data_cfg, "map_batch_size", 1000),
    )

    train_loader, val_loader, tokenizer = data_prep.prep()
    model_config = replace(
        app_config.model,
        vocab_size=tokenizer.vocab_size,
        max_length=data_cfg.max_tokens,
    )
    model = DecoderLanguageModel(model_config)

    training_cfg = app_config.training
    optimizer_cfg = getattr(app_config, "optimizer", None)
    optimizer = build_optimizer(
        model,
        lr=getattr(optimizer_cfg, "lr", training_cfg.lr),
        weight_decay=getattr(optimizer_cfg, "weight_decay", training_cfg.weight_decay),
        name=getattr(optimizer_cfg, "name", "adamw"),
        betas=getattr(optimizer_cfg, "betas", None),
        eps=getattr(optimizer_cfg, "eps", None),
    )

    loss_cfg = getattr(app_config, "loss", None)
    criterion = build_loss(
        name=getattr(loss_cfg, "name", "crossentropyloss"),
        beta=getattr(loss_cfg, "beta", 1.0),
    )

    def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
        vocab = logits.size(-1)
        return criterion(logits.view(-1, vocab), targets.view(-1))

    training_config = _build_training_config(training_cfg)
    scheduler = _build_scheduler(optimizer, training_cfg)

    checkpoint_path = Path(
        getattr(app_config, "checkpoint_path", Path("results/tiny_stories_transformer.pt"))
    ).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        config=training_config,
        val_loader=val_loader,
        scheduler=scheduler,
        logger=wandb_logger,
        best_checkpoint_dir=checkpoint_path.parent,
    )
    history = trainer.fit()

    val_metrics = evaluate(
        trainer.model,
        val_loader,
        loss_fn,
        torch.device(training_config.device),
        training_config.non_blocking,
        progress_desc="Validation",
    )

    history_path = getattr(app_config, "history_path", None)
    plot_path = getattr(app_config, "plot_path", None)
    maybe_save_history(history, history_path)
    maybe_plot_history(history, plot_path)

    best_state = trainer.best_model_state_dict()
    torch.save(
        {
            "model_state_dict": best_state,
            "config": _build_checkpoint_config(app_config, model_config),
        },
        checkpoint_path,
    )

    if wandb_run is not None and wandb is not None:
        wandb.log({f"val/{key}": value for key, value in val_metrics.items()})
        wandb_run.summary.update({f"val/{key}": value for key, value in val_metrics.items()})
        wandb_run.finish()

    summary: Dict[str, Any] = {
        "train_history": history,
        "val_metrics": val_metrics,
        "checkpoint_path": checkpoint_path.as_posix(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
