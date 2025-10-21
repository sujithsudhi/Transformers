from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn

try:
    import wandb
except ImportError:  # wandb is optional; training should continue without it.
    wandb = None

from tool.utils import _to_serializable


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_loss() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def init_wandb_run(app_config: Any) -> Tuple[Optional["wandb.sdk.wandb_run.Run"], Optional[Callable[[Dict[str, Any]], None]]]:
    if wandb is None:
        print("wandb is not installed; skipping experiment logging.")
        return None, None
    if os.getenv("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        print("WANDB_DISABLED detected; skipping experiment logging.")
        return None, None

    api_key = getattr(app_config, "wandb_api_key", None) or os.getenv("WANDB_API_KEY")
    if api_key:
        os.environ.setdefault("WANDB_API_KEY", str(api_key))
        try:
            wandb.login(key=str(api_key), relogin=True, anonymous="allow")
        except Exception as exc:  # pragma: no cover - network edge cases
            print(f"Failed to authenticate with Weights & Biases: {exc}")

    project = getattr(app_config, "wandb_project", None) or os.getenv("WANDB_PROJECT", "transformers-imdb")
    entity = getattr(app_config, "wandb_entity", None) or os.getenv("WANDB_ENTITY")
    run_name = getattr(app_config, "wandb_run_name", None) or os.getenv("WANDB_NAME")

    run_config: Dict[str, Any] = {
        "data": _to_serializable(app_config.data),
        "model": _to_serializable(app_config.model),
        "training": _to_serializable(app_config.training),
    }
    dataset_cfg = getattr(app_config, "dataset", None)
    if dataset_cfg is not None:
        run_config["dataset"] = _to_serializable(dataset_cfg)

    run = wandb.init(
        project=project,
        entity=entity,
        config=run_config,
        name=run_name,
    )

    logger = build_wandb_logger(run)
    return run, logger


def build_wandb_logger(run: Optional["wandb.sdk.wandb_run.Run"]) -> Callable[[Dict[str, Any]], None]:
    if wandb is None or run is None:
        def noop_logger(entry: Dict[str, Any]) -> None:
            return

        return noop_logger

    def log_callback(entry: Dict[str, Any]) -> None:
        payload: Dict[str, Any] = {"epoch": entry["epoch"], "lr": entry.get("lr")}
        train_metrics = entry.get("train") or {}
        payload.update({f"train/{k}": v for k, v in train_metrics.items() if v is not None})
        val_metrics = entry.get("val")
        if val_metrics:
            payload.update({f"val/{k}": v for k, v in val_metrics.items() if v is not None})
        run.log(payload)

    return log_callback


def maybe_save_history(history: list[Dict[str, Any]], path: Optional[Path]) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training history written to {resolved}")


def maybe_plot_history(history: list[Dict[str, Any]], path: Optional[Path]) -> None:
    if path is None or not history:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib is not available; skipping training curve plot.")
        return

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    train_points = [
        (entry.get("epoch"), entry.get("train", {}).get("loss"))
        for entry in history
        if entry.get("train") and entry.get("train", {}).get("loss") is not None
    ]
    val_points = [
        (entry.get("epoch"), entry.get("val", {}).get("loss"))
        for entry in history
        if entry.get("val") and entry.get("val", {}).get("loss") is not None
    ]

    if not train_points and not val_points:
        return

    plt.figure(figsize=(8, 5))
    if train_points:
        epochs, losses = zip(*train_points)
        plt.plot(epochs, losses, marker="o", label="Train loss")
    if val_points:
        epochs, losses = zip(*val_points)
        plt.plot(epochs, losses, marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved)
    plt.close()
    print(f"Training curve plotted to {resolved}")


__all__ = [
    "build_optimizer",
    "build_loss",
    "init_wandb_run",
    "build_wandb_logger",
    "maybe_save_history",
    "maybe_plot_history",
]
