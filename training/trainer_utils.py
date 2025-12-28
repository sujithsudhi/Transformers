from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Mapping

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

def build_cross_entropy_loss() -> nn.Module:
    return nn.CrossEntropyLoss()


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

    val_accuracy_points = [
        (entry.get("epoch"), entry.get("val", {}).get("accuracy"))
        for entry in history
        if entry.get("val") and entry.get("val", {}).get("accuracy") is not None
    ]
    train_accuracy_points = [
        (entry.get("epoch"), entry.get("train", {}).get("accuracy"))
        for entry in history
        if entry.get("train") and entry.get("train", {}).get("accuracy") is not None
    ]

    has_accuracy = bool(train_accuracy_points or val_accuracy_points)
    num_rows = 2 if has_accuracy else 1
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, 4 * num_rows), sharex=True)
    if num_rows == 1:
        axes = [axes]
    loss_ax = axes[0]

    train_epochs, train_losses = [], []
    if train_points:
        train_epochs, train_losses = zip(*train_points)
        loss_ax.plot(
            train_epochs,
            train_losses,
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="Train loss",
        )

    val_epochs, val_losses = [], []
    if val_points:
        val_epochs, val_losses = zip(*val_points)
        loss_ax.plot(
            val_epochs,
            val_losses,
            marker="s",
            linewidth=2.0,
            color="#d62728",
            label="Validation loss",
        )
        best_idx = int(min(range(len(val_losses)), key=lambda idx: val_losses[idx]))
        best_epoch = val_epochs[best_idx]
        best_loss = val_losses[best_idx]
        loss_ax.scatter(
            [best_epoch],
            [best_loss],
            color="#d62728",
            marker="x",
            s=80,
            zorder=5,
        )
        loss_ax.annotate(
            f"best val {best_loss:.4f}",
            xy=(best_epoch, best_loss),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=9,
            color="#d62728",
        )

    if train_points and val_points:
        train_map = {epoch: loss for epoch, loss in train_points}
        val_map = {epoch: loss for epoch, loss in val_points}
        shared_epochs = sorted(set(train_map).intersection(val_map))
        if shared_epochs:
            loss_ax.fill_between(
                shared_epochs,
                [train_map[epoch] for epoch in shared_epochs],
                [val_map[epoch] for epoch in shared_epochs],
                color="#9ecae1",
                alpha=0.2,
                label="Train-Val gap",
            )

    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training / Validation Loss")
    loss_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    loss_ax.legend()

    if has_accuracy:
        acc_ax = axes[1]
        if train_accuracy_points:
            epochs_acc, acc_values = zip(*train_accuracy_points)
            acc_ax.plot(
                epochs_acc,
                acc_values,
                marker="o",
                linewidth=2.0,
                color="#2ca02c",
                label="Train accuracy",
            )
        if val_accuracy_points:
            epochs_acc, acc_values = zip(*val_accuracy_points)
            acc_ax.plot(
                epochs_acc,
                acc_values,
                marker="s",
                linewidth=2.0,
                color="#ff7f0e",
                label="Validation accuracy",
            )
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.set_title("Training / Validation Accuracy")
        acc_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        acc_ax.legend()
    else:
        loss_ax.set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(resolved)
    plt.close()
    print(f"Training curve plotted to {resolved}")


def collect_classification_outputs(model: nn.Module,
                                   dataloader,
                                   device: torch.device,
                                   *,
                                   non_blocking: bool = True,
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    def _move_to_device(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=non_blocking)
        if isinstance(obj, Mapping):
            return {k: _move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_move_to_device(v) for v in obj)
        return obj

    def _split_batch(batch: Any) -> tuple[Any, Any]:
        if isinstance(batch, Mapping):
            if "inputs" in batch and "targets" in batch:
                return batch["inputs"], batch["targets"]
            if "x" in batch and "y" in batch:
                return batch["x"], batch["y"]
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError("Unsupported batch structure; expected (inputs, targets).")

    model = model.to(device)
    model.eval()

    logits_list: list[torch.Tensor] = []
    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=-1)

    with torch.no_grad():
        for raw_batch in dataloader:
            batch_inputs, targets = _split_batch(_move_to_device(raw_batch))
            targets = targets if isinstance(targets, torch.Tensor) else targets
            if targets.ndim > 1 and targets.shape[-1] == 1:
                targets = targets.view(targets.size(0), -1)

            if isinstance(batch_inputs, Mapping):
                logits = model(**batch_inputs)
            elif isinstance(batch_inputs, (list, tuple)):
                logits = model(*batch_inputs)
            else:
                logits = model(batch_inputs)
            if logits.ndim == 2 and logits.shape[1] == 1:
                probs = sigmoid(logits)
            else:
                probs = softmax(logits)

            logits_list.append(logits.detach().cpu())
            probs_list.append(probs.detach().cpu())
            targets_list.append(targets.detach().cpu())

    logits_tensor = torch.cat(logits_list, dim=0) if logits_list else torch.empty(0)
    probs_tensor = torch.cat(probs_list, dim=0) if probs_list else torch.empty(0)
    targets_tensor = torch.cat(targets_list, dim=0) if targets_list else torch.empty(0)
    return logits_tensor, probs_tensor, targets_tensor


def prepare_classification_labels(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        preds = (probabilities >= 0.5).long().view(-1)
    else:
        preds = probabilities.argmax(dim=-1).view(-1)

    true = targets.view(-1)
    if true.dtype != torch.long:
        true = true.long()
    return preds, true


def compute_class_distribution(labels: torch.Tensor) -> Dict[str, int]:
    flattened = labels.view(-1)
    unique, counts = torch.unique(flattened, return_counts=True)
    return {str(int(idx)): int(count) for idx, count in zip(unique, counts)}


__all__ = [
    "build_optimizer",
    "build_loss",
    "init_wandb_run",
    "build_wandb_logger",
    "maybe_save_history",
    "maybe_plot_history",
    "collect_classification_outputs",
    "prepare_classification_labels",
    "compute_class_distribution",
]
