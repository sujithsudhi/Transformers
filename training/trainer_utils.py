from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    import wandb
except ImportError:  # wandb is optional; training should continue without it.
    wandb = None

from tool.utils import _to_serializable


def _load_trainer_core_optimizer(module_name : str,
                                 class_name  : str,
                                ) -> Optional[type[torch.optim.Optimizer]]:
    """
    Try to load an optional optimizer class from trainer-core.
    Args:
        module_name : Fully qualified module path inside trainer-core.
        class_name  : Optimizer class name expected in that module.
    Returns:
        Optimizer class when available, otherwise `None`.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None

    optimizer_cls = getattr(module, class_name, None)
    if optimizer_cls is None:
        return None
    return optimizer_cls


def build_optimizer(model        : nn.Module,
                    lr           : float,
                    weight_decay : float,
                    *,
                    name         : str = "adamw",
                    betas        : Optional[Sequence[float]] = None,
                    eps          : Optional[float] = None,
                   ) -> torch.optim.Optimizer:
    """
    Build the configured optimizer for a model.
    Args:
        model        : Model whose parameters will be optimized.
        lr           : Learning rate for the optimizer.
        weight_decay : Weight decay coefficient.
        name         : Optimizer name from config.
        betas        : Optional Adam beta coefficients.
        eps          : Optional numerical stability term for AdamW.
    Returns:
        Configured torch optimizer instance.
    """
    optimizer_name = name.strip().lower()

    optimizer_kwargs: Dict[str, Any] = {
        "lr"          : float(lr),
        "weight_decay": float(weight_decay),
    }
    if betas is not None:
        if len(betas) != 2:
            raise ValueError("betas must contain exactly two values.")
        optimizer_kwargs["betas"] = (float(betas[0]), float(betas[1]))
    if eps is not None:
        optimizer_kwargs["eps"] = float(eps)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    if optimizer_name == "adam":
        optimizer_cls = _load_trainer_core_optimizer("trainer_core.optimizers.adam", "customAdam")
        if optimizer_cls is not None:
            return optimizer_cls(model.parameters(), **optimizer_kwargs)
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)

    if optimizer_name == "lion":
        optimizer_cls = _load_trainer_core_optimizer("trainer_core.optimizers.lion", "customLion")
        if optimizer_cls is None:
            raise ValueError(
                "Unsupported optimizer 'lion'. trainer-core does not expose customLion in this environment."
            )
        lion_kwargs = {"lr"          : float(lr),
                       "weight_decay": float(weight_decay)}
        if betas is not None:
            lion_kwargs["betas"] = (float(betas[0]), float(betas[1]))
        return optimizer_cls(model.parameters(), **lion_kwargs)

    if optimizer_name == "sgd":
        optimizer_cls = _load_trainer_core_optimizer("trainer_core.optimizers.sgd", "customSGD")
        sgd_kwargs = {"lr"          : float(lr),
                      "weight_decay": float(weight_decay)}
        if optimizer_cls is not None:
            return optimizer_cls(model.parameters(), **sgd_kwargs)
        return torch.optim.SGD(model.parameters(), **sgd_kwargs)

    if optimizer_name == "rmsprop":
        optimizer_cls = _load_trainer_core_optimizer("trainer_core.optimizers.rmsprop", "customRMSprop")
        rmsprop_kwargs = {"lr"          : float(lr),
                          "weight_decay": float(weight_decay)}
        if eps is not None:
            rmsprop_kwargs["eps"] = float(eps)
        if optimizer_cls is not None:
            return optimizer_cls(model.parameters(), **rmsprop_kwargs)
        return torch.optim.RMSprop(model.parameters(), **rmsprop_kwargs)

    raise ValueError(
        f"Unsupported optimizer '{name}'. Supported optimizers are AdamW, Adam, Lion, SGD, and RMSprop."
    )


def build_loss(*,
               name : str = "bcewithlogits",
               beta : float = 1.0,
              ) -> nn.Module:
    """
    Build a loss module from a config-friendly loss name.
    Args:
        name : Loss name from config.
        beta : Beta value used by SmoothL1 loss variants.
    Returns:
        Configured torch loss module.
    """
    loss_name = name.strip().lower().replace("_", "").replace("-", "")

    if loss_name in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
        return nn.BCEWithLogitsLoss()
    if loss_name in {"crossentropy", "crossentropyloss"}:
        return nn.CrossEntropyLoss()
    if loss_name in {"mse", "mseloss"}:
        return nn.MSELoss()
    if loss_name in {"smoothl1", "smoothl1loss", "huber", "huberloss"}:
        if beta <= 0:
            raise ValueError("beta must be > 0 for SmoothL1Loss.")
        return nn.SmoothL1Loss(beta=float(beta))

    raise ValueError(
        f"Unsupported loss '{name}'. Supported losses are BCEWithLogits, CrossEntropy, MSE, and SmoothL1."
    )


def build_cross_entropy_loss() -> nn.Module:
    """
    Build the standard language-model cross-entropy loss.
    Returns:
        Cross-entropy loss module for next-token prediction.
    """
    return build_loss(name="crossentropyloss")


def _resolve_wandb_bool_env(name : str) -> Optional[bool]:
    """
    Resolve a boolean environment variable used by wandb configuration.
    Args:
        name : Environment variable name to inspect.
    Returns:
        Parsed boolean value, or `None` when the variable is unset.
    """
    value = os.getenv(name)
    if value is None:
        return None

    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return bool(lowered)


def _resolve_wandb_value(env_name     : str,
                         config_value : Optional[str],
                         default      : Optional[str] = None,
                        ) -> Optional[str]:
    """
    Resolve a wandb setting from environment, config, or default.
    Args:
        env_name     : Environment variable name to check first.
        config_value : Config-provided fallback value.
        default      : Default value used when both other sources are empty.
    Returns:
        Resolved string value or `None`.
    """
    env_value = os.getenv(env_name)
    if env_value not in {None, ""}:
        return env_value
    if config_value not in {None, ""}:
        return config_value
    return default


def init_wandb_run(app_config : Any,
                  ) -> Tuple[Optional["wandb.sdk.wandb_run.Run"], Optional[Callable[[Dict[str, Any]], None]]]:
    """
    Initialize an optional Weights & Biases run and logger callback.
    Args:
        app_config : Top-level application config containing wandb settings.
    Returns:
        Tuple of `(wandb_run, logger_callback)` or `(None, None)` when logging is disabled.
    """
    if wandb is None:
        print("wandb is not installed; skipping experiment logging.")
        return None, None

    disabled_override = _resolve_wandb_bool_env("WANDB_DISABLED")
    if disabled_override is None:
        disabled = bool(getattr(app_config, "wandb_disabled", False))
    else:
        disabled = disabled_override

    if disabled:
        print("Weights & Biases logging disabled; skipping experiment logging.")
        return None, None

    api_key = _resolve_wandb_value("WANDB_API_KEY", getattr(app_config, "wandb_api_key", None))
    if api_key:
        os.environ.setdefault("WANDB_API_KEY", str(api_key))
        try:
            wandb.login(key=str(api_key), relogin=True, anonymous="allow")
        except Exception as exc:  # pragma: no cover - network edge cases
            print(f"Failed to authenticate with Weights & Biases: {exc}")

    project = _resolve_wandb_value(
        "WANDB_PROJECT",
        getattr(app_config, "wandb_project", None),
        default="transformers-imdb",
    )
    entity = _resolve_wandb_value("WANDB_ENTITY", getattr(app_config, "wandb_entity", None))
    run_name = _resolve_wandb_value("WANDB_NAME", getattr(app_config, "wandb_run_name", None))

    run_config: Dict[str, Any] = {
        "data": _to_serializable(app_config.data),
        "model": _to_serializable(app_config.model),
        "training": _to_serializable(app_config.training),
    }
    for section_name in ("dataset", "dataloader", "optimizer", "loss"):
        section_cfg = getattr(app_config, section_name, None)
        if section_cfg is not None:
            run_config[section_name] = _to_serializable(section_cfg)

    run = wandb.init(
        project=project,
        entity=entity,
        config=run_config,
        name=run_name,
    )

    logger = build_wandb_logger(run)
    return run, logger


def build_wandb_logger(run : Optional["wandb.sdk.wandb_run.Run"],
                      ) -> Callable[[Dict[str, Any]], None]:
    """
    Build a trainer callback that logs train and validation metrics to wandb.
    Args:
        run : Active wandb run, if one has been initialized.
    Returns:
        Callback that accepts trainer metric dictionaries.
    """
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


def maybe_save_history(history : list[Dict[str, Any]],
                       path    : Optional[Path],
                      ) -> None:
    """
    Write training history to JSON when an output path is configured.
    Args:
        history : Trainer history entries to serialize.
        path    : Output path for the JSON history file.
    """
    if path is None:
        return
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training history written to {resolved}")


def maybe_plot_history(history : list[Dict[str, Any]],
                       path    : Optional[Path],
                      ) -> None:
    """
    Render training curves when matplotlib and an output path are available.
    Args:
        history : Trainer history entries to visualize.
        path    : Output path for the rendered plot image.
    """
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
                                   non_blocking : bool = True,
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect logits, probabilities, and targets for a classification dataloader.
    Args:
        model        : Classification model to evaluate.
        dataloader   : Dataloader yielding classification batches.
        device       : Device used for forward passes.
        non_blocking : Whether tensor transfers should request non-blocking behavior.
    Returns:
        Tuple of `(logits, probabilities, targets)` tensors concatenated across the dataloader.
    """
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


def prepare_classification_labels(probabilities : torch.Tensor,
                                  targets       : torch.Tensor,
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert probability outputs and targets into discrete label tensors.
    Args:
        probabilities : Tensor of shape (batch_size, num_outputs) or (batch_size, 1).
        targets       : Tensor containing target labels.
    Returns:
        Tuple of `(preds, true)` label tensors flattened for metric computation.
    """
    if probabilities.ndim == 2 and probabilities.shape[1] == 1:
        preds = (probabilities >= 0.5).long().view(-1)
    else:
        preds = probabilities.argmax(dim=-1).view(-1)

    true = targets.view(-1)
    if true.dtype != torch.long:
        true = true.long()
    return preds, true


def compute_class_distribution(labels : torch.Tensor) -> Dict[str, int]:
    """
    Count examples per class label and return a JSON-friendly mapping.
    Args:
        labels : Tensor containing class labels.
    Returns:
        Mapping from class label string to example count.
    """
    flattened = labels.view(-1)
    unique, counts = torch.unique(flattened, return_counts=True)
    return {str(int(idx)): int(count) for idx, count in zip(unique, counts)}


__all__ = [
    "build_optimizer",
    "build_loss",
    "build_cross_entropy_loss",
    "init_wandb_run",
    "build_wandb_logger",
    "maybe_save_history",
    "maybe_plot_history",
    "collect_classification_outputs",
    "prepare_classification_labels",
    "compute_class_distribution",
]
