from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from data import build_imdb_dataloaders

from models import (TransformersModel,
                    TransformersModelConfig,
                    TrainingConfig,
                    Trainer,
                    evaluate,
                    load_training_config,)

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


# Function: build_optimizer
# Description: Construct an AdamW optimizer applying provided hyperparameters.
def build_optimizer(model       : nn.Module,
                    lr          : float,
                    weight_decay: float,
                   ) -> torch.optim.Optimizer:
    
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


''' Function: build_loss
    Description: Return the BCEWithLogitsLoss for binary sentiment targets.
    Args:
        None
    Returns:
        BCEWithLogitsLoss module for binary classification.
'''
def build_loss() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def _init_wandb_run(app_config: Any) -> tuple[Optional["wandb.sdk.wandb_run.Run"], Optional[Any]]:
    if wandb is None:
        return None, None
    if os.getenv("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None, None
    project = os.getenv("WANDB_PROJECT", "transformers-imdb")
    entity = os.getenv("WANDB_ENTITY")
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
        name=os.getenv("WANDB_NAME"),
    )

    def log_callback(entry: Dict[str, Any]) -> None:
        payload: Dict[str, Any] = {"epoch": entry["epoch"], "lr": entry.get("lr")}
        train_metrics = entry.get("train") or {}
        payload.update({f"train/{k}": v for k, v in train_metrics.items() if v is not None})
        val_metrics = entry.get("val")
        if val_metrics:
            payload.update({f"val/{k}": v for k, v in val_metrics.items() if v is not None})
        wandb.log(payload)

    return run, log_callback


''' Function: maybe_save_history
    Description: Persist training history JSON if an output path is provided.
    Args:
        history : List of training history dictionaries per epoch.
        path    : Optional file path for saving JSON.
    Returns:
        None
'''
def maybe_save_history(history: list[Dict[str, Any]], path: Optional[Path]) -> None:
    if path is None:
        return
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training history written to {resolved}")


''' Function: maybe_plot_history
    Description: Render training/validation loss curves when matplotlib is available.
    Args:
        history : List of training history dictionaries.
        path    : Optional file path for saving plot image.
    Returns:
        None
'''
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


''' Function: load_config_target
    Description: Import and instantiate a configuration object referenced by string.
    Args:
        target : Python module path with attribute (e.g., 'module:ClassName').
    Returns:
        Loaded configuration object or instance.
'''
def load_config_target(target: str) -> Any:
    if not target:
        raise ValueError("Configuration target string cannot be empty.")
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    if isinstance(attr, type):
        return attr()
    return attr


''' Function: main
    Description: Orchestrate configuration loading, dataloader creation, and training.
    Args:
        None
    Returns:
        None
'''
def main() -> None:
    torch.manual_seed(42)

    app_config = load_config_target("configs.imdb:IMDBConfig")
    wandb_run, wandb_logger = _init_wandb_run(app_config)

    # Validate configuration contract before applying overrides.
    if not hasattr(app_config, "data") or not hasattr(app_config, "model"):
        raise TypeError("Configuration object must expose 'data', 'model', and 'training'.")

    data_cfg = app_config.data

    train_loader, test_loader = build_imdb_dataloaders(batch_size    = data_cfg.batch_size,
                                                       max_tokens    = data_cfg.max_tokens,
                                                       num_workers   = data_cfg.num_workers,
                                                       cache_dir     = data_cfg.cache_dir,
                                                       dataset_name  = getattr(data_cfg, "dataset_name", "imdb"),
                                                       dataset_root  = getattr(data_cfg, "dataset_root", Path("data/imdb")),
                                                      )
    # Extract feature dimension from dataset to configure projection layer.
    feature_dim = train_loader.dataset.feature_dim  # type: ignore[attr-defined]

    model_kwargs              = asdict(app_config.model)
    model_kwargs["input_dim"] = feature_dim
    model_config              = TransformersModelConfig(**model_kwargs)

    # Instantiate transformer backbone with resolved configuration.
    model                     = TransformersModel(model_config)

    training_cfg              = app_config.training

    optimizer                 = build_optimizer(model, lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)
    loss_fn                   = build_loss()

    training_config           = load_training_config({"epochs"                     : training_cfg.epochs,
                                                      "device"                     : training_cfg.device,
                                                      "gradient_clip_norm"         : training_cfg.gradient_clip_norm,
                                                      "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
                                                      "use_amp"                    : training_cfg.use_amp,
                                                      "log_interval"               : training_cfg.log_interval,
                                                      "non_blocking"               : training_cfg.non_blocking,
                                                     }
                                                    )

    trainer = Trainer(model        = model,
                      optimizer    = optimizer,
                      loss_fn      = loss_fn,
                      train_loader = train_loader,
                      config       = training_config,
                      val_loader   = None,
                      logger       = wandb_logger,
                     )
    history = trainer.fit()

    test_metrics = evaluate(trainer.model,
                            test_loader,
                            loss_fn,
                            training_config.device,
                            training_config.non_blocking,
                            progress_desc="Test",
                           )

    history_path   = getattr(app_config, "history_path", None)
    plot_path      = getattr(app_config, "plot_path", None)
    checkpoint_path = Path(getattr(app_config, "checkpoint_path", Path("results/model.pt")))

    maybe_save_history(history, history_path)
    maybe_plot_history(history, plot_path)

    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": {
                "model": _to_serializable(app_config.model),
                "training": _to_serializable(app_config.training),
                "data": _to_serializable(app_config.data),
            },
        },
        checkpoint_path,
    )

    if wandb_run is not None:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wandb_run.summary.update({f"test/{k}": v for k, v in test_metrics.items()})

        artifact = wandb.Artifact("transformer-imdb-model", type="model")
        artifact.add_file(checkpoint_path.as_posix())
        if history_path is not None:
            resolved_history = history_path.expanduser().resolve()
            if resolved_history.exists():
                artifact.add_file(resolved_history.as_posix())
        if plot_path is not None:
            resolved_plot = plot_path.expanduser().resolve()
            if resolved_plot.exists():
                wandb.log({"plots/loss": wandb.Image(resolved_plot.as_posix())})
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    summary: Dict[str, Any] = {"train_history": history, "test_metrics": test_metrics}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
