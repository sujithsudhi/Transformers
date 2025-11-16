from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_imdb_dataloaders

from models import TransformersModel, TransformersModelConfig
from tool.utils import _to_serializable, load_config_target
from training import (TrainingConfig,
                      Trainer,
                      build_loss,
                      build_optimizer,
                      evaluate,
                      init_wandb_run,
                      load_training_config,
                      maybe_plot_history,
                      maybe_save_history)

try:
    import wandb
except ImportError:  # wandb is optional; keep training usable without it.
    wandb = None


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

    wandb_api_key = getattr(app_config, "wandb_api_key", None)

    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = str(wandb_api_key)

    if getattr(app_config, "wandb_disabled", False):
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)
    wandb_run, wandb_logger = init_wandb_run(app_config)

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
    vocab_size = getattr(train_loader.dataset, "vocab_size", None)
    if vocab_size is not None:
        model_kwargs["vocab_size"] = vocab_size
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
                                                      "early_stopping_patience"    : training_cfg.early_stopping_patience,
                                                      "lr_reduction_patience"      : training_cfg.lr_reduction_patience,
                                                      "lr_reduction_factor"        : training_cfg.lr_reduction_factor,
                                                     }
                                                    )

    trainer = Trainer(model        = model,
                      optimizer    = optimizer,
                      loss_fn      = loss_fn,
                      train_loader = train_loader,
                      config       = training_config,
                      val_loader   = test_loader,
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
    best_state = trainer.best_model_state_dict()
    torch.save(
        {
            "model_state_dict"   :  best_state,
            "config": {"model"   : _to_serializable(app_config.model),
                       "training": _to_serializable(app_config.training),
                       "data"    : _to_serializable(app_config.data),
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
    


if __name__ == "__main__":
    main()
