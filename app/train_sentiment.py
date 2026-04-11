from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import IMDBDataPrep
from models import ClassifierModel
from tool.utils import _to_serializable, load_config_target
from training import (Trainer,
                      TrainingConfig,
                      build_loss,
                      build_optimizer,
                      collect_classification_outputs,
                      compute_class_distribution,
                      evaluate,
                      init_wandb_run,
                      load_training_config,
                      maybe_plot_history,
                      maybe_save_history,
                      prepare_classification_labels,
                     )
from viz import plot_class_distribution, plot_confusion_matrix

try:
    import wandb
except ImportError:  # wandb is optional; keep training usable without it.
    wandb = None

CONFIG_TARGET = "configs.imdb:IMDBConfig"


def _build_training_config(training_cfg: Any) -> TrainingConfig:
    """
    Build the trainer-core configuration from the app training section.
    Args:
        training_cfg : Training config section from the application config.
    Returns:
        Trainer-core configuration object for the training loop.
    """
    return load_training_config({"epochs"                     : training_cfg.epochs,
                                 "device"                     : training_cfg.device,
                                 "gradient_clip_norm"         : training_cfg.gradient_clip_norm,
                                 "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
                                 "use_amp"                    : training_cfg.use_amp,
                                 "amp_dtype"                  : training_cfg.amp_dtype,
                                 "log_interval"               : training_cfg.log_interval,
                                 "non_blocking"               : training_cfg.non_blocking,
                                 "early_stopping_patience"    : training_cfg.early_stopping_patience,
                                 "lr_reduction_patience"      : training_cfg.lr_reduction_patience,
                                 "lr_reduction_factor"        : training_cfg.lr_reduction_factor,
                                })


def _build_checkpoint_config(app_config   : Any,
                             model_config : Any,
                            ) -> Dict[str, Any]:
    """
    Build the checkpoint config payload for serialization.
    Args:
        app_config   : Top-level application config object.
        model_config : Resolved model config used to build the classifier.
    Returns:
        JSON-serializable checkpoint config dictionary.
    """
    checkpoint_config = {"model"    : _to_serializable(model_config),
                         "training" : _to_serializable(app_config.training),
                         "data"     : _to_serializable(app_config.data),
                        }
    for section_name in ("dataloader", "optimizer", "loss"):
        section_value = getattr(app_config, section_name, None)
        if section_value is not None:
            checkpoint_config[section_name] = _to_serializable(section_value)
    return checkpoint_config


def main() -> None:
    """
    Train and evaluate the IMDB sentiment classifier from config.
    """
    # Loading application config
    app_config = load_config_target(CONFIG_TARGET)
    if not hasattr(app_config, "data") or not hasattr(app_config, "model") or not hasattr(app_config, "training"):
        raise TypeError("Configuration object must expose 'data', 'model', and 'training'.")

    # Runtime setup
    torch.manual_seed(int(getattr(app_config.training, "seed", 42)))
    wandb_run, wandb_logger = init_wandb_run(app_config)

    # Preparing data for training
    data_cfg = app_config.data
    dataloader_cfg = getattr(app_config, "dataloader", None)
    loader_batch_size = getattr(dataloader_cfg, "batch_size", 32)
    loader_num_workers = getattr(dataloader_cfg, "num_workers", 0)
    loader_pin_memory = getattr(dataloader_cfg, "pin_memory", True)

    data_prep = IMDBDataPrep(data_path      = data_cfg.data_path,
                             batch_size     = loader_batch_size,
                             max_tokens     = data_cfg.max_tokens,
                             num_workers    = loader_num_workers,
                             url_path       = data_cfg.url_path,
                             tokenizer_name = getattr(data_cfg, "tokenizer_name", "bert-base-uncased"),
                             pin_memory     = loader_pin_memory,
                             download       = getattr(data_cfg, "download", True),
                            )

    train_loader, test_loader = data_prep.prep()

    feature_dim = train_loader.dataset.feature_dim  # type: ignore[attr-defined]
    model_kwargs = asdict(app_config.model)
    model_kwargs["input_dim"] = feature_dim

    vocab_size = getattr(train_loader.dataset, "vocab_size", None)
    if vocab_size is not None:
        model_kwargs["vocab_size"] = vocab_size

    model_config = replace(app_config.model, **model_kwargs)
    model = ClassifierModel(model_config)

    # Building optimizer and loss
    training_cfg = app_config.training
    optimizer_cfg = getattr(app_config, "optimizer", None)
    optimizer = build_optimizer(model,
                                lr           = getattr(optimizer_cfg, "lr", training_cfg.lr),
                                weight_decay = getattr(optimizer_cfg, "weight_decay", training_cfg.weight_decay),
                                name         = getattr(optimizer_cfg, "name", "adamw"),
                                betas        = getattr(optimizer_cfg, "betas", None),
                                eps          = getattr(optimizer_cfg, "eps", None),
                               )

    loss_cfg = getattr(app_config, "loss", None)
    loss_fn = build_loss(name = getattr(loss_cfg, "name", "bcewithlogits"),
                         beta = getattr(loss_cfg, "beta", 1.0),
                        )
    training_config = _build_training_config(training_cfg)

    checkpoint_path = Path(getattr(app_config, "checkpoint_path", Path("results/imdb_transformer.pt")))
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint_path.parent.mkdir(parents = True, exist_ok = True)

    trainer = Trainer(model               = model,
                      optimizer           = optimizer,
                      loss_fn             = loss_fn,
                      train_loader        = train_loader,
                      config              = training_config,
                      val_loader          = test_loader,
                      logger              = wandb_logger,
                      best_checkpoint_dir = checkpoint_path.parent,
                     )
    history = trainer.fit()

    # Evaluation
    device = torch.device(training_config.device)
    test_metrics = evaluate(trainer.model,
                            test_loader,
                            loss_fn,
                            device,
                            training_config.non_blocking,
                            progress_desc = "Test",
                           )

    history_path = getattr(app_config, "history_path", None)
    plot_path = getattr(app_config, "plot_path", None)
    maybe_save_history(history, history_path)
    maybe_plot_history(history, plot_path)

    # Saving checkpoint and artifacts
    _, probabilities, targets = collect_classification_outputs(trainer.model,
                                                               test_loader,
                                                               device,
                                                               non_blocking = training_config.non_blocking,
                                                              )
    preds, true_labels = prepare_classification_labels(probabilities, targets)
    total_examples = max(1, true_labels.numel())
    accuracy = float((preds == true_labels).sum().item()) / total_examples
    test_metrics["accuracy"] = accuracy
    test_metrics["examples"] = int(total_examples)

    best_state = trainer.best_model_state_dict()
    torch.save({"model_state_dict": best_state,
                "config"          : _build_checkpoint_config(app_config, model_config),
               },
               checkpoint_path)

    default_results_dir = checkpoint_path.parent
    if plot_path is not None:
        default_results_dir = Path(plot_path).expanduser().resolve().parent
    elif history_path is not None:
        default_results_dir = Path(history_path).expanduser().resolve().parent

    confusion_path = Path(getattr(app_config, "confusion_matrix_path", default_results_dir / "imdb_confusion_matrix.png"))
    distribution_path = Path(getattr(app_config, "class_distribution_path", default_results_dir / "imdb_class_distribution.png"))
    confusion_path = confusion_path.expanduser().resolve()
    distribution_path = distribution_path.expanduser().resolve()
    confusion_path.parent.mkdir(parents = True, exist_ok = True)
    distribution_path.parent.mkdir(parents = True, exist_ok = True)

    label_names: Optional[list[str]] = None
    if hasattr(app_config, "class_labels"):
        label_names = list(getattr(app_config, "class_labels"))

    confusion_fig, _ = plot_confusion_matrix(y_true     = true_labels.numpy(),
                                             y_pred     = preds.numpy(),
                                             labels     = label_names,
                                             normalize  = True,
                                             title      = "Test Confusion Matrix",
                                            )
    confusion_fig.savefig(confusion_path, dpi = 200)

    class_fig, _ = plot_class_distribution(labels      = true_labels.numpy(),
                                           label_names = label_names,
                                           title       = "Test Class Distribution",
                                          )
    class_fig.savefig(distribution_path, dpi = 200)

    class_counts = compute_class_distribution(true_labels)

    if wandb_run is not None and wandb is not None:
        wandb.log({f"test/{key}": value for key, value in test_metrics.items()})
        wandb_run.summary.update({f"test/{key}": value for key, value in test_metrics.items()})

        artifact = wandb.Artifact("transformer-imdb-model", type="model")
        artifact.add_file(checkpoint_path.as_posix())

        if history_path is not None:
            resolved_history = Path(history_path).expanduser().resolve()
            if resolved_history.exists():
                artifact.add_file(resolved_history.as_posix())

        if plot_path is not None:
            resolved_plot = Path(plot_path).expanduser().resolve()
            if resolved_plot.exists():
                wandb.log({"plots/loss": wandb.Image(resolved_plot.as_posix())})

        if confusion_path.exists():
            wandb.log({"plots/confusion_matrix": wandb.Image(confusion_path.as_posix())})
        if distribution_path.exists():
            wandb.log({"plots/class_distribution": wandb.Image(distribution_path.as_posix())})

        artifact.add_file(confusion_path.as_posix())
        artifact.add_file(distribution_path.as_posix())
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    # Final summary
    summary: Dict[str, Any] = {"train_history"           : history,
                               "test_metrics"            : test_metrics,
                               "class_distribution"      : class_counts,
                               "confusion_matrix_path"   : confusion_path.as_posix(),
                               "class_distribution_path" : distribution_path.as_posix(),
                              }
    print(json.dumps(summary, indent = 2))


if __name__ == "__main__":
    main()
