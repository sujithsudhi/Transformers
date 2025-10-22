"""Training utilities and reusable loops."""

from .trainer_utils import (
    build_loss,
    build_optimizer,
    build_wandb_logger,
    collect_classification_outputs,
    compute_class_distribution,
    init_wandb_run,
    maybe_plot_history,
    maybe_save_history,
    prepare_classification_labels,
)
from .training import (
    TrainingConfig,
    Trainer,
    evaluate,
    fit,
    load_training_config,
    train_one_epoch,
)

__all__ = [
    "TrainingConfig",
    "Trainer",
    "train_one_epoch",
    "evaluate",
    "fit",
    "load_training_config",
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
