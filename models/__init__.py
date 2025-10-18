"""Model package exposing transformer architecture and training helpers."""

from .foundation import FoundationModel, FoundationModelConfig
from .training import (
    TrainingConfig,
    Trainer,
    evaluate,
    fit,
    load_training_config,
    train_one_epoch,
)

__all__ = [
    "FoundationModel",
    "FoundationModelConfig",
    "TrainingConfig",
    "Trainer",
    "train_one_epoch",
    "evaluate",
    "fit",
    "load_training_config",
]

