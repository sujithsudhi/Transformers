"""Compatibility shim that re-exports the shared trainer engine."""

from trainer_core.engine import Trainer, evaluate, fit, train_one_epoch
from trainer_core.config import TrainingConfig, load_training_config

__all__ = [
    "TrainingConfig",
    "Trainer",
    "train_one_epoch",
    "evaluate",
    "fit",
    "load_training_config",
]
