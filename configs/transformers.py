"""Generic transformer configuration dataclasses used across scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import (AppConfig,
                   BaseDataConfig,
                   BaseDataloaderConfig,
                   BaseLossConfig,
                   BaseModelConfig,
                   BaseOptimizerConfig,
                   BaseTrainingConfig,
                  )


@dataclass(frozen=True)
class TransformersDataConfig(BaseDataConfig):
    """Shared data defaults for transformer-based applications."""

    dataset_root : Path = Path("data")
    download     : bool = True


@dataclass(frozen=True)
class TransformersModelConfig(BaseModelConfig):
    """Shared model fields expected by training/export/inference scripts."""

    max_length : int = 256
    input_dim  : int = 0
    vocab_size : int = 0


@dataclass(frozen=True)
class TransformersTrainingConfig(BaseTrainingConfig):
    """Shared training defaults for transformer applications."""


@dataclass(frozen=True)
class TransformersDataloaderConfig(BaseDataloaderConfig):
    """Shared dataloader defaults for transformer applications."""


@dataclass(frozen=True)
class TransformersOptimizerConfig(BaseOptimizerConfig):
    """Shared optimizer defaults for transformer applications."""


@dataclass(frozen=True)
class TransformersLossConfig(BaseLossConfig):
    """Shared loss defaults for transformer applications."""


@dataclass(frozen=True)
class TransformersConfig(AppConfig):
    """Top-level generic transformer application config."""

    name       : str                          = "transformers"
    data       : TransformersDataConfig       = field(default_factory=TransformersDataConfig)
    model      : TransformersModelConfig      = field(default_factory=TransformersModelConfig)
    training   : TransformersTrainingConfig   = field(default_factory=TransformersTrainingConfig)
    dataloader : TransformersDataloaderConfig = field(default_factory=TransformersDataloaderConfig)
    optimizer  : TransformersOptimizerConfig  = field(default_factory=TransformersOptimizerConfig)
    loss       : TransformersLossConfig       = field(default_factory=TransformersLossConfig)
