"""Example configuration for a generic transformers-model application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from scene_data.scene import SceneDatasetConfig

from .base import (
    AppConfig,
    BaseDataConfig,
    BaseDataloaderConfig,
    BaseLossConfig,
    BaseModelConfig,
    BaseOptimizerConfig,
    BaseTrainingConfig,
)


@dataclass(frozen=True)
class TransformersDataConfig(BaseDataConfig):
    """High-level dataset parameters shared with other training scripts."""

    dataset_name : str  = "scene"
    dataset_root : Path = Path("data/scenes")
    cache_dir    : Path = Path("data/cache/scenes")
    batch_size   : int  = 16
    max_tokens   : int  = 256
    num_workers  : int  = 0


@dataclass(frozen=True)
class TransformersModelConfig(BaseModelConfig):
    """Transformer hyperparameters tailored to the scene dataset."""

    embed_dim    : int   = 256
    depth        : int   = 6
    num_heads    : int   = 8
    mlp_ratio    : float = 4.0
    cls_head_dim : int   = 128


@dataclass(frozen=True)
class TransformersTrainingConfig(BaseTrainingConfig):
    """Training loop defaults for the transformers scene workflow."""

    epochs       : int   = 10
    lr           : float = 3e-4
    weight_decay : float = 0.01


@dataclass(frozen=True)
class TransformersDatasetConfig(SceneDatasetConfig):
    """Scene dataset configuration with project-specific defaults."""

    dataset_root  : Path             = Path("data/scenes")
    metadata_file : str              = "metadata.json"
    splits        : Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    shuffle       : bool             = True
    seed          : int              = 42


@dataclass(frozen=True)
class TransformersDataloaderConfig(BaseDataloaderConfig):
    """Torch DataLoader parameters controlling batching behaviour."""

    batch_size  : int  = 16
    num_workers : int  = 0
    pin_memory  : bool = True


@dataclass(frozen=True)
class TransformersOptimizerConfig(BaseOptimizerConfig):
    """Optimizer specification leveraged by the training script."""

    name         : str                 = "adamw"
    lr           : float               = 3e-4
    weight_decay : float               = 0.01
    betas        : tuple[float, float] = (0.9, 0.999)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name"         : self.name,
            "lr"           : self.lr,
            "weight_decay" : self.weight_decay,
            "betas"        : self.betas,
        }


@dataclass(frozen=True)
class TransformersLossConfig(BaseLossConfig):
    """Loss specification looked up by the training script."""

    name : str   = "mse"
    beta : float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "name" : self.name,
            "beta" : self.beta,
        }


@dataclass(frozen=True)
class TransformersConfig(AppConfig):
    """Top-level application configuration used by train_transformers.py."""

    name         : str                           = "transformers"
    data         : TransformersDataConfig        = field(default_factory=TransformersDataConfig)
    model        : TransformersModelConfig       = field(default_factory=TransformersModelConfig)
    training     : TransformersTrainingConfig    = field(default_factory=TransformersTrainingConfig)
    dataset      : TransformersDatasetConfig     = field(default_factory=TransformersDatasetConfig)
    dataloader   : TransformersDataloaderConfig  = field(default_factory=TransformersDataloaderConfig)
    optimizer    : TransformersOptimizerConfig   = field(default_factory=TransformersOptimizerConfig)
    loss         : TransformersLossConfig        = field(default_factory=TransformersLossConfig)
    history_path : Path                          = Path("results/transformers_history.json")
    plot_path    : Path                          = Path("results/transformers_history.png")
    checkpoint_path: Path                        = Path("results/transformers_model.pt")
