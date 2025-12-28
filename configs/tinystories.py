"""IMDB sentiment-classification configuration leveraging shared defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from .base import (AppConfig,
                   BaseDataConfig,
                   BaseDataloaderConfig,
                   BaseLossConfig,
                   BaseModelConfig,
                   BaseOptimizerConfig,
                   BaseTrainingConfig,
                  )


@dataclass(frozen=True)
class TinyStoriesDataConfig(BaseDataConfig):
    """Data loading parameters for the IMDB sentiment dataset."""

    data_path    : Path = Path("/media/4TB/Datasets/Basics/IMDB")
    cache_dir    : Path = Path("data/cache/imdb")
    batch_size   : int  = 1024
    max_tokens   : int  = 512
    num_workers  : int  = 8
    dataset_name : str  = "tinystories"
    dataset_root : Path = Path("data/tinystories")
    url_path     : str  = None


@dataclass(frozen=True)
class TinyStoriesModelConfig(BaseModelConfig):
    """ClassifierModel hyperparameters tuned for IMDB classification."""

    embed_dim    : int   = 512
    depth        : int   = 4
    num_heads    : int   = 8
    mlp_ratio    : float = 2.0
    dropout      : float = 0.1
    max_length   : int   = 512
    input_dim    : int   = 0
    vocab_size   : int   = 0


@dataclass(frozen=True)
class TinyStoriesTrainingConfig(BaseTrainingConfig):
    """Training loop defaults for IMDB sentiment fine-tuning."""

    epochs                  : int        = 300
    lr                      : float      = 3e-4
    weight_decay            : float      = 0.01
    early_stopping_patience : int | None = 10
    lr_reduction_patience   : int | None = 5


@dataclass(frozen=True)
class TinyStoriesDataloaderConfig(BaseDataloaderConfig):
    """Torch DataLoader parameters for IMDB sentences."""

    batch_size  : int  = 32
    num_workers : int  = 4
    pin_memory  : bool = True


@dataclass(frozen=True)
class TinyStoriesOptimizerConfig(BaseOptimizerConfig):
    """Optimizer configuration tuned for binary sentiment targets."""

    name         : str   = "adamw"
    lr           : float = 3e-4
    weight_decay : float = 0.1
    betas        : float = (0.9, 0.95)
    eps          : float = 1e-8


@dataclass(frozen=True)
class TinyStoriesLossConfig(BaseLossConfig):
    """Loss specification for binary sentiment classification."""

    name : str = "crossentropyloss"

    def as_dict(self) -> Dict[str, object]:
        return { "name" : self.name}


@dataclass(frozen=True)
class TinyStoriesConfig(AppConfig):
    """Application config describing IMDB sentiment fine-tuning."""

    name            : str                  = "tinystories"
    data            : TinyStoriesDataConfig       = field(default_factory=TinyStoriesDataConfig)
    model           : TinyStoriesModelConfig      = field(default_factory=TinyStoriesModelConfig)
    training        : TinyStoriesTrainingConfig   = field(default_factory=TinyStoriesTrainingConfig)
    dataloader      : TinyStoriesDataloaderConfig = field(default_factory=TinyStoriesDataloaderConfig)
    optimizer       : TinyStoriesOptimizerConfig  = field(default_factory=TinyStoriesOptimizerConfig)
    loss            : TinyStoriesLossConfig       = field(default_factory=TinyStoriesLossConfig)
    history_path    : Path                 = Path("results/tiny_stories_history.json")
    plot_path       : Path                 = Path("results/tiny_stories_history.png")
    checkpoint_path : Path                 = Path("results/tiny_stories_transformer.pt")
    wandb_disabled  : bool                 = True
    wandb_project   : str                  = "transformers-tinystories"
    wandb_run_name  : str                  = "Custom Layers"
