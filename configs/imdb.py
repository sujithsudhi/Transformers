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
class IMDBDataConfig(BaseDataConfig):
    """Data loading parameters for the IMDB sentiment dataset."""

    data_path    : Path = Path("/media/4TB/Datasets/Basics/IMDB")
    cache_dir    : Path = Path("data/cache/imdb")
    batch_size   : int  = 512
    max_tokens   : int  = 512
    num_workers  : int  = 8
    dataset_name : str  = "imdb"
    dataset_root : Path = Path("data/imdb")
    url_path     : str  = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


@dataclass(frozen=True)
class IMDBModelConfig(BaseModelConfig):
    """TransformersModel hyperparameters tuned for IMDB classification."""

    embed_dim    : int   = 128
    depth        : int   = 4
    num_heads    : int   = 4
    mlp_ratio    : float = 2.0
    cls_head_dim : int   = 128
    num_outputs  : int   = 1
    dropout      : float = 0.1


@dataclass(frozen=True)
class IMDBTrainingConfig(BaseTrainingConfig):
    """Training loop defaults for IMDB sentiment fine-tuning."""

    epochs                  : int        = 300
    lr                      : float      = 3e-4
    weight_decay            : float      = 0.01
    early_stopping_patience : int | None = 10
    lr_reduction_patience   : int | None = 5


@dataclass(frozen=True)
class IMDBDataloaderConfig(BaseDataloaderConfig):
    """Torch DataLoader parameters for IMDB sentences."""

    batch_size  : int  = 32
    num_workers : int  = 2
    pin_memory  : bool = True


@dataclass(frozen=True)
class IMDBOptimizerConfig(BaseOptimizerConfig):
    """Optimizer configuration tuned for binary sentiment targets."""

    name         : str   = "adamw"
    lr           : float = 2e-4
    weight_decay : float = 0.01


@dataclass(frozen=True)
class IMDBLossConfig(BaseLossConfig):
    """Loss specification for binary sentiment classification."""

    name : str = "bcewithlogits"

    def as_dict(self) -> Dict[str, object]:
        return { "name" : self.name}


@dataclass(frozen=True)
class IMDBConfig(AppConfig):
    """Application config describing IMDB sentiment fine-tuning."""

    name            : str                  = "imdb"
    data            : IMDBDataConfig       = field(default_factory=IMDBDataConfig)
    model           : IMDBModelConfig      = field(default_factory=IMDBModelConfig)
    training        : IMDBTrainingConfig   = field(default_factory=IMDBTrainingConfig)
    dataloader      : IMDBDataloaderConfig = field(default_factory=IMDBDataloaderConfig)
    optimizer       : IMDBOptimizerConfig  = field(default_factory=IMDBOptimizerConfig)
    loss            : IMDBLossConfig       = field(default_factory=IMDBLossConfig)
    history_path    : Path                 = Path("results/imdb_history.json")
    plot_path       : Path                 = Path("results/imdb_history.png")
    checkpoint_path : Path                 = Path("results/imdb_transformer.pt")
    wandb_disabled  : bool                 = False
    wandb_project   : str                  = "transformers-imdb"
