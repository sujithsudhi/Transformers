"""IMDB-specific configuration classes inheriting shared defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import AppConfig, BaseDataConfig, BaseModelConfig, BaseTrainingConfig


@dataclass(frozen=True)
class IMDBDataConfig(BaseDataConfig):
    cache_dir: Path = Path("data/cache/imdb")
    batch_size: int = 32
    max_tokens: int = 256
    num_workers: int = 0
    dataset_name: str = "imdb"


@dataclass(frozen=True)
class IMDBModelConfig(BaseModelConfig):
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    cls_head_dim: int = 128


@dataclass(frozen=True)
class IMDBTrainingConfig(BaseTrainingConfig):
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True)
class IMDBConfig(AppConfig):
    name: str = "imdb"
    data: IMDBDataConfig = field(default_factory=IMDBDataConfig)
    model: IMDBModelConfig = field(default_factory=IMDBModelConfig)
    training: IMDBTrainingConfig = field(default_factory=IMDBTrainingConfig)
    history_path: Path = Path("results/imdb_history.json")
    plot_path: Path = Path("results/imdb_history.png")

