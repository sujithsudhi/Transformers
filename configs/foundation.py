"""Example configuration for a generic foundation-model application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import AppConfig, BaseDataConfig, BaseModelConfig, BaseTrainingConfig


@dataclass(frozen=True)
class FoundationDataConfig(BaseDataConfig):
    dataset_name: str = "neuscenes"
    dataset_root: Path = Path("/path/to/neuscenes")
    cache_dir: Path = Path("data/cache/neuscenes")
    batch_size: int = 16
    max_tokens: int = 256


@dataclass(frozen=True)
class FoundationModelConfig(BaseModelConfig):
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    cls_head_dim: int = 128


@dataclass(frozen=True)
class FoundationTrainingConfig(BaseTrainingConfig):
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01


@dataclass(frozen=True)
class FoundationConfig(AppConfig):
    name: str = "foundation"
    data: FoundationDataConfig = field(default_factory=FoundationDataConfig)
    model: FoundationModelConfig = field(default_factory=FoundationModelConfig)
    training: FoundationTrainingConfig = field(default_factory=FoundationTrainingConfig)
    history_path: Path = Path("results/foundation_history.json")
    plot_path: Path = Path("results/foundation_history.png")

