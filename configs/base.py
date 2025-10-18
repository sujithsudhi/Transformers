"""Common configuration dataclasses shared across applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BaseDataConfig:
    """Base dataset configuration with common loader parameters."""

    cache_dir: Path = Path("data/cache")
    batch_size: int = 32
    max_tokens: int = 256
    num_workers: int = 0
    dataset_name: str = "imdb"


@dataclass(frozen=True)
class BaseModelConfig:
    """Base transformer hyperparameters."""

    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_cls_token: bool = True
    cls_head_dim: Optional[int] = 128
    num_outputs: int = 1
    pooling: str = "cls"


@dataclass(frozen=True)
class BaseTrainingConfig:
    """Default training loop configuration."""

    epochs: int = 5
    device: str = "auto"
    lr: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int = 1
    use_amp: str | bool = "auto"
    log_interval: int = 50
    non_blocking: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration aggregating data/model/training."""

    name: str = "base"
    data: BaseDataConfig = field(default_factory=BaseDataConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    training: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    history_path: Path = Path("results/history.json")
    plot_path: Path = Path("results/history.png")

