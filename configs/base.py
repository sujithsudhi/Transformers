"""Common configuration dataclasses shared across applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


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
class BaseDatasetConfig:
    """Default scene-style dataset configuration."""

    dataset_root: Path = Path("data/scenes")
    metadata_file: str = "metadata.json"
    splits: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    shuffle: bool = True
    seed: int = 42

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "metadata_file": self.metadata_file,
            "splits": dict(self.splits),
            "shuffle": self.shuffle,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class BaseDataloaderConfig:
    """Default DataLoader parameters."""

    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }


@dataclass(frozen=True)
class BaseOptimizerConfig:
    """Default optimizer configuration."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
        }


@dataclass(frozen=True)
class BaseLossConfig:
    """Default loss specification."""

    name: str = "mse"
    beta: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "beta": self.beta}


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration aggregating data/model/training."""

    name: str = "base"
    data: BaseDataConfig = field(default_factory=BaseDataConfig)
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    training: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    dataset: BaseDatasetConfig = field(default_factory=BaseDatasetConfig)
    dataloader: BaseDataloaderConfig = field(default_factory=BaseDataloaderConfig)
    optimizer: BaseOptimizerConfig = field(default_factory=BaseOptimizerConfig)
    loss: BaseLossConfig = field(default_factory=BaseLossConfig)
    history_path: Path = Path("results/history.json")
    plot_path: Path = Path("results/history.png")
