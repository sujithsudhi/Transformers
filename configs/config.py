"""Python-based configuration for dataset, training, and optimisation settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from src.datasets.neuscenes import DatasetConfig as NeuscenesDatasetConfig


def _default_splits() -> Dict[str, float]:
    return {"train": 0.7, "val": 0.2, "test": 0.1}


def _default_model() -> Dict[str, Any]:
    return {
        "input_dim": 3,
        "embed_dim": 256,
        "depth": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "use_cls_token": True,
        "cls_head_dim": 128,
        "num_outputs": 1,
        "pooling": "cls",
    }


@dataclass(frozen=True)
class DatasetSettings:
    """Dataset configuration expressed as Python values instead of JSON."""

    dataset_root: Path = Path("/Users/sujithks/projects/Datasets/Neuscene")
    splits: Dict[str, float] = field(default_factory=_default_splits)
    shuffle: bool = True
    seed: int = 42

    def to_dataset_config(self) -> NeuscenesDatasetConfig:
        """Convert to the runtime DatasetConfig used by loaders."""
        total = sum(self.splits.values())
        if total <= 0:
            raise ValueError("At least one positive dataset split ratio is required.")
        normalized = {name: value / total for name, value in self.splits.items()}
        return NeuscenesDatasetConfig(
            dataset_root=self.dataset_root.expanduser().resolve(),
            splits=normalized,
            shuffle=self.shuffle,
            seed=self.seed,
        )


@dataclass(frozen=True)
class DataloaderSettings:
    """PyTorch dataloader hyperparameters."""

    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": int(self.batch_size),
            "num_workers": int(self.num_workers),
            "pin_memory": bool(self.pin_memory),
        }


@dataclass(frozen=True)
class OptimizerSettings:
    """Torch optimizer configuration."""

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lr": self.lr, "weight_decay": self.weight_decay}


@dataclass(frozen=True)
class LossSettings:
    """Loss function definition."""

    name: str = "mse"

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class TrainingSettings:
    """Training hyperparameters and model settings."""

    epochs: int = 10
    device: str = "auto"
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int = 1
    use_amp: str | bool = "auto"
    log_interval: int = 50
    non_blocking: bool = True
    model: Dict[str, Any] = field(default_factory=_default_model)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "epochs": self.epochs,
            "device": self.device,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_amp": self.use_amp,
            "log_interval": self.log_interval,
            "non_blocking": self.non_blocking,
        }
        payload["model"] = dict(self.model)
        return payload


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration consumed by the training script."""

    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    dataloader: DataloaderSettings = field(default_factory=DataloaderSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    loss: LossSettings = field(default_factory=LossSettings)
    history_output: Path = Path("experiments/training_history.json")

