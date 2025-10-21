"""Train the transformers model using configuration-driven orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scene import (
    DatasetMetadata,
    SceneDatasetConfig,
    SceneRecord,
    load_scene_metadata,
    split_scene_dataset,
)
from models import TransformersModel, TransformersModelConfig
from training import Trainer, evaluate, load_training_config


class SceneFeatureDataset(Dataset):
    """Dataset converting scene aggregates into tensor features."""

    FEATURE_DIM = 3

    def __init__(self, scenes: List[SceneRecord], sensor_index: Dict[str, int]) -> None:
        self.scenes = scenes
        self.sensor_index = sensor_index
        self.num_sensors = len(sensor_index)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scene = self.scenes[idx]
        features = torch.zeros(self.num_sensors, self.FEATURE_DIM, dtype=torch.float32)
        for sensor_name, stats in scene.sensors.items():
            position = self.sensor_index.get(sensor_name)
            if position is None:
                continue
            features[position, 0] = float(stats.file_count)
            features[position, 1] = float(stats.total_size_bytes) / 1_000_000.0
            features[position, 2] = 1.0
        target = features[:, 0].sum().unsqueeze(0)
        return features, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the transformers model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.transformers:TransformersConfig",
        help="Python path to the application config class (e.g. 'configs.transformers:TransformersConfig').",
    )
    return parser.parse_args()


def load_config_target(target: str) -> Any:
    """Import and instantiate a configuration object referenced by string."""
    if not target:
        raise ValueError("Configuration target string cannot be empty.")
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    return attr() if isinstance(attr, type) else attr


def build_sensor_index(metadata: DatasetMetadata) -> Dict[str, int]:
    """Derive a stable sensor index from dataset metadata."""
    names = sorted({sensor for scene in metadata.scenes for sensor in scene.sensors.keys()})
    if not names:
        raise RuntimeError("No sensors discovered in dataset metadata.")
    return {name: idx for idx, name in enumerate(names)}


def build_optimizer(model: nn.Module, spec: Dict[str, Any]) -> torch.optim.Optimizer:
    """Instantiate an optimizer from configuration."""
    name = spec.get("name", "adamw").lower()
    kwargs = {k: v for k, v in spec.items() if k != "name"}
    if "betas" in kwargs and isinstance(kwargs["betas"], list):
        kwargs["betas"] = tuple(kwargs["betas"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer '{name}'.")


def build_loss(spec: Dict[str, Any]) -> nn.Module:
    """Instantiate a loss function from configuration."""
    name = spec.get("name", "mse").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smoothl1":
        beta = float(spec.get("beta", 1.0))
        return nn.SmoothL1Loss(beta=beta)
    raise ValueError(f"Unsupported loss '{name}'.")


def build_dataloaders(
    splits: Dict[str, List[SceneRecord]],
    sensor_index: Dict[str, int],
    dataloader_cfg: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create train/validation/test dataloaders from scene splits."""
    batch_size = int(dataloader_cfg.get("batch_size", 16))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", torch.cuda.is_available()))

    def make_loader(scenes: Iterable[SceneRecord], shuffle: bool) -> Optional[DataLoader]:
        scenes = list(scenes)
        if not scenes:
            return None
        dataset = SceneFeatureDataset(scenes, sensor_index)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    train_scenes = splits.get("train") or next(iter(splits.values()), [])
    train_loader = make_loader(train_scenes, shuffle=True)
    if train_loader is None:
        raise RuntimeError("Training split is empty; cannot proceed.")
    val_loader = make_loader(splits.get("val", []), shuffle=False)
    test_loader = make_loader(splits.get("test", []), shuffle=False)
    return train_loader, val_loader, test_loader


def maybe_save_history(history: List[Dict[str, Any]], path: Optional[Path]) -> None:
    """Persist training history if an output path is provided."""
    if path is None:
        return
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training history written to {resolved}")


def maybe_plot_history(history: List[Dict[str, Any]], path: Optional[Path]) -> None:
    """Render train/validation loss curves and persist them if requested."""
    if path is None or not history:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - matplotlib is optional
        print("matplotlib is not available; skipping training curve plot.")
        return

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    train_points = [
        (entry.get("epoch"), entry.get("train", {}).get("loss"))
        for entry in history
        if entry.get("train") and entry.get("train", {}).get("loss") is not None
    ]
    val_points = [
        (entry.get("epoch"), entry.get("val", {}).get("loss"))
        for entry in history
        if entry.get("val") and entry.get("val", {}).get("loss") is not None
    ]
    if not train_points and not val_points:
        return

    plt.figure(figsize=(8, 5))
    if train_points:
        train_epochs, train_losses = zip(*train_points)
        plt.plot(train_epochs, train_losses, marker="o", label="Train Loss")
    if val_points:
        val_epochs, val_losses = zip(*val_points)
        plt.plot(val_epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved)
    plt.close()
    print(f"Training curve plotted to {resolved}")


def _to_mapping(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "as_dict"):
        return obj.as_dict()  # type: ignore[return-value]
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError("Configuration objects must expose as_dict(), be dataclasses, or mappings.")


def main() -> None:
    args = parse_args()
    app_config_obj = load_config_target(args.config)
    app_config = app_config_obj() if isinstance(app_config_obj, type) else app_config_obj

    required_attrs = ("dataset", "model", "training", "dataloader", "optimizer", "loss")
    missing = [name for name in required_attrs if not hasattr(app_config, name)]
    if missing:
        missing_csv = ", ".join(missing)
        raise AttributeError(
            f"Application config '{type(app_config).__name__}' is missing required attributes: "
            f"{missing_csv}"
        )

    dataset_settings = app_config.dataset
    if isinstance(dataset_settings, type):
        dataset_settings = dataset_settings()
    if hasattr(dataset_settings, "to_dataset_config"):
        dataset_config = dataset_settings.to_dataset_config()
    elif is_dataclass(dataset_settings):
        dataset_config = SceneDatasetConfig(**asdict(dataset_settings))
    elif isinstance(dataset_settings, SceneDatasetConfig):
        dataset_config = dataset_settings
    else:
        raise TypeError("Dataset configuration must be a SceneDatasetConfig instance or compatible.")

    dataset_config = SceneDatasetConfig(
        dataset_root=Path(dataset_config.dataset_root).expanduser().resolve(),
        metadata_file=dataset_config.metadata_file,
        splits=dict(dataset_config.splits),
        shuffle=dataset_config.shuffle,
        seed=dataset_config.seed,
    )

    training_payload = _to_mapping(app_config.training)
    training_config = load_training_config(training_payload)

    metadata = load_scene_metadata(
        dataset_config.dataset_root, metadata_file=dataset_config.metadata_file
    )
    splits = split_scene_dataset(metadata, dataset_config)
    sensor_index = build_sensor_index(metadata)

    dataloader_cfg = _to_mapping(app_config.dataloader)
    train_loader, val_loader, test_loader = build_dataloaders(
        splits,
        sensor_index,
        dataloader_cfg,
    )

    model_settings = asdict(app_config.model)
    model_settings["input_dim"] = model_settings.get("input_dim", SceneFeatureDataset.FEATURE_DIM)
    model_settings.setdefault("num_outputs", 1)
    model_config = TransformersModelConfig(**model_settings)
    model = TransformersModel(model_config)

    optimizer_cfg = _to_mapping(app_config.optimizer)
    optimizer = build_optimizer(model, optimizer_cfg)
    loss_cfg = _to_mapping(app_config.loss)
    loss_fn = build_loss(loss_cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        config=training_config,
        val_loader=val_loader,
    )
    history = trainer.fit()

    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            trainer.model,
            test_loader,
            loss_fn,
            training_config.device,
            training_config.non_blocking,
            progress_desc="Test",
        )

    history_path = getattr(app_config, "history_path", None)
    maybe_save_history(history, history_path)

    plot_path: Optional[Path] = getattr(app_config, "plot_path", None)
    if plot_path is None and history_path is not None:
        plot_path = history_path.with_suffix(".png")
    maybe_plot_history(history, plot_path)

    summary = {"train_history": history, "test_metrics": test_metrics}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
