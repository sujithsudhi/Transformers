"""Train the Neuscenes foundation model using configuration-driven orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import Trainer, evaluate, load_training_config
from src.datasets import load_neuscenes_metadata, split_neuscenes
from src.datasets.neuscenes import NeuscenesMetadata, SceneStats
from src.models import FoundationModelConfig, NeuscenesFoundationModel


"""Dataset converting Neuscenes scene aggregates into tensor features."""
class NeuscenesSceneDataset(Dataset):
    FEATURE_DIM = 3



    def __init__(self, scenes: List[SceneStats], sensor_index: Dict[str, int]) -> None:
        self.scenes = scenes
        self.sensor_index = sensor_index
        self.num_sensors = len(sensor_index)

    """Return total number of available scenes."""

    def __len__(self) -> int:
        return len(self.scenes)

    """Return feature tensor and regression target for a scene.

    Args:
        idx: Zero-based scene index.

    Returns:
        Tuple containing sensor-feature tensor and target tensor.
    """

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


"""Parse CLI arguments controlling the Neuscenes foundation run."""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Neuscenes foundation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.config:AppConfig",
        help="Python path to the application config class (e.g. 'configs.config:AppConfig').",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional Python path overriding the dataset config class.",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default=None,
        help="Optional Python path overriding the training config class.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=None,
        help="Optional path to store training history JSON.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Optional path to save a loss curve plot image.",
    )
    return parser.parse_args()


"""Load a CONFIG_CLASS implementation from a Python file."""
def load_config_class(path: Path) -> type:
    module_name, attr_name = path.stem.split(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


"""Resolve a JSON-style config reference into a dict."""
def load_json_config(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        cleaned_lines = []
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            cleaned_lines.append(line.split("//", 1)[0])
    cleaned = "\n".join(cleaned_lines)
    return json.loads(cleaned)


"""Resolve relative paths against the base application config."""
def resolve_path(base_config: Path, candidate: Optional[Path | str]) -> Optional[Path]:
    if isinstance(candidate, Path):
        return candidate.expanduser().resolve()
    if isinstance(candidate, str):
        return base_config.parent.joinpath(candidate).resolve()
    return None


"""Derive a stable sensor index from dataset metadata.

Args:
    metadata: Loaded Neuscenes metadata.

Returns:
    Mapping from sensor name to contiguous index.
"""
def build_sensor_index(metadata: NeuscenesMetadata) -> Dict[str, int]:
    names = sorted({sensor for scene in metadata.scenes for sensor in scene.sensors.keys()})
    if not names:
        raise RuntimeError("No sensors discovered in Neuscenes metadata.")
    return {name: idx for idx, name in enumerate(names)}


"""Instantiate an optimizer from configuration.

Args:
    model: Model whose parameters will be optimized.
    spec: Optimizer configuration dictionary.

Returns:
    Configured torch optimizer instance.
"""
def build_optimizer(model: nn.Module, spec: Dict[str, Any]) -> torch.optim.Optimizer:
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


"""Instantiate a loss function from configuration.

Args:
    spec: Loss configuration dictionary.

Returns:
    Configured torch loss module.
"""
def build_loss(spec: Dict[str, Any]) -> nn.Module:
    name = spec.get("name", "mse").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smoothl1":
        beta = float(spec.get("beta", 1.0))
        return nn.SmoothL1Loss(beta=beta)
    raise ValueError(f"Unsupported loss '{name}'.")


"""Create train/validation/test dataloaders from scene splits.

Args:
    splits: Partitioned scenes keyed by split name.
    sensor_index: Consistent sensor ordering.
    dataloader_cfg: Loader hyperparameters.

Returns:
    Tuple of train, validation, and test dataloaders (latter two optional).
"""
def build_dataloaders(
    splits: Dict[str, List[SceneStats]],
    sensor_index: Dict[str, int],
    dataloader_cfg: Dict[str, Any],
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:

    batch_size = int(dataloader_cfg.get("batch_size", 16))
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", torch.cuda.is_available()))

    def make_loader(scenes: Iterable[SceneStats], shuffle: bool) -> Optional[DataLoader]:
        scenes = list(scenes)
        if not scenes:
            return None
        dataset = NeuscenesSceneDataset(scenes, sensor_index)
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


"""Persist training history if an output path is provided.

Args:
    history: Epoch-wise metrics to serialize.
    path: Target file location or None.
"""
def maybe_save_history(history: List[Dict[str, Any]], path: Optional[Path]) -> None:
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


"""Entry point orchestrating config loading, dataloaders, and training."""
def main() -> None:
    args = parse_args()
    app_config_path = args.config.expanduser().resolve()
    app_config_cls = load_config_class(app_config_path)
    app_config = app_config_cls()

    dataset_settings = (
        load_config_target(args.dataset_config) if args.dataset_config else app_config.dataset
    )
    training_settings = (
        load_config_target(args.training_config) if args.training_config else app_config.training
    )

    def to_dataset_config(obj: Any):
        if hasattr(obj, "to_dataset_config"):
            return obj.to_dataset_config()
        from src.datasets.neuscenes import DatasetConfig as NeuscenesDatasetConfig

        if isinstance(obj, NeuscenesDatasetConfig):
            return obj
        if isinstance(obj, dict):
            splits_raw = {k: float(v) for k, v in obj["splits"].items()}
            total = sum(splits_raw.values())
            if total <= 0:
                raise ValueError("At least one positive dataset split ratio is required.")
            normalized = {name: value / total for name, value in splits_raw.items()}
            return NeuscenesDatasetConfig(
                dataset_root=Path(obj["dataset_root"]).expanduser().resolve(),
                splits=normalized,
                shuffle=bool(obj.get("shuffle", True)),
                seed=int(obj.get("seed", 42)),
            )
        raise TypeError(
            "Unsupported dataset configuration type; expected object with to_dataset_config()."
        )

    def to_dict(obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
        if isinstance(obj, dict):
            return dict(obj)
        from dataclasses import asdict, is_dataclass

        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError("Configuration objects must expose as_dict() or be dicts.")

    def to_training_payload(obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
        if isinstance(obj, dict):
            return dict(obj)
        from dataclasses import asdict, is_dataclass

        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError("Unsupported training configuration type; provide as_dict() or mapping.")

    dataset_config = to_dataset_config(dataset_settings)
    training_payload = to_training_payload(training_settings)

    training_config = load_training_config(training_payload)

    metadata = load_neuscenes_metadata(dataset_config.dataset_root)
    splits = split_neuscenes(metadata, dataset_config)
    sensor_index = build_sensor_index(metadata)

    dataloader_cfg = asdict(app_config.dataloader)
    train_loader, val_loader, test_loader = build_dataloaders(
        splits,
        sensor_index,
        dataloader_cfg,
    )

    model_cfg = dict(training_payload.get("model", {}))
    if not model_cfg:
        raise ValueError("Model configuration missing from training settings.")
    if "input_dim" not in model_cfg:
        raise ValueError("Model configuration must include 'input_dim'.")
    if model_cfg["input_dim"] != NeuscenesSceneDataset.FEATURE_DIM:
        raise ValueError(
            f"Configured input_dim ({model_cfg['input_dim']}) does not match dataset feature "
            f"dimension ({NeuscenesSceneDataset.FEATURE_DIM})."
        )
    model_config = FoundationModelConfig(**model_cfg)
    model = NeuscenesFoundationModel(model_config)

    optimizer = build_optimizer(model, to_dict(app_config.optimizer))
    loss_fn = build_loss(to_dict(app_config.loss))

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

    history_target = args.history_out or getattr(app_config, "history_output", None)
    history_path = Path(history_target) if history_target is not None else None
    maybe_save_history(history, history_path)

    if args.plot_out is not None:
        plot_path: Optional[Path] = args.plot_out
    elif history_path is not None:
        plot_path = history_path.with_suffix(".png")
    else:
        plot_path = PROJECT_ROOT / "experiments" / "training_history.png"
    maybe_plot_history(history, plot_path)

    os.environ["FOUNDATION_APP_CONFIG"] = describe_config(app_config)
    os.environ["FOUNDATION_DATASET_CONFIG"] = (
        args.dataset_config or describe_config(dataset_settings)
    )
    os.environ["FOUNDATION_TRAINING_CONFIG"] = (
        args.training_config or describe_config(training_settings)
    )

    payload: Dict[str, Any] = {"train_history": history}
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
