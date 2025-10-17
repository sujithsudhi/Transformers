"""Train the Neuscenes foundation model using configuration-driven orchestration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import Trainer, evaluate, load_training_config
from src.datasets import load_neuscenes_config, load_neuscenes_metadata, split_neuscenes
from src.datasets.neuscenes import NeuscenesMetadata, SceneStats
from src.models import FoundationModelConfig, NeuscenesFoundationModel

class NeuscenesSceneDataset(Dataset):
    """Dataset converting Neuscenes scene aggregates into tensor features."""

    FEATURE_DIM = 3

    """Store scene metadata and fixed sensor ordering.

    Args:
        scenes: Ordered collection of scene statistics.
        sensor_index: Mapping from sensor name to tensor position.
    """

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


"""Parse CLI arguments for the training application."""
def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training application."""
    parser = argparse.ArgumentParser(description="Train the Neuscenes foundation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "app.json",
        help="Application configuration JSON.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=None,
        help="Optional override for dataset config.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="Optional override for training config.",
    )
    parser.add_argument(
        "--history-out",
        type=Path,
        default=None,
        help="Optional path to store training history JSON.",
    )
    return parser.parse_args()


"""Load a JSON configuration file.

Args:
    path: Location of the JSON config.

Returns:
    Parsed configuration dictionary.
"""
def load_json_config(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    cleaned = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("//"))
    return json.loads(cleaned)


"""Wrapper loading the application-level configuration."""
def load_app_config(path: Path) -> Dict[str, Any]:
    return load_json_config(path)


"""Resolve relative paths against a base configuration file.

Args:
    base_config: Absolute path of the base config.
    candidate: Path string or Path object to resolve.

Returns:
    Absolute path or None if no candidate supplied.
"""
def resolve_path(base_config: Path, candidate: Optional[Path | str]) -> Optional[Path]:
    if candidate is None:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = (base_config.parent / path).resolve()
    return path


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


# Entry point coordinating configuration loading and training.
def main() -> None:
    """Entry point coordinating configuration loading and training."""
    args = parse_args()
    app_config_path = args.config.expanduser().resolve()
    app_config = load_app_config(app_config_path)

    dataset_config_path = resolve_path(
        app_config_path, args.dataset_config or app_config.get("dataset_config")
    )
    training_config_path = resolve_path(
        app_config_path, args.training_config or app_config.get("training_config")
    )
    if dataset_config_path is None or training_config_path is None:
        raise ValueError("Both dataset_config and training_config must be provided.")

    os.environ["FOUNDATION_APP_CONFIG"] = str(app_config_path)
    os.environ["FOUNDATION_DATASET_CONFIG"] = str(dataset_config_path)
    os.environ["FOUNDATION_TRAINING_CONFIG"] = str(training_config_path)

    dataset_config = load_neuscenes_config(dataset_config_path)
    training_config = load_training_config(training_config_path)
    metadata = load_neuscenes_metadata(dataset_config.dataset_root)
    splits = split_neuscenes(metadata, dataset_config)
    sensor_index = build_sensor_index(metadata)
    dataloader_cfg = app_config.get("dataloader", {})
    train_loader, val_loader, test_loader = build_dataloaders(
        splits,
        sensor_index,
        dataloader_cfg,
    )
    training_payload = load_json_config(training_config_path)
    model_cfg = dict(training_payload.get("model", {}))
    if not model_cfg:
        raise ValueError("Model configuration missing from training config.")
    if "input_dim" not in model_cfg:
        raise ValueError("Model configuration must include 'input_dim'.")
    if model_cfg["input_dim"] != NeuscenesSceneDataset.FEATURE_DIM:
        raise ValueError(
            f"Configured input_dim ({model_cfg['input_dim']}) does not match dataset feature "
            f"dimension ({NeuscenesSceneDataset.FEATURE_DIM})."
        )
    model_config = FoundationModelConfig(**model_cfg)
    model = NeuscenesFoundationModel(model_config)

    optimizer = build_optimizer(model, app_config.get("optimizer", {}))
    loss_fn = build_loss(app_config.get("loss", {}))

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
        )

    history_path = args.history_out or resolve_path(
        app_config_path, app_config.get("history_output")
    )
    maybe_save_history(history, history_path)

    payload: Dict[str, Any] = {"train_history": history}
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
