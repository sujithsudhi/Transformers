from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Tuple
import json
from random import Random


@dataclass(frozen=True)
class SensorStats:
    file_count: int
    total_size_bytes: int
    extension_histogram: Dict[str, int]


@dataclass(frozen=True)
class SceneStats:
    scene_id: str
    sensors: Dict[str, SensorStats]


@dataclass(frozen=True)
class NeuscenesMetadata:
    root: Path
    scenes: List[SceneStats]

    @property
    def num_scenes(self) -> int:
        return len(self.scenes)


@dataclass(frozen=True)
class DatasetConfig:
    dataset_root: Path
    splits: Dict[str, float]
    shuffle: bool = True
    seed: int = 42


def _iter_files(directory: Path) -> Iterable[Path]:
    for path in directory.iterdir():
        if path.is_file():
            yield path


def _gather_sensor_stats(sensor_dir: Path) -> SensorStats:
    extension_histogram: Counter[str] = Counter()
    total_size_bytes = 0
    files = list(_iter_files(sensor_dir))
    for file_path in files:
        extension_histogram[file_path.suffix.lower()] += 1
        total_size_bytes += file_path.stat().st_size
    return SensorStats(
        file_count=len(files),
        total_size_bytes=total_size_bytes,
        extension_histogram=dict(extension_histogram),
    )


def load_neuscenes_metadata(dataset_root: Path | str) -> NeuscenesMetadata:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    scenes: List[SceneStats] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        sensors: Dict[str, SensorStats] = {}
        for sensor_dir in sorted(entry.iterdir()):
            if not sensor_dir.is_dir():
                continue
            sensors[sensor_dir.name] = _gather_sensor_stats(sensor_dir)
        scenes.append(SceneStats(scene_id=entry.name, sensors=sensors))
    if not scenes:
        raise RuntimeError(f"No scenes discovered under {root}")
    return NeuscenesMetadata(root=root, scenes=scenes)


def _aggregate_sensor_counts(scenes: Iterable[SceneStats]) -> Tuple[Counter[str], Counter[str]]:
    sensor_file_counts: Counter[str] = Counter()
    extension_histogram: Counter[str] = Counter()
    for scene in scenes:
        for sensor_name, stats in scene.sensors.items():
            sensor_file_counts[sensor_name] += stats.file_count
            extension_histogram.update(stats.extension_histogram)
    return sensor_file_counts, extension_histogram


def summarize_neuscenes(metadata: NeuscenesMetadata) -> Dict[str, object]:
    scene_file_totals = [
        sum(sensor.file_count for sensor in scene.sensors.values())
        for scene in metadata.scenes
    ]
    total_files = sum(scene_file_totals)
    total_size_bytes = sum(
        sensor.total_size_bytes
        for scene in metadata.scenes
        for sensor in scene.sensors.values()
    )
    sensor_counts, extension_counts = _aggregate_sensor_counts(metadata.scenes)
    files_per_scene = {
        "min": min(scene_file_totals),
        "max": max(scene_file_totals),
        "mean": mean(scene_file_totals),
        "median": median(scene_file_totals),
    }
    return {
        "dataset_root": str(metadata.root),
        "num_scenes": metadata.num_scenes,
        "total_files": total_files,
        "total_size_gb": round(total_size_bytes / (1024**3), 4),
        "files_per_scene": files_per_scene,
        "top_sensors": sensor_counts.most_common(10),
        "top_extensions": extension_counts.most_common(10),
    }


def _normalize_split_ratios(splits: Mapping[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    total = 0.0
    for name, ratio in splits.items():
        value = float(ratio)
        if value < 0:
            raise ValueError(f"Split ratio for '{name}' must be non-negative.")
        normalized[name] = value
        total += value
    if not normalized or total <= 0:
        raise ValueError("At least one positive dataset split ratio is required.")
    return {name: value / total for name, value in normalized.items()}


def load_neuscenes_config(config_path: Path | str) -> DatasetConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Neuscenes config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    dataset_root = Path(payload["dataset_root"]).expanduser().resolve()
    splits = _normalize_split_ratios(payload.get("splits", {}))
    return DatasetConfig(
        dataset_root=dataset_root,
        splits=splits,
        shuffle=bool(payload.get("shuffle", True)),
        seed=int(payload.get("seed", 42)),
    )


def split_neuscenes(metadata: NeuscenesMetadata, config: DatasetConfig) -> Dict[str, List[SceneStats]]:
    scenes = list(metadata.scenes)
    if config.shuffle:
        Random(config.seed).shuffle(scenes)
    total = len(scenes)
    if total == 0:
        raise RuntimeError("Cannot split Neuscenes metadata with zero scenes.")
    splits: Dict[str, List[SceneStats]] = {}
    remaining = total
    cursor = 0
    for index, (name, ratio) in enumerate(config.splits.items()):
        if index == len(config.splits) - 1:
            count = remaining
        else:
            count = min(remaining, int(round(ratio * total)))
        splits[name] = scenes[cursor : cursor + count]
        cursor += count
        remaining -= count
    return splits
