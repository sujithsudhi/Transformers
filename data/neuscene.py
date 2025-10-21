from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class SensorSummary:
    """Aggregate statistics describing a single sensor within a scene."""

    file_count: int
    total_size_bytes: float

    @staticmethod
    def from_mapping(payload: Mapping[str, object]) -> "SensorSummary":
        return SensorSummary(
            file_count=int(payload.get("file_count", 0)),
            total_size_bytes=float(payload.get("total_size_bytes", 0.0)),
        )


@dataclass(frozen=True)
class SceneRecord:
    """Snapshot of a scene comprising multiple sensor measurements."""

    scene_id: str
    sensors: Dict[str, SensorSummary]

    @staticmethod
    def from_mapping(payload: Mapping[str, object]) -> "SceneRecord":
        scene_id = str(payload.get("scene_id", ""))
        sensors_raw = payload.get("sensors", {})
        if not isinstance(sensors_raw, Mapping):
            raise TypeError("Scene sensors must be provided as a mapping.")
        sensors: Dict[str, SensorSummary] = {}
        for name, spec in sensors_raw.items():
            if not isinstance(spec, Mapping):
                raise TypeError("Sensor specification must be a mapping of statistics.")
            sensors[str(name)] = SensorSummary.from_mapping(spec)
        return SceneRecord(scene_id=scene_id or "", sensors=sensors)


@dataclass(frozen=True)
class DatasetMetadata:
    """Collection of scene records describing the dataset footprint."""

    scenes: List[SceneRecord]

    @staticmethod
    def from_dict(payload: Mapping[str, object]) -> "DatasetMetadata":
        scenes_raw = payload.get("scenes", [])
        if not isinstance(scenes_raw, Sequence):
            raise TypeError("Metadata must contain a 'scenes' sequence.")
        scenes = [SceneRecord.from_mapping(scene) for scene in scenes_raw]
        return DatasetMetadata(scenes=list(scenes))


@dataclass(frozen=True)
class SceneDatasetConfig:
    """Configuration describing how to load and split scene metadata."""

    dataset_root: Path
    metadata_file: str = "metadata.json"
    splits: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    shuffle: bool = True
    seed: int = 42

    def to_dataset_config(self) -> "SceneDatasetConfig":
        return self


def load_scene_metadata(dataset_root: Path, metadata_file: str = "metadata.json") -> DatasetMetadata:
    """Load dataset metadata from a JSON file located under the dataset root."""
    resolved_root = dataset_root.expanduser().resolve()
    metadata_path = resolved_root / metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{metadata_file}' not found under {resolved_root}. "
            "Provide a JSON file containing a 'scenes' list."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Metadata JSON must contain a mapping at the top level.")
    return DatasetMetadata.from_dict(payload)


def _normalise_splits(splits: Mapping[str, float]) -> Dict[str, float]:
    if not splits:
        raise ValueError("At least one dataset split must be provided.")
    total = float(sum(splits.values()))
    if total <= 0:
        raise ValueError("Dataset split weights must sum to a positive value.")
    return {name: float(weight) / total for name, weight in splits.items()}


def split_scene_dataset(
    metadata: DatasetMetadata,
    config: SceneDatasetConfig,
) -> Dict[str, List[SceneRecord]]:
    """Partition scene metadata into train/val/test splits."""
    scenes = list(metadata.scenes)
    if not scenes:
        raise RuntimeError("No scenes available in the metadata file.")

    if config.shuffle:
        random.Random(config.seed).shuffle(scenes)

    target_weights = _normalise_splits(config.splits)
    total_scenes = len(scenes)

    counts: Dict[str, int] = {}
    for name, weight in target_weights.items():
        counts[name] = int(math.floor(weight * total_scenes))

    assigned = sum(counts.values())
    remainder = total_scenes - assigned
    if remainder > 0:
        sorted_names = sorted(target_weights.items(), key=lambda item: item[1], reverse=True)
        idx = 0
        while remainder > 0 and sorted_names:
            split_name, _ = sorted_names[idx % len(sorted_names)]
            counts[split_name] += 1
            remainder -= 1
            idx += 1

    result: Dict[str, List[SceneRecord]] = {}
    start = 0
    for name, count in counts.items():
        end = start + count
        result[name] = scenes[start:end]
        start = end
    if start < total_scenes:
        result.setdefault("train", []).extend(scenes[start:])
    return result
