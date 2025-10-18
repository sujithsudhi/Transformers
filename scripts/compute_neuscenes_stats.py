from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    load_neuscenes_metadata,
    split_neuscenes,
    summarize_neuscenes,
)  # noqa: E402
from src.datasets.neuscenes import DatasetConfig  # noqa: E402


def load_config_target(target: str) -> Any:
    """Import and instantiate a configuration class referenced by string."""
    if not target:
        raise ValueError("Configuration target string cannot be empty.")
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    if isinstance(attr, type):
        return attr()
    return attr


def to_dataset_config(obj: Any) -> DatasetConfig:
    """Normalise dataset configuration inputs into DatasetConfig."""
    if hasattr(obj, "to_dataset_config"):
        return obj.to_dataset_config()
    if isinstance(obj, DatasetConfig):
        return obj
    if isinstance(obj, dict):
        splits_raw = {k: float(v) for k, v in obj["splits"].items()}
        total = sum(splits_raw.values())
        if total <= 0:
            raise ValueError("At least one positive dataset split ratio is required.")
        normalized = {name: value / total for name, value in splits_raw.items()}
        return DatasetConfig(
            dataset_root=Path(obj["dataset_root"]).expanduser().resolve(),
            splits=normalized,
            shuffle=bool(obj.get("shuffle", True)),
            seed=int(obj.get("seed", 42)),
        )
    raise TypeError("Unsupported dataset configuration input.")


"""Parse command-line arguments for dataset statistics reporting."""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for the Neuscenes dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs.config:DatasetSettings",
        help="Python path to the dataset config class (e.g. 'configs.config:DatasetSettings').",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optionally override the dataset root defined in the config.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the statistics as indented JSON.",
    )
    return parser.parse_args()


"""Entry point: load configs, compute metadata, and print summary JSON."""
def main() -> None:
    args = parse_args()
    dataset_config = load_neuscenes_config(args.config) if args.config else None
    dataset_root = args.dataset_root or (dataset_config.dataset_root if dataset_config else None)
    if dataset_root is None:
        raise ValueError("Provide a dataset root via --dataset-root or the config file.")
    metadata = load_neuscenes_metadata(dataset_root)
    summary = summarize_neuscenes(metadata)
    if dataset_config:
        # Surface split sizes to help validate partitioning logic.
        splits = split_neuscenes(metadata, dataset_config)
        summary["split_counts"] = {name: len(items) for name, items in splits.items()}
    dump_kwargs = {"indent": 2, "sort_keys": True} if args.pretty else {}
    print(json.dumps(summary, **dump_kwargs))


if __name__ == "__main__":
    main()
