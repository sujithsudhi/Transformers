from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    load_neuscenes_config,
    load_neuscenes_metadata,
    split_neuscenes,
    summarize_neuscenes,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for the Neuscenes dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/Users/sujithks/projects/Foundation-Model/configs/dataset.json"),
        help="Path to the Neuscenes dataset configuration JSON file.",
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


def main() -> None:
    args = parse_args()
    dataset_config = load_neuscenes_config(args.config) if args.config else None
    dataset_root = args.dataset_root or (dataset_config.dataset_root if dataset_config else None)
    if dataset_root is None:
        raise ValueError("Provide a dataset root via --dataset-root or the config file.")
    metadata = load_neuscenes_metadata(dataset_root)
    summary = summarize_neuscenes(metadata)
    if dataset_config:
        splits = split_neuscenes(metadata, dataset_config)
        summary["split_counts"] = {name: len(items) for name, items in splits.items()}
    dump_kwargs = {"indent": 2, "sort_keys": True} if args.pretty else {}
    print(json.dumps(summary, **dump_kwargs))


if __name__ == "__main__":
    main()
