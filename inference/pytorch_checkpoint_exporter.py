from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Ensure repository root is importable when executing as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool.utils import load_config_target  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a PyTorch checkpoint to a NumPy-friendly format."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs.imdb:IMDBConfig",
        help="Config path used to resolve default checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (defaults to config checkpoint_path).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference/exports/imdb_checkpoint",
        help="Output prefix (directory and file stem).",
    )
    parser.add_argument(
        "--format",
        choices=("npz", "pt"),
        default="npz",
        help="Export format for model weights.",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict with model_state_dict.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    return checkpoint


def _export_npz(state_dict: Dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    np.savez(output_path, **arrays)


def _export_pt(state_dict: Dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)


def _write_metadata(meta: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def main() -> None:
    args = parse_args()

    app_config = load_config_target(args.config)
    default_ckpt = getattr(app_config, "checkpoint_path", None)
    checkpoint_path = Path(args.checkpoint or default_ckpt).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("model_state_dict must be a dict of tensors.")

    output_prefix = Path(args.output).expanduser().resolve()
    if args.format == "npz":
        weights_path = output_prefix.with_suffix(".npz")
        _export_npz(state_dict, weights_path)
    else:
        weights_path = output_prefix.with_suffix(".pt")
        _export_pt(state_dict, weights_path)

    meta = {
        "checkpoint": checkpoint_path.as_posix(),
        "format": args.format,
        "weights": weights_path.as_posix(),
        "config": checkpoint.get("config", {}),
        "state_dict_keys": sorted(state_dict.keys()),
    }
    meta_path = output_prefix.with_suffix(".json")
    _write_metadata(meta, meta_path)

    print(f"Exported weights to {weights_path}")
    print(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    main()
