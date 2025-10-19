"""Run inference on an exported ONNX model using ONNX Runtime with TensorRT EP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "ONNX Runtime is required. Install it via 'pip install onnxruntime-gpu'."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ONNX Runtime inference helper.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to a .npy file containing input tensor data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for synthetic input if --input is not provided.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Sequence length for synthetic input if --input is not provided.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=4,
        help="Feature dimension for synthetic input if --input is not provided.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        help="Execution providers ordered by preference.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print session information and timings.",
    )
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> np.ndarray:
    if args.input is not None:
        return np.load(args.input).astype(np.float32)
    rng = np.random.default_rng()
    return rng.standard_normal(
        size=(args.batch_size, args.sequence_length, args.input_dim), dtype=np.float32
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    providers: List[str] = args.providers
    session = ort.InferenceSession(
        args.model.as_posix(),
        providers=providers,
    )
    if args.verbose:
        print("Active providers:", session.get_providers())

    inputs = _load_inputs(args)
    ort_inputs = {session.get_inputs()[0].name: inputs}
    outputs = session.run(None, ort_inputs)

    print("Logits:", outputs[0])


if __name__ == "__main__":
    main()
