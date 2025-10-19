"""Run inference on a TensorRT engine produced from the TransformersModel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "TensorRT python bindings and pycuda are required. Install them before running this script."
    ) from exc


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorRT engine inference helper.")
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
        help="Path to the TensorRT engine (.plan).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional .npy file providing input tensor data.",
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
    return parser.parse_args()


def _load_engine(path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(TRT_LOGGER)
    with path.open("rb") as handle:
        engine_data = handle.read()
    return runtime.deserialize_cuda_engine(engine_data)


def _allocate_buffers(engine: trt.ICudaEngine) -> Tuple[List[cuda.DeviceAllocation], List[cuda.DeviceAllocation]]:
    inputs = []
    outputs = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = max(size, 1)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
        if engine.binding_is_input(binding):
            inputs.append(device_mem)
        else:
            outputs.append(device_mem)
    return inputs, outputs


def main() -> None:
    args = parse_args()
    engine = _load_engine(args.engine)
    context = engine.create_execution_context()

    if not context:
        raise RuntimeError("Failed to create TensorRT execution context.")

    # Determine shapes
    input_shape = (args.batch_size, args.sequence_length, args.input_dim)
    context.set_binding_shape(0, input_shape)
    output_shape = tuple(context.get_binding_shape(1))

    host_input = (
        np.load(args.input).astype(np.float32)
        if args.input is not None
        else np.random.standard_normal(size=input_shape).astype(np.float32)
    )
    host_output = np.empty(output_shape, dtype=np.float32)

    d_inputs, d_outputs = _allocate_buffers(engine)

    cuda.memcpy_htod(d_inputs[0], host_input.tobytes())
    context.execute_v2([int(mem) for mem in d_inputs + d_outputs])
    cuda.memcpy_dtoh(host_output, d_outputs[0])

    print("Logits:", host_output)


if __name__ == "__main__":
    main()
