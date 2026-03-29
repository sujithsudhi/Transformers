"""Export a fine-tuned TransformersModel checkpoint to ONNX (and optionally TFLite)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import TransformersModelConfig  # noqa: E402
from models import ClassifierModel  # noqa: E402


def _load_config(target: str) -> Any:
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    return attr() if isinstance(attr, type) else attr


def _resolve_model_config(app_config: Any, input_dim: int | None) -> TransformersModelConfig:
    raw_config = app_config.model
    payload: Dict[str, Any] = {}
    if is_dataclass(raw_config):
        payload.update(asdict(raw_config))
    elif hasattr(raw_config, "__dict__"):
        payload.update(vars(raw_config))
    elif hasattr(raw_config, "_asdict"):
        payload.update(raw_config._asdict())  # type: ignore[attr-defined]
    elif hasattr(raw_config, "as_dict"):
        payload.update(raw_config.as_dict())  # type: ignore[attr-defined]
    else:
        payload.update(dict(raw_config))
    resolved_vocab_size = payload.get("vocab_size")
    if resolved_vocab_size is not None and int(resolved_vocab_size) <= 0:
        resolved_vocab_size = None
    if resolved_vocab_size is not None:
        payload["vocab_size"] = int(resolved_vocab_size)
    resolved_input_dim = input_dim or payload.get("input_dim")
    if resolved_vocab_size is None and resolved_input_dim is None:
        raise ValueError("Model configuration must define --input-dim or store input_dim.")
    if resolved_input_dim is not None:
        payload["input_dim"] = int(resolved_input_dim)
    payload.setdefault("num_outputs", getattr(raw_config, "num_outputs", 1))
    return TransformersModelConfig(**payload)


def _load_checkpoint(model: ClassifierModel, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
        )
        if state_dict is None:
            model.load_state_dict(checkpoint, strict=False)
            return
        model.load_state_dict(state_dict, strict=False)
    else:
        raise TypeError("Unsupported checkpoint format; expected dict with state_dict.")


def _export_to_onnx(
    model: ClassifierModel,
    output_path: Path,
    dummy_input: torch.Tensor,
    dynamic_axes: bool,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model,
                      dummy_input,
                      output_path,
                      opset_version = opset,
                      input_names   = ["inputs"],
                      output_names  = ["logits"],
                      dynamic_axes  = {"inputs": {0: "batch", 1: "tokens"}, "logits": {0: "batch"}}
                                      if dynamic_axes
                                      else None,
                     )
    print(f"ONNX model exported to {output_path}")


def _export_to_tflite(onnx_path: Path, tflite_path: Path) -> None:
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "TFLite export requires 'onnx', 'onnx-tf', and 'tensorflow' packages."
        ) from exc

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        tf_rep.export_graph(tmp_path.as_posix())
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_path.as_posix())
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with tflite_path.open("wb") as handle:
        handle.write(tflite_model)
    print(f"TFLite model exported to {tflite_path}")


def _build_dummy_input(
    batch_size: int,
    sequence_length: int,
    input_dim: int | None,
    vocab_size: int | None,
    device: torch.device,
) -> torch.Tensor:
    if vocab_size is not None and vocab_size > 0:
        return torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
    if input_dim is None or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer when vocab_size is not set.")
    return torch.randn(batch_size, sequence_length, input_dim, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a ClassifierModel to ONNX/TFLite.")
    parser.add_argument("--config",
                        type     = str,
                        required = True,
                        help     = "Python path to the AppConfig implementation (e.g. 'configs.imdb:IMDBConfig').",
                       )
    parser.add_argument("--checkpoint",
                        type     = Path,
                        required = True,
                        help     = "Path to the fine-tuned model checkpoint (.pt).",
                       )
    parser.add_argument("--onnx-out",
                        type     = Path,
                        required = True,
                        help     = "Destination path for the exported ONNX model.",
                       )
    parser.add_argument("--tflite-out",
                        type    = Path,
                        default = None,
                        help    = "Optional destination path for TFLite conversion.",
                       )
    parser.add_argument("--input-dim",
                        type    = int,
                        default = None,
                        help    = "Dimensionality of each input token feature vector.",
                       )
    parser.add_argument("--sequence-length",
                        type    = int,
                        default = 256,
                        help    = "Dummy sequence length used during export.",
                       )
    parser.add_argument("--batch-size",
                        type    = int,
                        default = 1,
                        help    = "Dummy batch size used during export.",
                       )
    parser.add_argument("--device",
                        type    = str,
                        default = "cpu",
                        help    = "Device used for export (cpu or cuda).",
                       )
    parser.add_argument("--opset",
                        type    = int,
                        default = 17,
                        help    = "ONNX opset version to target.",
                       )
    parser.add_argument("--no-dynamic-axes",
                        action = "store_true",
                        help   = "Disable dynamic axes in the exported ONNX graph.",
                       )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    app_config   = _load_config(args.config)
    model_config = _resolve_model_config(app_config, args.input_dim)
    model        = ClassifierModel(model_config).to(device)
    _load_checkpoint(model, args.checkpoint)

    vocab_size = getattr(model_config, "vocab_size", None)
    if vocab_size is not None and int(vocab_size) <= 0:
        vocab_size = None

    dummy_input = _build_dummy_input(batch_size       = args.batch_size,
                                     sequence_length = args.sequence_length,
                                     input_dim       = getattr(model_config, "input_dim", None),
                                     vocab_size      = vocab_size,
                                     device          = device,
                                    )

    _export_to_onnx(model,
                    args.onnx_out,
                    dummy_input,
                    dynamic_axes = not args.no_dynamic_axes,
                    opset        = args.opset,
                   )

    if args.tflite_out is not None:
        _export_to_tflite(args.onnx_out, args.tflite_out)


if __name__ == "__main__":
    main()
