"""Render a checkpoint architecture diagram using torchview/graphviz."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ClassifierModel  # noqa: E402


DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "embed_dim": 128,
    "depth": 4,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "use_cls_token": True,
    "cls_head_dim": 128,
    "num_outputs": 1,
    "pooling": "cls",
    "max_length": 256,
}


def _load_config(target: str) -> Any:
    if ":" in target:
        module_name, attr_name = target.split(":", 1)
    else:
        module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    return attr() if isinstance(attr, type) else attr


def _to_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()  # type: ignore[return-value]
    if hasattr(obj, "_asdict"):
        return obj._asdict()  # type: ignore[attr-defined]
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError("Unsupported config object type.")


def _extract_model_config(checkpoint: Dict[str, Any], app_config: Any) -> Dict[str, Any]:
    if isinstance(checkpoint.get("config"), dict):
        payload = checkpoint["config"].get("model")
        if payload is not None:
            return _to_mapping(payload)
    if "model_config" in checkpoint:
        return _to_mapping(checkpoint["model_config"])
    if app_config is not None and hasattr(app_config, "model"):
        return _to_mapping(app_config.model)
    return {}


def _resolve_model_config(
    raw_payload: Dict[str, Any],
    sequence_length: int,
    input_dim: Optional[int],
    vocab_size: Optional[int],
) -> SimpleNamespace:
    payload = dict(DEFAULT_MODEL_CONFIG)
    payload.update(raw_payload)

    if vocab_size is not None:
        payload["vocab_size"] = int(vocab_size)
    else:
        if payload.get("vocab_size") in (0, "0"):
            payload["vocab_size"] = None

    max_length = payload.get("max_length")
    if not max_length or int(max_length) <= 0:
        max_length = sequence_length
    payload["max_length"] = max(int(max_length), int(sequence_length))

    if payload.get("vocab_size") is None:
        resolved_input = input_dim or payload.get("input_dim")
        if not resolved_input or int(resolved_input) <= 0:
            raise ValueError("Provide --input-dim when vocab_size is not set.")
        payload["input_dim"] = int(resolved_input)

    payload.setdefault("num_outputs", 1)
    return SimpleNamespace(**payload)


def _load_state_dict(model: ClassifierModel, checkpoint: Dict[str, Any]) -> None:
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint.get("model")
    )
    if state_dict is None:
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise KeyError("Checkpoint missing model_state_dict/state_dict/model keys.")
    model.load_state_dict(state_dict, strict=False)


def _build_dummy_input(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    vocab_size: Optional[int],
    input_dim: Optional[int],
) -> torch.Tensor:
    if vocab_size is not None:
        return torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
    if input_dim is None:
        raise ValueError("input_dim must be set when vocab_size is None.")
    return torch.randn(batch_size, sequence_length, input_dim, device=device)


def _render_graph(
    model: ClassifierModel,
    dummy_input: torch.Tensor,
    output_path: Path,
    fmt: str,
    engine: str,
    device: str,
) -> None:
    try:
        from torchview import draw_graph
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torchview is required. Install with 'pip install torchview graphviz'."
        ) from exc
    try:
        from graphviz.backend.execute import ExecutableNotFound
    except ImportError:
        ExecutableNotFound = RuntimeError  # type: ignore[assignment]

    graph = draw_graph(
        model,
        input_data=dummy_input,
        expand_nested=True,
        graph_name="ClassifierModel",
        device=device,
    )

    base_path = output_path.with_suffix("")
    try:
        rendered = graph.visual_graph.render(
            filename=base_path.as_posix(),
            format=fmt,
            cleanup=True,
            engine=engine,
        )
    except ExecutableNotFound:
        dot_path = output_path.with_suffix(".dot")
        dot_path.write_text(graph.visual_graph.source, encoding="utf-8")
        raise RuntimeError(
            "Graphviz executables (e.g., 'dot') were not found on PATH. "
            f"Wrote DOT source to {dot_path} instead."
        )
    else:
        final_path = Path(rendered)
        if final_path.resolve() != output_path.resolve():
            final_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a checkpoint architecture diagram using torchview.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the saved model checkpoint (.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional AppConfig path (e.g. configs.imdb:IMDBConfig).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/checkpoint_graph.png"),
        help="Where to write the diagram (format inferred from suffix).",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Sequence length used to build dummy inputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used to build dummy inputs.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Token feature dimension when vocab_size is not set.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size (when using token IDs as input).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for tracing (cpu or cuda).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="dot",
        help="Graphviz layout engine (dot, neato, fdp, sfdp).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dict with a state_dict.")

    app_config = _load_config(args.config) if args.config else None
    raw_model_cfg = _extract_model_config(checkpoint, app_config)
    model_cfg = _resolve_model_config(
        raw_model_cfg,
        sequence_length=args.sequence_length,
        input_dim=args.input_dim,
        vocab_size=args.vocab_size,
    )

    device = torch.device(args.device)
    model = ClassifierModel(model_cfg).to(device)
    _load_state_dict(model, checkpoint)
    model.eval()

    dummy_input = _build_dummy_input(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
        vocab_size=getattr(model_cfg, "vocab_size", None),
        input_dim=getattr(model_cfg, "input_dim", None),
    )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_path.suffix.lstrip(".") or "png"

    _render_graph(
        model=model,
        dummy_input=dummy_input,
        output_path=output_path,
        fmt=fmt,
        engine=args.engine,
        device=args.device,
    )
    print(f"Diagram written to {output_path}")


if __name__ == "__main__":
    main()
