from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Ensure repository root is importable when executing as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_imdb_dataloaders  # noqa: E402
from models import ClassifierModel, TransformersModelConfig  # noqa: E402
from tool.utils import _to_serializable, load_config_target  # noqa: E402
from training import (  # noqa: E402
    collect_classification_outputs,
    compute_class_distribution,
    evaluate,
    load_training_config,
    prepare_classification_labels,
)
from viz.plots import plot_class_distribution, plot_confusion_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained sentiment model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs.imdb:IMDBConfig",
        help="Python path to the configuration object (e.g. 'configs.imdb:IMDBConfig').",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/model.pt",
        help="Path to the model checkpoint produced during training.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/validation",
        help="Directory where validation artifacts will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size defined in the configuration.",
    )
    parser.add_argument(
        "--label-names",
        type=str,
        default=None,
        help="Optional comma-separated class labels used for plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for evaluation (defaults to training config).",
    )
    return parser.parse_args()


def _build_dataloader(
    split: str,
    batch_size: int,
    max_tokens: int,
    num_workers: int,
    cache_dir: Optional[Path],
    dataset_name: str,
    dataset_root: Optional[Path],
    download: bool,
):
    train_loader, test_loader = build_imdb_dataloaders(
        batch_size=batch_size,
        max_tokens=max_tokens,
        num_workers=num_workers,
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        download=download,
    )
    return train_loader if split == "train" else test_loader


def _save_metrics(
    metrics: Dict[str, float],
    class_counts: Dict[str, int],
    output_dir: Path,
    split: str,
    label_names: Optional[List[str]] = None,
) -> Path:
    payload = {
        "metrics": metrics,
        "class_distribution": class_counts,
    }
    if label_names is not None:
        payload["label_names"] = label_names

    metrics_path = output_dir / f"{split}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return metrics_path


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    app_config = load_config_target(args.config)
    data_cfg = app_config.data
    model_cfg = asdict(app_config.model)

    batch_size = args.batch_size or data_cfg.batch_size
    dataloader = _build_dataloader(
        split=args.split,
        batch_size=batch_size,
        max_tokens=data_cfg.max_tokens,
        num_workers=data_cfg.num_workers,
        cache_dir=data_cfg.cache_dir,
        dataset_name=getattr(data_cfg, "dataset_name", "imdb"),
        dataset_root=getattr(data_cfg, "dataset_root", None),
        download=getattr(data_cfg, "download", True),
    )

    feature_dim = getattr(dataloader.dataset, "feature_dim", None)
    if feature_dim is None:
        raise AttributeError("Dataset does not expose required 'feature_dim'.")
    model_cfg["input_dim"] = feature_dim
    model_config = TransformersModelConfig(**model_cfg)
    model = ClassifierModel(model_config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(state_dict)

    training_config_dict = _to_serializable(app_config.training)
    training_config = load_training_config(training_config_dict)
    if args.device:
        training_config.device = args.device

    device = torch.device(training_config.device)
    loss_fn: torch.nn.Module
    if model_config.num_outputs == 1:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
        non_blocking=training_config.non_blocking,
        progress_desc=f"{args.split.title()} Eval",
    )

    _, probabilities, targets = collect_classification_outputs(
        model,
        dataloader,
        device,
        non_blocking=training_config.non_blocking,
    )
    preds, true_labels = prepare_classification_labels(probabilities, targets)

    total = max(1, true_labels.numel())
    accuracy = float((preds == true_labels).sum().item()) / total
    metrics["accuracy"] = accuracy
    metrics["examples"] = int(total)

    label_names: Optional[List[str]] = None
    if args.label_names:
        label_names = [name.strip() for name in args.label_names.split(",")]
    elif hasattr(app_config, "class_labels"):
        label_names = list(getattr(app_config, "class_labels"))

    class_counts = compute_class_distribution(true_labels)
    metrics_path = _save_metrics(metrics, class_counts, output_dir, args.split, label_names)
    print(f"Validation metrics written to {metrics_path}")

    confusion_fig, _ = plot_confusion_matrix(
        y_true=true_labels.numpy(),
        y_pred=preds.numpy(),
        labels=label_names,
        normalize=True,
        title=f"{args.split.title()} Confusion Matrix",
    )
    confusion_path = output_dir / f"{args.split}_confusion_matrix.png"
    confusion_fig.savefig(confusion_path, dpi=200)
    print(f"Confusion matrix saved to {confusion_path}")

    class_fig, _ = plot_class_distribution(
        labels=true_labels.numpy(),
        label_names=label_names,
        title=f"{args.split.title()} Class Distribution",
    )
    class_path = output_dir / f"{args.split}_class_distribution.png"
    class_fig.savefig(class_path, dpi=200)
    print(f"Class distribution saved to {class_path}")


if __name__ == "__main__":
    main()
