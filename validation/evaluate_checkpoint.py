"""Utility script for validating a saved IMDB transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

# Ensure project root on sys.path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.imdb import DataPrep
from models import ClassifierModel, TransformersModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on IMDB validation/test split for a saved checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file produced by training.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "val", "validation"),
        help="Dataset split to evaluate. 'val' and 'validation' map to the test split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size used for evaluation (defaults to checkpoint data config).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run inference on ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("validation/metrics.json"),
        help="Where to write the computed metrics JSON file.",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=Path("validation/confusion_matrix.png"),
        help="Where to save the rendered confusion matrix plot.",
    )
    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_checkpoint(path: Path, map_location: torch.device) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return torch.load(resolved, map_location=map_location)


def build_model(model_cfg: Dict[str, object], dataset: Any) -> ClassifierModel:
    config_payload = dict(model_cfg)
    vocab_size = getattr(dataset, "vocab_size", None)
    feature_dim = getattr(dataset, "feature_dim", None)
    if "vocab_size" not in config_payload and vocab_size is not None:
        config_payload["vocab_size"] = int(vocab_size)
    if "input_dim" not in config_payload:
        if vocab_size is not None:
            config_payload["input_dim"] = int(vocab_size)
        elif feature_dim is not None:
            config_payload["input_dim"] = int(feature_dim)
        else:
            raise ValueError(
                "Checkpoint does not define input_dim and dataset lacks feature_dim."
            )
    model_config = TransformersModelConfig(**config_payload)
    return ClassifierModel(model_config)


def evaluate_model(model      : ClassifierModel,
                   dataloader : DataLoader,
                   device     : torch.device,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:

            if isinstance(batch, dict):

                inputs = batch.get("inputs")
                labels = batch.get("targets")
                
                if inputs is None or labels is None:
                    raise ValueError("Dataloader must yield inputs and labels/targets keys.")
            else:
                inputs, labels = batch
            
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            elif isinstance(inputs, (tuple, list)):
                inputs = tuple(v.to(device) for v in inputs)
            else:
                inputs = inputs.to(device)
            
            labels = labels.to(device)
            
            logits = model(**inputs).squeeze(-1)

            probs = torch.sigmoid(logits)

            predictions.extend((probs >= 0.5).long().cpu().numpy())

            targets.extend(labels.long().cpu().numpy())

    return np.array(targets).flatten(), np.array(predictions).flatten()


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                   ) -> Tuple[Dict[str, float], np.ndarray, str]:
    
    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true,
                                                               y_pred,
                                                               average="binary",
                                                               zero_division=0.0,
                                                              )
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='pred')

    class_report = classification_report(y_true,
                                         y_pred,
                                         digits=4,
                                         zero_division=0.0,
                                        )
    metrics = {"accuracy"  : accuracy,
               "precision" : float(precision),
               "recall"    : float(recall),
               "f1_score"  : float(f1),
              }
    
    return metrics, conf_matrix, class_report


def save_metrics(
    metrics: Dict[str, object],
    report: str,
    matrix: np.ndarray,
    path: Path,
) -> None:
    payload = dict(metrics)
    payload["classification_report"] = report
    payload["confusion_matrix"] = matrix.tolist()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_confusion_plot(matrix: np.ndarray, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to export the confusion matrix plot.",
        ) from exc

    labels = ["neg", "pos"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "f"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)

    config = checkpoint.get("config")
    if config is None:
        raise KeyError("Checkpoint missing embedded configuration under 'config'.")

    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}

    valData = DataPrep(data_path  = data_cfg.get("data_path"),
                       batch_size = data_cfg.get("batch_size", 32),
                       max_tokens = data_cfg.get("max_tokens", 256),      
                       )
    
    dataloader  = valData.prep(split="test")
    
    dataset = dataloader.dataset
    model = build_model(model_cfg, dataset)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    y_true, y_pred = evaluate_model(model, dataloader, device)
    metrics, matrix, report = compute_metrics(y_true, y_pred)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nClassification report:\n")
    print(report)

    save_metrics(metrics, report, matrix, args.metrics_path)
    save_confusion_plot(matrix, args.confusion_matrix_path)
    print(f"\nMetrics written to {args.metrics_path}")
    print(f"Confusion matrix plot saved to {args.confusion_matrix_path}")


if __name__ == "__main__":
    main()
