"""Plotting helpers for model metrics and predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - optional dependency during docs builds
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover - optional at runtime
    from sklearn.metrics import confusion_matrix
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn is required for confusion-matrix plotting. "
        "Install it via 'pip install scikit-learn'."
    ) from exc


def _to_numpy(array: Sequence[float] | np.ndarray | "torch.Tensor") -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if torch is not None and isinstance(array, torch.Tensor):  # pragma: no branch
        return array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def plot_loss_curves(
    history: Sequence[Mapping[str, Mapping[str, float]]],
    metrics: Tuple[str, ...] = ("train", "val"),
    metric_key: str = "loss",
    title: str = "Training History",
):
    """Plot training/validation loss curves from the trainer history list."""
    if not history:
        raise ValueError("History is empty; nothing to plot.")

    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(history)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in metrics:
        series = [entry.get(name, {}).get(metric_key) for entry in history]
        if any(value is not None for value in series):
            ax.plot(epochs, series, marker="o", label=f"{name.title()} {metric_key}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key.title())
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(y_true: Sequence[int] | np.ndarray | "torch.Tensor",
                          y_pred: Sequence[int] | np.ndarray | "torch.Tensor",
                          labels: Optional[Sequence[str]] = None,
                          normalize: bool = True,
                          title: str = "Confusion Matrix",
                         ):
    """Plot a confusion matrix given integer labels/predictions."""
    y_true_np = _to_numpy(y_true).astype(int)
    y_pred_np = _to_numpy(y_pred).astype(int)

    matrix = confusion_matrix(y_true_np, y_pred_np)
    if normalize:
        with np.errstate(all="ignore"):
            matrix = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)
            matrix = np.nan_to_num(matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    num_classes = matrix.shape[0]
    tick_labels = labels if labels is not None else [str(idx) for idx in range(num_classes)]
    ax.set_xticks(np.arange(num_classes), labels=tick_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(num_classes), labels=tick_labels)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig, ax


def plot_prediction_histogram(scores: Sequence[float] | np.ndarray | "torch.Tensor",
                              bins: int = 20,
                              title: str = "Prediction Score Distribution",
                             ):
    """Plot a histogram of model prediction scores (probabilities or logits)."""
    values = _to_numpy(scores).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.75, edgecolor="black")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.7)
    fig.tight_layout()
    return fig, ax


def plot_class_distribution(
    labels: Sequence[int] | np.ndarray | "torch.Tensor",
    label_names: Optional[Sequence[str]] = None,
    title: str = "Class Distribution",
):
    """Plot a bar chart showing the distribution of class labels within a dataset."""
    label_array = _to_numpy(labels).astype(int).ravel()
    unique, counts = np.unique(label_array, return_counts=True)

    indices = np.argsort(unique)
    unique = unique[indices]
    counts = counts[indices]

    if label_names is not None:
        name_lookup = {
            idx: label_names[idx] if 0 <= idx < len(label_names) else str(idx)
            for idx in unique
        }
        display_names = [name_lookup[idx] for idx in unique]
    else:
        display_names = [str(idx) for idx in unique]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(display_names, counts, color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.6, axis="y")
    fig.tight_layout()
    return fig, ax
