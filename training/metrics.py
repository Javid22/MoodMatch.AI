"""
Small metric-tracking helpers used during training, plus the fuller
evaluation metrics (precision/recall/F1/confusion matrix) used by
`training/evaluate.py`.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


class AverageMeter:
    """Tracks a running average of a value (e.g. loss or accuracy) across
    the batches of one epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def batch_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct predictions in a single batch (0.0 - 1.0)."""
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.size(0)


def compute_classification_metrics(
    y_true: List[int], y_pred: List[int], class_names: List[str]
) -> dict:
    """Compute accuracy, macro precision/recall/F1, per-class metrics, and
    the confusion matrix for a set of predictions.

    Returns a dict suitable for printing and/or saving as a report.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float((y_true == y_pred).mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> None:
    """Render the confusion matrix as a heatmap and save it to disk.

    This makes it easy to see which emotions the model confuses with each
    other (e.g. "sad" predicted as "neutral").
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Annotate each cell with its count.
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
