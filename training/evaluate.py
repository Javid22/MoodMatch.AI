"""
Loads the best saved checkpoint and evaluates it on the held-out test set.

Reports:
  - test accuracy
  - macro precision / recall / F1
  - per-class precision / recall / F1 / support
  - confusion matrix (also saved as a PNG so you can see which emotions
    the model confuses with each other)

Run with:

    python -m training.evaluate
"""

import json
import os

import torch

from src.model.mobilenet import build_model
from training import config
from training.dataset import get_dataloaders
from training.metrics import compute_classification_metrics, plot_confusion_matrix


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a dataloader and collect (y_true, y_pred)."""
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    return y_true, y_pred


def print_report(metrics: dict, class_names) -> None:
    print(f"\nTest Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")

    print("\nPer-class metrics:")
    name_width = max(len(name) for name in class_names) + 1
    print(f"  {'class':<{name_width}} precision  recall    f1        support")
    for name in class_names:
        m = metrics["per_class"][name]
        print(
            f"  {name + ':':<{name_width}} "
            f"{m['precision']:<10.4f}{m['recall']:<10.4f}{m['f1']:<10.4f}{m['support']}"
        )

    print("\nConfusion matrix (rows = true label, columns = predicted label):")
    cm = metrics["confusion_matrix"]
    header = "        " + " ".join(f"{name[:8]:>8}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join(f"{val:>8}" for val in cm[i])
        print(f"{name[:8]:>8} {row}")


def main():
    config.set_seed()
    device = config.get_device()

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {config.BEST_MODEL_PATH}. "
            "Run `python -m training.train` (and optionally "
            "`python -m training.fine_tune`) first."
        )

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    cfg_snapshot = checkpoint["config"]
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val accuracy: {checkpoint['val_accuracy'] * 100:.2f}%)")

    model = build_model(
        num_classes=cfg_snapshot["num_classes"],
        dropout=cfg_snapshot["dropout"],
        hidden_units=cfg_snapshot["hidden_units"],
        pretrained=False,  # weights are overwritten by the checkpoint below
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Rebuilding the dataloaders reproduces the exact same test split
    # (fixed random seed), so evaluation never leaks train/val data.
    _, _, test_loader, split_class_names, _ = get_dataloaders()
    assert split_class_names == class_names, (
        "Class list from the dataset does not match the checkpoint's class "
        "list. Did the dataset change since training?"
    )

    y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    print_report(metrics, class_names)

    # Save the confusion matrix image.
    cm_path = os.path.join(config.EVAL_OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)
    print(f"\nConfusion matrix image saved to {cm_path}")

    # Save the full metrics report as JSON (confusion matrix as a plain list).
    report = {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": class_names,
    }
    report_path = os.path.join(config.EVAL_OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics report saved to {report_path}")


if __name__ == "__main__":
    main()
