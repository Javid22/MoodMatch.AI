"""
Stage 1 - Transfer learning.

Freezes the MobileNetV3-Large feature extractor and trains only the new
emotion-classifier head, using a relatively high learning rate
(config.STAGE1_LR) since the head starts from random weights.

Run with:

    python -m training.train

The shared helper functions in this file (train_one_epoch, evaluate,
save_checkpoint, EarlyStopping) are reused by training/fine_tune.py for
Stage 2, so the two stages behave identically apart from which layers are
trainable and which learning rate is used.
"""

import torch
import torch.nn as nn
from torch.optim import Adam

from src.model.mobilenet import build_model, freeze_feature_extractor, print_trainable_summary
from training import config
from training.dataset import get_dataloaders
from training.metrics import AverageMeter, batch_accuracy


class EarlyStopping:
    """Stops training when validation loss hasn't improved for `patience`
    epochs in a row."""

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Update state with the latest val_loss. Returns True if training
        should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_accuracy(outputs, labels), batch_size)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run one evaluation pass (no gradient updates). Returns (avg_loss,
    avg_accuracy)."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_accuracy(outputs, labels), batch_size)

    return loss_meter.avg, acc_meter.avg


def save_checkpoint(path, model, optimizer, epoch, val_accuracy, class_names, cfg_snapshot):
    """Save everything needed to reproduce inference or resume training:
    model weights, class names/order, epoch, optimizer state, val accuracy,
    and the configuration used."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "class_names": class_names,
            "config": cfg_snapshot,
        },
        path,
    )


def get_config_snapshot(num_classes: int) -> dict:
    """A plain dict of the config values needed to rebuild the model
    architecture at inference/evaluation time."""
    return {
        "num_classes": num_classes,
        "dropout": config.DROPOUT,
        "hidden_units": config.CLASSIFIER_HIDDEN_UNITS,
        "image_size": config.IMAGE_SIZE,
    }


def run_training_stage(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    class_names,
    stage_name: str,
    initial_best_accuracy: float = 0.0,
):
    """Shared epoch loop used by both Stage 1 (train.py) and Stage 2
    (fine_tune.py). Tracks losses/accuracies, saves the best checkpoint by
    validation accuracy, and applies early stopping on validation loss.

    `initial_best_accuracy` lets Stage 2 carry over Stage 1's best val
    accuracy, so best_model.pth is only overwritten if fine-tuning actually
    improves on it (not simply beats a fresh 0.0 baseline)."""
    early_stopping = EarlyStopping()
    best_val_accuracy = initial_best_accuracy
    cfg_snapshot = get_config_snapshot(len(class_names))

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\n[{stage_name}] Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}  Train Accuracy: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}  Val Accuracy:   {val_acc * 100:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_checkpoint(
                config.BEST_MODEL_PATH, model, optimizer, epoch, val_acc, class_names, cfg_snapshot
            )
            print(f"  -> New best model saved (val accuracy {val_acc * 100:.2f}%)")

        if early_stopping.step(val_loss):
            print(f"\nEarly stopping triggered after {epoch} epochs (no val-loss improvement).")
            break

    # Always save the final-epoch state too, separate from the best one.
    save_checkpoint(
        config.FINAL_MODEL_PATH, model, optimizer, epoch, val_acc, class_names, cfg_snapshot
    )
    print(f"\nFinal {stage_name} checkpoint saved to {config.FINAL_MODEL_PATH}")
    print(f"Best {stage_name} val accuracy: {best_val_accuracy * 100:.2f}%")

    return best_val_accuracy


def main():
    config.set_seed()
    device = config.get_device()

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders()
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(
        num_classes=len(class_names),
        dropout=config.DROPOUT,
        hidden_units=config.CLASSIFIER_HIDDEN_UNITS,
        pretrained=True,
    ).to(device)

    # Stage 1: freeze the backbone, train only the classifier head.
    freeze_feature_extractor(model)
    print_trainable_summary(model)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only the classifier head has requires_grad=True at this point, but we
    # filter explicitly to be safe.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=config.STAGE1_LR)

    run_training_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=config.STAGE1_EPOCHS,
        class_names=class_names,
        stage_name="Stage 1 (transfer learning)",
    )


if __name__ == "__main__":
    main()
