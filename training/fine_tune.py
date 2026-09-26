"""
Stage 2 - Fine-tuning.

Loads the best checkpoint produced by Stage 1 (training/train.py), unfreezes
the last N feature blocks of the MobileNetV3-Large backbone (early layers
stay frozen), and continues training with a much smaller learning rate
(config.STAGE2_LR) so the pretrained weights aren't destroyed.

Run with:

    python -m training.fine_tune

Must be run after training/train.py has produced
models/checkpoints/best_model.pth.
"""

import os

import torch
import torch.nn as nn
from torch.optim import Adam

from src.model.mobilenet import build_model, print_trainable_summary, unfreeze_last_n_blocks
from training import config
from training.dataset import get_dataloaders
from training.train import run_training_stage


def main():
    config.set_seed()
    device = config.get_device()

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No Stage 1 checkpoint found at {config.BEST_MODEL_PATH}. "
            "Run `python -m training.train` first."
        )

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    cfg_snapshot = checkpoint["config"]
    print(f"Loaded Stage 1 checkpoint (val accuracy: {checkpoint['val_accuracy'] * 100:.2f}%)")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Rebuild the exact same architecture used in Stage 1, then load its
    # trained weights.
    model = build_model(
        num_classes=cfg_snapshot["num_classes"],
        dropout=cfg_snapshot["dropout"],
        hidden_units=cfg_snapshot["hidden_units"],
        pretrained=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Stage 2: unfreeze the last N backbone blocks (keep early layers
    # frozen) plus the classifier head.
    unfreeze_last_n_blocks(model, config.NUM_UNFROZEN_LAYERS)
    print_trainable_summary(model)

    # Re-build dataloaders. Because the split uses a fixed random seed, this
    # produces the exact same train/val/test split as Stage 1.
    train_loader, val_loader, test_loader, split_class_names, class_weights = get_dataloaders()
    assert split_class_names == class_names, (
        "Class list from the dataset does not match the Stage 1 checkpoint. "
        "Did the dataset change since Stage 1 was trained?"
    )

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=config.STAGE2_LR)

    run_training_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=config.STAGE2_EPOCHS,
        class_names=class_names,
        stage_name="Stage 2 (fine-tuning)",
        initial_best_accuracy=checkpoint["val_accuracy"],
    )


if __name__ == "__main__":
    main()
