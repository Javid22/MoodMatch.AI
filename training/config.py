"""
Central configuration for the MoodMatch.Ai training pipeline.

Every important "knob" for training lives here so it is not scattered
across the codebase. Import this module from the training scripts, e.g.:

    from training import config

and read values like ``config.IMAGE_SIZE``.
"""

import os
import random

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT = the MoodMatch.Ai/ folder (one level up from training/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
EXPORTED_DIR = os.path.join(MODELS_DIR, "exported")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final_model.pth")

# Where evaluation artifacts (confusion matrix image, metrics report) go.
EVAL_OUTPUT_DIR = os.path.join(MODELS_DIR, "checkpoints")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Class names are NOT hard-coded: dataset.py discovers them from the
# sub-folder names inside RAW_DATA_DIR. NUM_CLASSES below is only a default
# used for documentation purposes / building a model before a dataset is
# available. Once a dataset is loaded, the *actual* number of classes always
# comes from `len(class_names)` discovered on disk.
NUM_CLASSES = 4

# Reproducible train/val/test split. Values must sum to 1.0.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Whether to weight the loss function by inverse class frequency to combat
# class imbalance. Toggle this off if your dataset is already balanced.
USE_CLASS_WEIGHTS = True

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # MobileNetV3 expects 224x224 input

# ImageNet normalization stats (required since the backbone is pretrained
# on ImageNet).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation settings (kept mild so facial expressions aren't
# distorted).
AUG_ROTATION_DEGREES = 10
AUG_RESIZED_CROP_SCALE = (0.9, 1.0)
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Model / classifier head
# ---------------------------------------------------------------------------
CLASSIFIER_HIDDEN_UNITS = 256
DROPOUT = 0.4

# ---------------------------------------------------------------------------
# Stage 1 - transfer learning (frozen backbone, train head only)
# ---------------------------------------------------------------------------
STAGE1_EPOCHS = 15
STAGE1_LR = 1e-3

# ---------------------------------------------------------------------------
# Stage 2 - fine-tuning (unfreeze later backbone layers)
# ---------------------------------------------------------------------------
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-5

# How many of the final "feature blocks" of MobileNetV3-Large to unfreeze
# during Stage 2. The backbone (`model.features`) has 17 blocks (indices
# 0-16); a value of 4 unfreezes the last 4 blocks and keeps the rest frozen.
NUM_UNFROZEN_LAYERS = 4

# ---------------------------------------------------------------------------
# Regularization / early stopping
# ---------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE = 5  # epochs with no val-loss improvement

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set the random seed for Python, NumPy, and PyTorch (CPU + CUDA).

    Call this once at the start of every training/evaluation script so
    results (data splits, weight init, augmentation) are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Automatically select CUDA if available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


# Make sure the output directories exist as soon as config is imported.
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, EXPORTED_DIR):
    os.makedirs(_dir, exist_ok=True)
