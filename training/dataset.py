"""
Dataset loading, splitting, and class-imbalance utilities.

The dataset path and class names are intentionally NOT hard-coded:
- Classes are discovered from the sub-folder names under the given root
  (via ``torchvision.datasets.ImageFolder``).
- The root folder can be swapped out via the ``data_dir`` argument, so this
  module keeps working if the dataset is regenerated or moved later.

This module implements a single reproducible 70/15/15 (configurable)
train/val/test split with per-split transforms and (optionally)
class-balanced sample weights for the loss function.
"""

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from training import config


class EmotionDataset(Dataset):
    """Wraps a subset of samples from an ImageFolder with its own transform.

    ImageFolder applies a single transform to every sample, but we need
    different transforms for train vs. val/test on the *same* underlying
    images. This thin wrapper stores (path, label) pairs plus a transform,
    and loads images from disk lazily.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def discover_classes(data_dir: str = config.RAW_DATA_DIR) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Scan ``data_dir`` and return (samples, class_names).

    class_names is the sorted list of sub-folder names (this is exactly the
    order torchvision.datasets.ImageFolder assigns numeric labels in), so
    saving this list alongside the model is enough to make label <-> name
    mapping deterministic at inference time.
    """
    base = ImageFolder(root=data_dir)
    return base.samples, base.classes


def print_class_distribution(samples: List[Tuple[str, int]], class_names: List[str]) -> None:
    """Print how many images belong to each class, e.g.:

        angry:   1234
        happy:   2345
        sad:     1111
        neutral: 1890
    """
    counts = {name: 0 for name in class_names}
    for _, label in samples:
        counts[class_names[label]] += 1

    print("\nClass distribution:")
    name_width = max(len(name) for name in class_names) + 1
    for name in class_names:
        print(f"  {name + ':':<{name_width}} {counts[name]}")
    print()


def stratified_split(
    samples: List[Tuple[str, int]],
    train_split: float = config.TRAIN_SPLIT,
    val_split: float = config.VAL_SPLIT,
    test_split: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
):
    """Split samples into train/val/test while preserving class balance.

    Uses sklearn's stratified train_test_split (twice) with a fixed seed so
    the split is reproducible. Percentages are configurable and must sum to
    (approximately) 1.0.
    """
    total = train_split + val_split + test_split
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")

    labels = [label for _, label in samples]

    # First split off the training set.
    train_samples, remaining_samples, train_labels, remaining_labels = train_test_split(
        samples,
        labels,
        train_size=train_split,
        random_state=seed,
        stratify=labels,
    )

    # Split the remainder into val/test, keeping their relative proportion.
    relative_val_size = val_split / (val_split + test_split)
    val_samples, test_samples = train_test_split(
        remaining_samples,
        train_size=relative_val_size,
        random_state=seed,
        stratify=remaining_labels,
    )

    return train_samples, val_samples, test_samples


def compute_class_weights(samples: List[Tuple[str, int]], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Classes with fewer samples get a higher weight so the loss doesn't
    just optimize for the majority class(es).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in samples:
        counts[label] += 1

    # Avoid division by zero for any class with 0 samples.
    counts = np.clip(counts, a_min=1, a_max=None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    data_dir: str = config.RAW_DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
):
    """Build train/val/test DataLoaders plus metadata.

    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    from training.transforms import get_train_transforms, get_val_transforms

    samples, class_names = discover_classes(data_dir)
    print_class_distribution(samples, class_names)

    train_samples, val_samples, test_samples = stratified_split(samples)

    train_dataset = EmotionDataset(train_samples, transform=get_train_transforms())
    val_dataset = EmotionDataset(val_samples, transform=get_val_transforms())
    test_dataset = EmotionDataset(test_samples, transform=get_val_transforms())

    print(
        f"Split sizes -> train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_samples, len(class_names))
        print(f"Using class-weighted loss: {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_names, class_weights
