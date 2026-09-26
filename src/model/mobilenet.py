"""
MobileNetV3-Large model builder + freeze/unfreeze helpers for transfer
learning and fine-tuning.

    Input face image
          v
    MobileNetV3-Large (feature extractor, `model.features`)
          v
    Feature extraction
          v
    Custom classifier (Linear -> ReLU -> Dropout -> Linear)
          v
    N emotion classes
"""

from typing import List

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(num_classes: int, dropout: float = 0.4, hidden_units: int = 256, pretrained: bool = True) -> nn.Module:
    """Build a MobileNetV3-Large model with a custom emotion-classifier head.

    Args:
        num_classes: number of output emotion classes (kept configurable
            rather than hard-coded so the dataset can grow/shrink freely).
        dropout: dropout probability used in the custom classifier head.
        hidden_units: size of the hidden layer in the custom classifier head.
        pretrained: whether to load ImageNet-pretrained weights for transfer
            learning.

    Returns:
        An nn.Module ready for training.
    """
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    # The original classifier is:
    #   Linear(960 -> 1280) -> Hardswish -> Dropout(0.2) -> Linear(1280 -> 1000)
    # We read the true input feature size from the existing head so this
    # keeps working even if torchvision changes internal dimensions.
    in_features = model.classifier[0].in_features

    # Replace the ImageNet head with our own emotion classifier:
    # Linear -> ReLU -> Dropout -> Linear
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_units, num_classes),
    )

    return model


def freeze_feature_extractor(model: nn.Module) -> None:
    """Freeze every parameter in the backbone (`model.features`).

    Used in Stage 1 (transfer learning): only the new classifier head is
    trained, the pretrained backbone is left untouched.
    """
    for param in model.features.parameters():
        param.requires_grad = False

    # The classifier head is always trainable.
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_n_blocks(model: nn.Module, n: int) -> None:
    """Unfreeze the last ``n`` feature blocks of the backbone for fine-tuning.

    MobileNetV3-Large's `model.features` is a sequential stack of blocks
    (indices 0..len-1). Early blocks learn generic, low-level features
    (edges, textures) and are kept frozen; later blocks learn more
    task-specific features and are unfrozen so they can adapt to
    facial-emotion data. The classifier head remains trainable too.
    """
    blocks = list(model.features.children())
    num_blocks = len(blocks)
    n = max(0, min(n, num_blocks))

    # Make sure everything starts frozen, then unfreeze just the tail end.
    for param in model.features.parameters():
        param.requires_grad = False

    for block in blocks[num_blocks - n :]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


def print_trainable_summary(model: nn.Module) -> None:
    """Print which top-level layers are frozen vs. trainable.

    Helpful for sanity-checking Stage 1 vs. Stage 2 configuration before
    kicking off a (potentially long) training run.
    """
    print("\nLayer freeze status:")

    feature_blocks: List[nn.Module] = list(model.features.children())
    for idx, block in enumerate(feature_blocks):
        trainable = any(p.requires_grad for p in block.parameters())
        status = "TRAINABLE" if trainable else "frozen"
        print(f"  features[{idx}]: {status}")

    classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
    print(f"  classifier: {'TRAINABLE' if classifier_trainable else 'frozen'}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\nTotal parameters: {total_params:,} | "
        f"Trainable: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)\n"
    )
