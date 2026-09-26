"""
Image preprocessing / augmentation pipelines.

Training and validation/test images are treated differently:

- Training transforms include mild augmentation to help the model
  generalize, without distorting facial expressions.
- Validation/test transforms only resize + normalize, so evaluation numbers
  reflect the model's real performance.

Both pipelines end with ImageNet normalization because MobileNetV3's
backbone was pretrained on ImageNet.
"""

from torchvision import transforms

from training import config


def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Transform pipeline used for the training split.

    Includes mild, expression-safe augmentation:
      - random horizontal flip
      - small random rotation
      - small random-resized-crop (slight zoom/crop variation)
      - small brightness/contrast jitter
    """
    return transforms.Compose(
        [
            # Slight crop/resize variation instead of a plain Resize, but the
            # scale range is kept close to 1.0 so faces aren't over-cropped.
            transforms.RandomResizedCrop(
                image_size,
                scale=config.AUG_RESIZED_CROP_SCALE,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=config.AUG_BRIGHTNESS,
                contrast=config.AUG_CONTRAST,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Transform pipeline used for validation/test/inference.

    No augmentation - just a deterministic resize + normalize so results are
    consistent and reproducible.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )
