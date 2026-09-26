"""
Run a single-image prediction using the best saved checkpoint.

Usage:

    python -m training.predict --image path/to/face.jpg

Class ordering is read directly from the checkpoint (saved during training),
so predictions never depend on directory ordering at inference time.
"""

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from src.model.mobilenet import build_model
from training import config
from training.transforms import get_val_transforms


def load_model_for_inference(checkpoint_path: str, device: torch.device):
    """Load a checkpoint and rebuild the exact model architecture it was
    trained with. Returns (model, class_names, image_size)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_snapshot = checkpoint["config"]
    class_names = checkpoint["class_names"]

    model = build_model(
        num_classes=cfg_snapshot["num_classes"],
        dropout=cfg_snapshot["dropout"],
        hidden_units=cfg_snapshot["hidden_units"],
        pretrained=False,  # weights come from the checkpoint, not ImageNet
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, cfg_snapshot["image_size"]


@torch.no_grad()
def predict_image(model, class_names, image_size, image_path, device):
    """Run inference on a single image and return (predicted_label,
    confidence, probabilities_dict) sorted by descending probability."""
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transforms(image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    logits = model(input_tensor)
    probabilities = F.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = int(probabilities.argmax())
    predicted_label = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    # Sort classes by probability, highest first, for display purposes.
    prob_dict = dict(sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True))

    return predicted_label, confidence, prob_dict


def main():
    parser = argparse.ArgumentParser(description="Predict the emotion in a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        default=config.BEST_MODEL_PATH,
        help="Path to a model checkpoint (default: models/checkpoints/best_model.pth).",
    )
    args = parser.parse_args()

    device = config.get_device()
    model, class_names, image_size = load_model_for_inference(args.checkpoint, device)

    predicted_label, confidence, prob_dict = predict_image(
        model, class_names, image_size, args.image, device
    )

    print(f"\nPredicted emotion: {predicted_label.capitalize()}")
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nProbabilities:")
    for label, prob in prob_dict.items():
        print(f"{label.capitalize()}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
