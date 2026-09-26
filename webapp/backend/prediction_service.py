"""
Thin wrapper around the existing MoodMatch.Ai prediction pipeline.

This module intentionally contains NO model/ML logic of its own. It just
loads the already-trained checkpoint ONCE (instead of spawning a new
`python -m training.predict` process for every webcam capture, which would
be slow) and re-uses the exact same functions the CLI uses:

    training.predict.load_model_for_inference
    training.predict.predict_image

Keeping this file separate from the FastAPI routes (see server.py) keeps
the "ML prediction" concern cleanly separated from the "game/UI" concern,
per the project's architecture requirements.
"""

import io
import os
import sys
from typing import Dict, List, Tuple

from PIL import Image

# Make sure the project root (MoodMatch.Ai/) is importable regardless of the
# working directory the server is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training import config  # noqa: E402
from training.predict import load_model_for_inference, predict_image  # noqa: E402


class PredictionService:
    """Loads the trained checkpoint once and serves predictions from memory."""

    def __init__(self, checkpoint_path: str = None):
        self.checkpoint_path = checkpoint_path or config.BEST_MODEL_PATH
        self.device = config.get_device()
        self.model, self.class_names, self.image_size = load_model_for_inference(
            self.checkpoint_path, self.device
        )
        print(f"[prediction_service] Loaded checkpoint: {self.checkpoint_path}")
        print(f"[prediction_service] Classes (in checkpoint order): {self.class_names}")

    def predict_bytes(self, image_bytes: bytes) -> Tuple[str, float, Dict[str, float]]:
        """Run inference on raw image bytes (e.g. a webcam capture).

        Returns (predicted_label, confidence, probabilities_dict), exactly
        like the CLI's predict_image(), but takes in-memory bytes instead of
        a file path so we don't need to spawn a subprocess per request.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # predict_image() in training/predict.py opens a path with PIL, so we
        # save to a temporary in-memory-backed file only long enough to reuse
        # that exact, already-validated code path unchanged.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            image.save(tmp.name, format="JPEG")
            return predict_image(
                self.model, self.class_names, self.image_size, tmp.name, self.device
            )

    def get_class_names(self) -> List[str]:
        return list(self.class_names)


# Singleton instance, created once when the FastAPI app starts up.
_service: PredictionService = None


def get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service
