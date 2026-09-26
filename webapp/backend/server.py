"""
MoodMatch.Ai game server.

This is a thin FastAPI layer that:
  1. Serves the game's frontend (HTML/CSS/JS).
  2. Serves sample dataset images to use as "target emotion" pictures.
  3. Exposes a prediction endpoint that wraps the existing trained model.

It does NOT contain any game rules (score/hearts/round logic) — those live
entirely in the frontend (game.js), per the requested architecture:

    UI -> Game State / Game Logic  (frontend, game.js)
    UI -> Webcam                   (frontend, game.js)
    UI -> Prediction Service       (this file -> prediction_service.py -> training/predict.py)

Run with (from the project root, MoodMatch.Ai/):

    source venv/bin/activate
    pip install fastapi uvicorn python-multipart
    python -m webapp.backend.server

Then open http://127.0.0.1:8000 in a browser.
"""

import os
import random
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from webapp.backend.face_utils import count_faces, extract_largest_face
from webapp.backend.prediction_service import get_service


class NoCacheStaticFiles(StaticFiles):
    """Serves static files but tells the browser to always revalidate.

    Without this, a browser can hold on to an old cached copy of game.js /
    style.css after we update them, so a plain reload during development
    keeps showing already-fixed bugs until the user does a hard refresh.
    `no-cache` still lets the browser use a cached copy, but only after
    checking with the server (via ETag) that it's still current.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(WEBAPP_DIR)

FRONTEND_DIR = os.path.join(WEBAPP_DIR, "frontend")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# When set to "1", every /api/predict call saves the raw webcam frame AND the
# exact cropped face that was fed to the model, so you can open the files and
# see precisely what the model saw. Off by default to avoid filling the disk
# during normal play.
#
#   MOODMATCH_SAVE_CAPTURES=1 python -m webapp.backend.server
SAVE_DEBUG_CAPTURES = os.environ.get("MOODMATCH_SAVE_CAPTURES") == "1"
DEBUG_CAPTURES_DIR = os.path.join(PROJECT_ROOT, "debug_captures")

# When enabled, every prediction that matches the round's target emotion has
# its face crop saved into data/raw/<emotion>/ — the exact layout the
# training pipeline already reads (see data/dataset_info.md) — so playing
# the game builds up a real, webcam-quality dataset that can be folded back
# into training later. Only *correct* predictions are kept: a wrong
# prediction gives no reliable ground-truth label for what expression the
# player actually made, so saving those would risk mislabeling the dataset.
#
#   MOODMATCH_COLLECT_DATA=0 python -m webapp.backend.server   # to disable
COLLECT_TRAINING_DATA = os.environ.get("MOODMATCH_COLLECT_DATA", "1") == "1"

# Fun, emotion-specific prompts shown to the player each round. Purely
# cosmetic game copy — has no effect on the model.
ROUND_INSTRUCTIONS = {
    "angry": [
        "Pretend someone stole your fries!",
        "Show your most serious angry face!",
        "Someone just cut you in line!",
        "Your Wi-Fi just disconnected mid-game.",
    ],
    "happy": [
        "Show your biggest smile!",
        "Pretend you just won the lottery!",
        "It's finally the weekend!",
        "Someone gave you a puppy!",
    ],
    "sad": [
        "Pretend your favorite food is gone.",
        "Show your saddest face.",
        "Your favorite show just got cancelled.",
        "You dropped your ice cream cone.",
    ],
    "neutral": [
        "Don't react!",
        "Keep a completely straight face.",
        "Pretend nothing happened at all.",
        "Show zero emotion. Zero.",
    ],
}

EMOJI = {"angry": "\U0001F620", "happy": "\U0001F60A", "sad": "\U0001F622", "neutral": "\U0001F610"}

app = FastAPI(title="MoodMatch.Ai Game Server")

# Allow the frontend to be opened from any local origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve sample dataset images (used as "target emotion" pictures) read-only.
app.mount("/dataset-images", StaticFiles(directory=DATASET_DIR), name="dataset-images")

# Serve frontend static assets (style.css, game.js, etc.)
app.mount("/static", NoCacheStaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


@app.get("/api/emotions")
def get_emotions():
    """Return the list of emotion classes the model actually supports.

    Read from the checkpoint (via the prediction service) rather than
    hard-coded, so the UI always matches whatever model is loaded.
    """
    service = get_service()
    return {"emotions": service.get_class_names()}


@app.get("/api/random-target")
def random_target():
    """Pick a random emotion + a random sample image for that emotion."""
    service = get_service()
    class_names = service.get_class_names()

    emotion = random.choice(class_names)
    class_dir = os.path.join(DATASET_DIR, emotion)

    try:
        candidates = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        filename = random.choice(candidates)
        image_url = f"/dataset-images/{emotion}/{filename}"
    except (FileNotFoundError, IndexError):
        image_url = None

    instructions = ROUND_INSTRUCTIONS.get(emotion, ["Make this expression!"])

    return {
        "emotion": emotion,
        "emoji": EMOJI.get(emotion, ""),
        "image_url": image_url,
        "instruction": random.choice(instructions),
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), target_emotion: str = Form(None)):
    """Analyze a webcam capture and return the predicted emotion.

    ``target_emotion`` is the round's target (e.g. "angry") sent by the
    frontend purely so debug captures can be tagged with what the player was
    *supposed* to be showing — it has no effect on the prediction itself.

    Response shapes:
      {"status": "ok", "predicted_emotion": ..., "confidence": ..., "probabilities": {...}}
      {"status": "no_face"}
      {"status": "multiple_faces"}
      {"status": "error", "message": "..."}
    """
    try:
        image_bytes = await image.read()

        num_faces = count_faces(image_bytes)
        if num_faces == 0:
            return JSONResponse({"status": "no_face"})
        if num_faces > 1:
            return JSONResponse({"status": "multiple_faces"})

        # Crop out just the detected face (not the whole webcam frame, which
        # includes background/desk/other people) and predict on that alone.
        face_bytes = extract_largest_face(image_bytes)
        if face_bytes is None:
            return JSONResponse({"status": "no_face"})

        # Computed unconditionally (not just under SAVE_DEBUG_CAPTURES below)
        # since COLLECT_TRAINING_DATA further down also needs a unique,
        # sortable id for the filename it saves.
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"

        #if SAVE_DEBUG_CAPTURES:
        # os.makedirs(DEBUG_CAPTURES_DIR, exist_ok=True)
        # # Sanitize target_emotion before it touches a filename — it comes
        # # from the client, so only keep alphanumerics/underscore/hyphen.
        # safe_type = "".join(c for c in (target_emotion or "unknown") if c.isalnum() or c in "_-") or "unknown"
        # raw_path = os.path.join(DEBUG_CAPTURES_DIR, f"{stamp}_{safe_type}_raw.jpg")
        # face_path = os.path.join(DEBUG_CAPTURES_DIR, f"{stamp}_{safe_type}_face.jpg")
        # with open(raw_path, "wb") as f:
        #     f.write(image_bytes)
        # with open(face_path, "wb") as f:
        #     f.write(face_bytes)
        # print(f"[server] Fed to model: {face_path}  (full webcam frame: {raw_path})")

        service = get_service()
        label, confidence, probabilities = service.predict_bytes(face_bytes)

        # Fold correctly-predicted webcam captures back into the training
        # set (see COLLECT_TRAINING_DATA above for why only correct ones).
        # These go into a "webcam/" subfolder inside each class directory
        # rather than mixed in with the original FER2013 files directly —
        # keeps the two sources easy to tell apart at a glance, while still
        # being picked up automatically: ImageFolder (used by dataset.py)
        # scans each class folder recursively, so no training code changes
        # are needed for this nesting.
        target = (target_emotion or "").strip().lower()
        if COLLECT_TRAINING_DATA and target and label == target:
            webcam_dir = os.path.join(DATASET_DIR, label, "webcam")
            os.makedirs(webcam_dir, exist_ok=True)
            sample_path = os.path.join(webcam_dir, f"Webcam_{stamp}.jpg")
            with open(sample_path, "wb") as f:
                f.write(face_bytes)
            print(f"[server] Added correctly-predicted '{label}' sample to training set: {sample_path}")

        return JSONResponse(
            {
                "status": "ok",
                "predicted_emotion": label,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )
    except Exception as exc:  # noqa: BLE001 - surface as a graceful game error
        print(f"[server] Prediction failed: {exc}")
        return JSONResponse({"status": "error", "message": str(exc)})


if __name__ == "__main__":
    import uvicorn

    # Load the model once at startup (instead of lazily on first request) so
    # the very first round doesn't stall waiting for the checkpoint to load.
    get_service()
    uvicorn.run(app, host="127.0.0.1", port=8000)
