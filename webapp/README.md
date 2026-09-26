# MoodMatch.AI — Game UI

A polished, playable game built on top of the existing trained emotion
model (`models/checkpoints/best_model.pth`). This folder does **not**
change anything about the ML training pipeline — it only reuses the
trained checkpoint through `training/predict.py`.

## How it's organized

```text
webapp/
├── backend/
│   ├── server.py              FastAPI app: routes + static file serving
│   ├── prediction_service.py  Loads the checkpoint ONCE, wraps training/predict.py
│   └── face_utils.py          Quick "is a face visible?" check (OpenCV Haar cascade)
└── frontend/
    ├── index.html             Start / game / game-over screens
    ├── style.css              Glassmorphism game UI + animations
    └── game.js                Game state machine, webcam capture, API calls
```

The ML logic and the game logic are kept fully separate:

- **Prediction service** (`prediction_service.py`) is the only place that
  touches the model. It loads the checkpoint once at startup instead of
  spawning `python -m training.predict` as a new process on every capture.
- **Game logic** (score, hearts, rounds, win/lose rules) lives entirely in
  `game.js`, in the browser. The backend has no game rules at all — it only
  answers "here's a random target" and "here's what emotion this image
  shows".

## Running it

From the project root (`MoodMatch.Ai/`):

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m webapp.backend.server
```

Then open **http://127.0.0.1:8000** in your browser and allow camera access.

> Note: `requirements.txt` pins `opencv-python<5.0.0`. OpenCV 5.x's PyPI
> wheel dropped `cv2.CascadeClassifier` and the bundled Haar cascade files
> that the "no face detected" check relies on, so an older 4.x release is
> required for that feature to work.

## API (used only by the frontend)

- `GET /api/emotions` — the emotion classes the loaded checkpoint supports.
- `GET /api/random-target` — a random `{emotion, emoji, image_url, instruction}`
  for the next round, sampled from `data/raw/<emotion>/`.
- `POST /api/predict` — accepts a `multipart/form-data` image file (a webcam
  capture) and returns:
  - `{"status": "ok", "predicted_emotion": ..., "confidence": ..., "probabilities": {...}}`
  - `{"status": "no_face"}` — no heart is lost for this in the game
  - `{"status": "multiple_faces"}` — no heart is lost for this either
  - `{"status": "error", "message": ...}` — technical failure, no heart lost

## Game rules (implemented client-side in `game.js`)

- 3 hearts, start score 0.
- Each round is gated on the player clicking **I'M READY**. There is no
  "MATCH!" button during the round itself — clicking I'M READY starts a
  3-2-1-MATCH! countdown on its own, and at the end of it the webcam frame
  freezes and is sent for prediction.
- After every result (match, miss, or a no-face/error case) the game
  pauses and shows **I'M READY** again — it never restarts a round on its
  own, the player decides when the next attempt happens.
- Correct match → `+1` score, confetti, then a new random target is loaded
  and I'M READY reappears.
- Wrong match → `-1` heart, the "NOT A MATCH" box clears, a new random
  target is loaded, and I'M READY reappears — until hearts run out.
- `no_face` / `multiple_faces` / `error` → shown to the player, **no** heart
  lost, and I'M READY reappears so the player can reposition and try again.
- 0 hearts → Game Over screen with final score and a **Try Again** button
  that resets score to 0, hearts to 3, and starts a new round.
- The camera can be turned on/off at any time via the 📷 icon in the header.
  Turning it off hides the I'M READY button; turning it back on shows it
  again so the player can resume.

## Face detection tuning (`face_utils.py`)

Real webcam frames get a two-tier detection pass: a balanced first pass,
and only if that finds *nothing* does a much more sensitive fallback pass
run (capped at reporting at most one face, since that pass is noisy enough
that a spurious second detection is far more likely than a real second
person). This significantly cuts down both "no face detected" and "too
many faces" false alarms compared to a single fixed set of parameters.
