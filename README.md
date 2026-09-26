# MoodMatch.Ai

This project trains a facial emotion classifier using **PyTorch + transfer
learning + fine-tuning** on top of **MobileNetV3-Large** (pretrained on
ImageNet), and includes a playable webcam game built on top of it.

- `training/`, `src/model/`, `data/`, `models/` — the ML training module
  (see below).
- `webapp/` — the **MoodMatch.AI game**: a webcam game where you try to
  match a target emotion before the AI catches you. See
  [`webapp/README.md`](webapp/README.md) for how to run it.

The rest of this README covers the ML training module.

## 1. What this project does

- Loads a folder of face images organized by emotion (e.g. `angry`,
  `happy`, `sad`, `neutral`, ...).
- Builds a MobileNetV3-Large model with a custom classifier head.
- Trains it in two stages:
  1. **Stage 1 (transfer learning)** — the pretrained backbone is frozen,
     only the new classifier head is trained.
  2. **Stage 2 (fine-tuning)** — the last few backbone layers are unfrozen
     and trained with a much smaller learning rate.
- Evaluates the trained model on a held-out test set (accuracy, precision,
  recall, F1, confusion matrix).
- Predicts the emotion in a single image from the command line.

## 2. Install dependencies

Requires Python 3.9+.

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Place the dataset

Put your images under `data/raw/`, one sub-folder per emotion class:

```text
data/raw/
├── angry/
├── happy/
├── sad/
└── neutral/
```

See [`data/dataset_info.md`](data/dataset_info.md) for full details. Class
names are **not hard-coded** — they're discovered automatically from the
folder names, so you can add/remove/rename emotion folders freely.

## 4. Train Stage 1 (transfer learning)

Freezes the MobileNetV3 backbone and trains only the new classifier head.

```bash
python -m training.train
```

This will:
- Print the device being used (`Using device: CUDA` or `Using device: CPU`).
- Print the class distribution and a reproducible 70/15/15 train/val/test
  split (percentages configurable in `training/config.py`).
- Print which layers are frozen vs. trainable.
- Train for `STAGE1_EPOCHS` epochs, printing train/val loss & accuracy each
  epoch.
- Save the best checkpoint to `models/checkpoints/best_model.pth` and the
  final-epoch checkpoint to `models/checkpoints/final_model.pth`.

## 5. Fine-tune Stage 2

Run **after** Stage 1. Loads `best_model.pth`, unfreezes the last
`NUM_UNFROZEN_LAYERS` backbone blocks (early layers stay frozen), and
continues training with a smaller learning rate.

```bash
python -m training.fine_tune
```

`best_model.pth` is only overwritten if fine-tuning actually beats Stage 1's
best validation accuracy.

## 6. Evaluate the model

Loads `best_model.pth` and evaluates it on the test split.

```bash
python -m training.evaluate
```

Prints test accuracy, macro precision/recall/F1, per-class metrics, and a
confusion matrix (which emotions get confused with each other). Also saves:

- `models/checkpoints/confusion_matrix.png`
- `models/checkpoints/evaluation_report.json`

## 7. Run prediction on one image

```bash
python -m training.predict --image path/to/face.jpg
```

Example output:

```text
Predicted emotion: Happy
Confidence: 91.42%

Probabilities:
Happy: 91.42%
Neutral: 4.12%
Sad: 2.31%
Angry: 1.05%
```

Class ordering is saved inside the checkpoint at training time, so
prediction never depends on directory ordering.

## 8. Where the trained model is saved

```text
models/checkpoints/best_model.pth    <- best model by validation accuracy
models/checkpoints/final_model.pth   <- last-epoch model of the most recent stage
models/exported/                     <- reserved for future export formats (e.g. ONNX)
```

Each checkpoint contains: model weights, class names (in order), epoch
number, optimizer state, validation accuracy, and the model configuration
needed to rebuild it.

## 9. CPU / GPU selection

Device selection is automatic — no configuration needed:

```python
import torch
torch.cuda.is_available()  # True -> uses CUDA, False -> uses CPU
```

Every script prints which device it's using at startup.

## Project structure

```text
MoodMatch.Ai/
├── data/
│   ├── raw/              <- put your dataset here (one folder per class)
│   ├── processed/        <- reserved for pre-processed data
│   └── dataset_info.md
├── models/
│   ├── checkpoints/      <- best_model.pth, final_model.pth, eval reports
│   └── exported/         <- reserved for future export formats
├── training/
│   ├── train.py          <- Stage 1: transfer learning
│   ├── fine_tune.py      <- Stage 2: fine-tuning
│   ├── evaluate.py       <- test-set evaluation
│   ├── predict.py        <- single-image prediction
│   ├── dataset.py        <- dataset loading, splitting, class balance
│   ├── transforms.py     <- train/val image preprocessing pipelines
│   ├── config.py         <- all configurable settings
│   └── metrics.py        <- metric tracking + evaluation metrics
├── src/
│   └── model/
│       └── mobilenet.py  <- model builder + freeze/unfreeze helpers
├── notebooks/            <- reserved for exploratory notebooks
├── requirements.txt
└── README.md
```

## Configuration

All key settings (image size, batch size, learning rates, epochs, dropout,
split ratios, random seed, etc.) live in [`training/config.py`](training/config.py)
— nothing important is hard-coded elsewhere.
