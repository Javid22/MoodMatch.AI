# Training Pipeline Guide

This document explains what each file in the MoodMatch.Ai ML training
module does, and the exact commands to run the pipeline end-to-end.

It complements the top-level [README.md](../README.md) (quick-start
commands) with a per-file breakdown of *why* each file exists and how the
pieces fit together.

## Dataset check performed

Before writing this guide, every image under `data/raw/` was validated
using the same read path `training/dataset.py` uses
(`PIL.Image.open(...).convert("RGB")`), plus a check against the file
extensions `torchvision.datasets.ImageFolder` actually scans for
(`.jpg .jpeg .png .ppm .bmp .pgm .tif .tiff .webp`).

Result:

| Action | Files | Reason |
|---|---|---|
| Deleted | `angry/angry10.avif`, `happy/happy1.avif`, `happy/happy2.avif`, `neutral/neutral5.avif`, `sad/sad0.avif`, `sad/sad4.avif` | `.avif` is not decodable by plain Pillow (no AVIF plugin installed) **and** is not in `ImageFolder`'s recognized extension list — the training pipeline could never see these files. |
| Renamed | `angry/angry3` → `angry/angry3.jpg`, `angry/angry4` → `angry/angry4.jpg` | Content was a valid JPEG, but the missing file extension meant `ImageFolder` would silently skip them during scanning. Renaming (not deleting) preserves the data. |
| Left as-is | `sad/sad6.jpg`, `sad/sad7.jpg` | Their actual content is WebP, not JPEG, but Pillow identifies image format from file content, not the extension, so these open correctly despite the misleading name. No functional issue. |

Current per-class image counts in `data/raw/`:

```text
angry:   9
happy:   9
neutral: 9
sad:     8
```

This is a very small dataset (fine for smoke-testing that the pipeline
runs end-to-end, e.g. checking loss decreases and checkpoints save
correctly) but far too small to train a model that will generalize. Add
more images per class before expecting real accuracy.

## File-by-file explanation

### `training/config.py`
Single source of truth for every tunable setting: paths (`RAW_DATA_DIR`,
checkpoint locations, etc.), image size (224), batch size, train/val/test
split ratios (70/15/15), both stages' epoch counts and learning rates
(`1e-3` for Stage 1, `1e-5` for Stage 2), dropout, how many backbone blocks
to unfreeze in Stage 2, early-stopping patience, and the random seed.
Also provides `set_seed()` (seeds Python/NumPy/PyTorch for reproducibility)
and `get_device()` (auto-detects CUDA vs CPU and prints which is used).
Every other file imports from here instead of hard-coding values.

### `training/transforms.py`
Builds the image preprocessing pipelines:
- `get_train_transforms()` — mild augmentation (random horizontal flip,
  small rotation, slight random-resized-crop, small brightness/contrast
  jitter) followed by resize-to-224 and ImageNet normalization. Kept mild
  on purpose so facial expressions aren't distorted.
- `get_val_transforms()` — deterministic resize + ImageNet normalization
  only, no augmentation. Used for validation, test, and prediction so
  results are consistent.

### `training/dataset.py`
Everything related to loading and splitting the dataset:
- `discover_classes()` — scans `data/raw/` with `torchvision.datasets
  .ImageFolder` and returns the (path, label) samples plus the sorted list
  of class names. Classes are **never hard-coded** — add/remove/rename a
  folder under `data/raw/` and this picks it up automatically.
- `print_class_distribution()` — prints how many images are in each class
  (see table above), so class imbalance is visible before training starts.
- `stratified_split()` — reproducible 70/15/15 train/val/test split (ratios
  configurable in `config.py`) using a fixed random seed, keeping class
  balance across all three splits via stratified sampling.
- `compute_class_weights()` — inverse-frequency weights per class, used to
  build a class-weighted `CrossEntropyLoss` when `config.USE_CLASS_WEIGHTS`
  is `True` (helpful since `sad` currently has one fewer image than the
  others).
- `EmotionDataset` — a small `Dataset` wrapper that lets the train split
  and the val/test splits use *different* transforms even though they come
  from the same underlying `ImageFolder` scan.
- `get_dataloaders()` — the one function the training scripts actually
  call; wires all of the above together and returns
  `train_loader, val_loader, test_loader, class_names, class_weights`.

### `src/model/mobilenet.py`
Builds the model and controls which layers are trainable:
- `build_model()` — loads `torchvision.models.mobilenet_v3_large` with
  ImageNet-pretrained weights and replaces its classification head with a
  custom `Linear → ReLU → Dropout → Linear` head sized to the number of
  emotion classes actually found in the dataset.
- `freeze_feature_extractor()` — freezes the entire pretrained backbone
  (`model.features`) for Stage 1, leaving only the new head trainable.
- `unfreeze_last_n_blocks()` — for Stage 2, refreezes everything then
  unfreezes just the last `config.NUM_UNFROZEN_LAYERS` backbone blocks
  (plus the head), so early generic-feature layers stay frozen while later,
  more task-specific layers adapt.
- `print_trainable_summary()` — prints frozen/trainable status per backbone
  block and the classifier, plus a total/trainable parameter count.

### `training/metrics.py`
Metric utilities shared by training and evaluation:
- `AverageMeter` — running average tracker for loss/accuracy within an
  epoch.
- `batch_accuracy()` — accuracy for one batch.
- `compute_classification_metrics()` — accuracy, macro precision/recall/F1,
  per-class precision/recall/F1/support, and the confusion matrix (used by
  `evaluate.py`).
- `plot_confusion_matrix()` — renders the confusion matrix as an annotated
  heatmap PNG.

### `training/train.py` — Stage 1 (transfer learning)
Freezes the backbone and trains only the new classifier head
(`config.STAGE1_LR = 1e-3`). Also defines the shared building blocks reused
by Stage 2:
- `train_one_epoch()` / `evaluate()` — one epoch of training / validation.
- `EarlyStopping` — stops training if validation loss hasn't improved for
  `config.EARLY_STOPPING_PATIENCE` epochs.
- `save_checkpoint()` — saves model weights, class names (in order),
  epoch, optimizer state, validation accuracy, and config into one `.pth`
  file.
- `run_training_stage()` — the epoch loop itself: trains, validates, prints
  `Epoch i/N`, `Train Loss/Accuracy`, `Val Loss/Accuracy` each epoch, saves
  `best_model.pth` whenever validation accuracy improves, applies early
  stopping, and always saves `final_model.pth` at the end.

Run it directly (`python -m training.train`) to execute Stage 1 only.

### `training/fine_tune.py` — Stage 2 (fine-tuning)
Loads `models/checkpoints/best_model.pth` from Stage 1, rebuilds the same
architecture, calls `unfreeze_last_n_blocks()` to unfreeze the last few
backbone blocks, and continues training with a much smaller learning rate
(`config.STAGE2_LR = 1e-5`) using the same `run_training_stage()` loop from
`train.py`. It seeds the "best accuracy so far" from the Stage 1
checkpoint, so `best_model.pth` is only overwritten if fine-tuning actually
beats Stage 1 — an early bad fine-tuning epoch can't regress the saved
best model.

Must be run **after** `training/train.py`.

### `training/evaluate.py`
Loads `best_model.pth`, rebuilds the exact test split (same seed as
training), runs inference over it, and prints test accuracy, macro
precision/recall/F1, and a per-class breakdown. Saves
`models/checkpoints/confusion_matrix.png` (visualizes which emotions get
confused with each other) and `models/checkpoints/evaluation_report.json`
(the same numbers in machine-readable form).

### `training/predict.py`
Command-line single-image inference. Loads a checkpoint, rebuilds the
model, applies the same validation transform used during training, and
prints the predicted emotion, confidence, and the full probability
breakdown across all classes — using the class order stored in the
checkpoint (never dependent on folder-scan order).

## How to run the whole pipeline

From the `MoodMatch.Ai/` project root, with a Python 3.11/3.12 virtual
environment active and dependencies installed:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. **Stage 1 — transfer learning** (trains the new classifier head only):

   ```bash
   python -m training.train
   ```

2. **Stage 2 — fine-tuning** (unfreezes the last few backbone layers, run
   after Stage 1 has produced `models/checkpoints/best_model.pth`):

   ```bash
   python -m training.fine_tune
   ```

3. **Evaluate** on the held-out test split:

   ```bash
   python -m training.evaluate
   ```

4. **Predict** the emotion in one image:

   ```bash
   python -m training.predict --image path/to/face.jpg
   ```

Every script auto-detects CUDA vs. CPU and prints which device it's using
(`Using device: CUDA` / `Using device: CPU`) — no manual configuration
needed.
