## Phase 2 addendum (current phase — supersedes the Phase 1 scope note below)

The ML training pipeline described below is **complete and validated** (see
`training/`, `models/checkpoints/best_model.pth`, `training/PIPELINE_GUIDE.md`).
The project has moved into building the actual application around it:

* `webapp/backend/` — FastAPI server wrapping the trained model
  (`server.py`, `prediction_service.py`, `face_utils.py`).
* `webapp/frontend/` — the MoodMatch.AI game UI (`index.html`, `style.css`,
  `game.js`): webcam capture, round/target flow, score/hearts, and now an
  animated floating-emoji background layer (`#emoji-bg-layer`, populated by
  `initFloatingEmojiBackground()` in `game.js`).

Because of this, the Phase 1 restriction just below ("DO NOT build the
webcam application, frontend, backend, API, game logic, scoring system, or
database") **no longer applies**. UI/UX, game-logic, and backend work on top
of the trained model are all in scope now. Keep the same conventions already
established in `webapp/` (glassmorphism aesthetic, green/pink accent
palette, vanilla JS/CSS with no framework, FastAPI backend) rather than
introducing new frameworks or a different visual style.

The rest of this file is kept as-is as the original Phase 1 spec, for
reference on how the ML module was meant to be (and was) built.

---

Create the **machine-learning training module only** for my project **MoodMatch.Ai**.

### Goal

Build a facial emotion classification model using **PyTorch + transfer learning + fine-tuning with MobileNetV3-Large**.

The model will eventually be used with a live webcam to detect emotions such as:

* angry
* happy
* sad
* neutral

For now, **DO NOT build the webcam application, frontend, backend, API, game logic, scoring system, or database**. I want to complete and validate the ML model first.

---

## 1. Technology

Use:

* Python 3
* PyTorch
* torchvision
* OpenCV
* NumPy
* Pillow
* scikit-learn
* matplotlib

Create a `requirements.txt`.

Do NOT use TensorFlow/Keras for model training.

---

## 2. Project structure

Create:

```text
MoodMatch.Ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset_info.md
│
├── models/
│   ├── checkpoints/
│   └── exported/
│
├── training/
│   ├── train.py
│   ├── fine_tune.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── config.py
│   └── metrics.py
│
├── src/
│   └── model/
│       └── mobilenet.py
│
├── notebooks/
│
├── requirements.txt
└── README.md
```

Keep the code modular and easy to understand.

---

## 3. Dataset

Initially, assume the dataset will be placed in:

```text
data/raw/
```

The dataset will contain images organized by emotion class.

Use a structure such as:

```text
data/raw/
├── angry/
├── happy/
├── sad/
└── neutral/
```

However, design the dataset loader so the dataset path and class names can easily be changed later.

I may generate/collect the image data myself using a separate script later, so **do not hard-code the dataset**.

---

## 4. Model

Use:

```python
torchvision.models.mobilenet_v3_large
```

with pretrained ImageNet weights.

Use transfer learning.

The original ImageNet classification head must be replaced with a custom 7-class emotion classifier.

Conceptually:

```text
Input face image
      ↓
MobileNetV3-Large
      ↓
Feature extraction
      ↓
Custom classifier
      ↓
7 emotion classes
```

Use something similar to:

```text
Linear
↓
ReLU
↓
Dropout
↓
Linear
```

Make the number of classes configurable rather than hard-coded wherever practical.

---

## 5. Training strategy

Implement **two-stage training**.

### Stage 1 — Transfer learning

Freeze the MobileNetV3 feature extractor.

Train only the new classification head.

Use an appropriate learning rate such as:

```text
1e-3
```

### Stage 2 — Fine-tuning

Unfreeze the later MobileNetV3 layers while keeping the early layers frozen.

Use a much smaller learning rate, such as:

```text
1e-5
```

Make the number of unfrozen layers configurable.

Clearly print during training which layers are frozen and which are trainable.

---

## 6. Image preprocessing

Use an input size appropriate for MobileNetV3, such as:

```text
224 × 224
```

Implement training and validation/test transforms separately.

Training should include reasonable augmentation such as:

* random horizontal flip
* small random rotation
* small crop/resize variation
* brightness/contrast variation

Do not use extreme augmentation that could distort facial expressions.

Normalize using the ImageNet normalization values because the backbone uses ImageNet pretrained weights.

---

## 7. Train/validation/test split

Implement a reproducible dataset split.

Use something approximately like:

```text
70% training
15% validation
15% test
```

Make the percentages configurable.

Use a fixed random seed so the split can be reproduced.

Try to maintain class balance when creating the split.

---

## 8. Handle class imbalance

Calculate the number of samples in each emotion class.

Print something like:

```text
angry:  XXXX
happy:   XXXX
sad:     XXXX
neutral: XXXX
```

If there is significant class imbalance, support class-weighted loss using `CrossEntropyLoss(weight=...)`.

Make this optional through configuration.

---

## 9. Training output

During training display:

```text
Epoch 1/20
Train Loss: ...
Train Accuracy: ...
Val Loss: ...
Val Accuracy: ...
```

Track:

* training loss
* validation loss
* training accuracy
* validation accuracy

Save the best model based on validation performance.

Save:

```text
models/checkpoints/best_model.pth
```

Also save the final checkpoint.

The checkpoint should contain enough information to reproduce inference, including:

* model state
* class names
* epoch
* optimizer state
* validation accuracy
* configuration

---

## 10. Avoid overfitting

Implement:

* early stopping
* model checkpointing
* dropout
* data augmentation

Make patience configurable.

---

## 11. Evaluation

Create:

```text
training/evaluate.py
```

It should load the best checkpoint and evaluate it on the test set.

Generate:

* test accuracy
* precision
* recall
* F1 score
* confusion matrix
* per-class metrics

Save the confusion matrix and metrics if practical.

The evaluation must show which emotions the model confuses with each other.

---

## 12. Prediction script

Create:

```text
training/predict.py
```

It should accept a single image path and output something like:

```text
Predicted emotion: Happy
Confidence: 91.42%

Probabilities:
Happy: 91.42%
Neutral: 4.12%
Sad: 2.31%
Angry: 1.05%
```

Make sure the class ordering is saved with the checkpoint so prediction never depends on accidental directory ordering.

---

## 13. CPU/GPU support

Automatically detect:

```python
cuda
```

if available.

Otherwise use:

```text
CPU
```

Print:

```text
Using device: CUDA
```

or:

```text
Using device: CPU
```

Do not assume CUDA exists.

---

## 14. Reproducibility

Create a configurable random seed.

Set seeds for:

* Python
* NumPy
* PyTorch

Make training reproducible as much as reasonably possible.

---

## 15. Configuration

Put important settings in:

```text
training/config.py
```

For example:

```text
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
STAGE1_EPOCHS = ...
STAGE2_EPOCHS = ...
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
DROPOUT = 0.4
RANDOM_SEED = 42
```

Do not scatter these values throughout the code.

---

## 16. Important: model format

Because training is being done with **PyTorch**, initially save the trained model as:

```text
models/checkpoints/best_model.pth
```

Do NOT use `.h5` for the primary PyTorch checkpoint.

Later, I will decide whether I need:

```text
ONNX
```

or another format for deployment.

If an `.h5` file is specifically required later, create a separate conversion/export step rather than changing the PyTorch training pipeline.

---

## 17. Code quality

The code should be:

* beginner-friendly
* well commented
* modular
* easy to modify
* not unnecessarily complicated

Explain important sections with comments.

Do not introduce unnecessary frameworks.

---

## 18. README

Create a README explaining:

1. What the project does
2. How to install dependencies
3. How to place the dataset
4. How to train Stage 1
5. How to fine-tune Stage 2
6. How to evaluate the model
7. How to run prediction on one image
8. Where the trained model is saved
9. How CPU/GPU selection works

Include exact commands.

---

## Final requirement

At this stage, I only want a **working PyTorch MobileNetV3-Large transfer-learning/fine-tuning training pipeline**.

Do not implement:

* webcam
* real-time camera processing
* game
* random emotion challenges
* scoring
* leaderboard
* frontend
* backend
* database
* user authentication

Those will be implemented later after the emotion model is working properly.

Before writing code, inspect the requested folder structure and then implement the ML portion cleanly.
