# Dataset Info

This folder holds the image data used to train the MoodMatch.Ai emotion
classifier. The training pipeline does **not** hard-code the dataset — it
discovers classes automatically from the folder names under `data/raw/`.

## Expected layout

Place images inside `data/raw/`, organized into one sub-folder per emotion
class:

```text
data/raw/
├── angry/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── happy/
│   ├── img001.jpg
│   └── ...
├── sad/
│   └── ...
└── neutral/
    └── ...
```

- Folder names become the class names (used exactly as-is, case-sensitive).
- You can add or remove class folders freely (e.g. add `surprise`, `fear`,
  `disgust`) — the model's output layer size adapts automatically based on
  however many class folders are found, so nothing in the training code
  needs to change.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp` (anything
  `PIL.Image.open` can read).
- Images can be any size/aspect ratio — they are resized to
  `IMAGE_SIZE x IMAGE_SIZE` (see `training/config.py`) during preprocessing.

## Where classes come from

`training/dataset.py` uses `torchvision.datasets.ImageFolder` to scan
`data/raw/` and build the class list from the sub-folder names, sorted
alphabetically. That exact class list (in that exact order) is saved inside
the model checkpoint at training time, so prediction never depends on
directory ordering at inference time.

## `data/processed/`

Reserved for any pre-processed/cached version of the dataset you may want to
generate later (for example, cropped/aligned faces). Not currently used by
the default training pipeline, which reads directly from `data/raw/`.

## Collecting your own data

If you generate/collect images with a separate script later, just drop them
into the appropriate `data/raw/<class_name>/` folder — no code changes are
required.
