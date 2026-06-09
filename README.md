# ivcm

Medical image classification and segmentation.

## Requirements

- [uv](https://docs.astral.sh/uv/) for environment and dependency management.

`uv` provisions the pinned Python interpreter (`.python-version`) automatically.

## Setup

```bash
uv sync                     # create .venv and install dependencies
uv run pre-commit install   # enable git pre-commit hooks
```

## Development

```bash
uv run pytest               # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pre-commit run --all-files
```

## Training

Training is built on **PyTorch Lightning** with **Hydra** configuration. Both a
classification and a segmentation pipeline are supported; the `task` config group is
the single switch between them. Inputs are single-channel images resized to
`${image_size}` (default `384×384`).

### Data

The real data is a **proprietary dataset** under `data/` (gitignored, not committed):

- **Classification** — 4 classes (`endothelium`, `epithelium`, `nerves`, `stroma`) with a
  fixed `train/`/`val/` split and no test set. Reports per-class **sensitivity,
  specificity, and AUC** on the validation set.
- **Segmentation** — binary nerve-vs-background, from a single volume pair. The slices are
  split into **5 contiguous folds** for cross-validation (4 train / 1 val per run);
  reports per-fold and **mean** sensitivity/specificity/AUC.

For each task the **default data is synthetic** (auto-generated, learnable signal), so the
whole pipeline runs and is smoke-tested with **no data present**. Real data is opt-in.

```bash
# Synthetic defaults — no data required (smoke/sanity):
uv run train task=classification
uv run train task=segmentation

# Real classification (fixed train/val split; per-class metrics on val):
uv run train task=classification data=classification_real

# Real segmentation 5-fold cross-validation (per-fold + mean metrics):
uv run cross-validate
```

`task` has **no default** — `uv run train` errors until a task is chosen. The proprietary
data lives at `./data` by default; set `IVCM_DATA_DIR` to point elsewhere (absolute path).

Anything in the config tree can be overridden from the command line, e.g.:

```bash
# Shorter run / different device / input size:
uv run train task=classification trainer.max_epochs=2 trainer.accelerator=cpu image_size=256

# Fastest sanity check (one train + val batch; Lightning suppresses logging
# and checkpointing in this mode, so no W&B run is initialized):
uv run train task=classification +trainer.fast_dev_run=true

# Disable experiment logging for a full run (no W&B, no fallback logger):
uv run train task=classification logger=none

# Swap components (everything is selected via Hydra `_target_`):
uv run train task=classification model.net.model_name=resnet34 optimizer.lr=1e-4
```

What is configurable (all via `_target_`): architecture (`timm` for classification,
`segmentation-models-pytorch` for segmentation), loss, optimizer, scheduler, and the
datasets/data module. The class set is driven by a single `class_names` list per task (the
count is derived as `${len:${class_names}}`). Augmentation is currently **D4** (90°
rotations + flips) plus a resize to `${image_size}`, via Albumentations.

### Outputs, logging, and tracking

- Each run gets a Hydra output directory under `outputs/<date>/<time>/` containing a
  `train.log` file. Console output is rendered with **Rich**.
- Experiment tracking uses **Weights & Biases**, **offline by default** (no login
  needed). To sync to the W&B server:

  ```bash
  uv run train task=classification logger.mode=online   # requires wandb login / WANDB_API_KEY
  ```

  Offline runs can be uploaded later with `wandb sync <run-dir>`. To turn logging off
  entirely, use `logger=none`.

## Configuration layout

```
configs/
  common.yaml             # shared root: seed, image_size, data_dir, trainer/logger defaults
  train.yaml              # root for `train`: includes common, mandatory `task: ???`
  cv.yaml                 # root for `cross-validate`: segmentation + real CV data
  task/                   # classification | segmentation (class_names + composes groups)
  data/                   # <task> = synthetic (default), <task>_real = proprietary data
  model/ loss/            # per-task components
  optimizer/ scheduler/   # shared components
  trainer/ logger/        # Lightning Trainer + W&B logger (logger=none to disable)
```

## Project layout

```
src/ivcm/
  data/        # real (.tif/.nii) + synthetic datasets, datamodules, D4+resize transforms
  models/      # classification + segmentation LightningModules (incl. metrics)
  utils/       # logging setup
  train.py     # Hydra entry point (`train` console script)
  cv.py        # segmentation cross-validation (`cross-validate` console script)
tests/         # fold + config + metric-key + smoke tests; real-data tests (skip if no data)
configs/       # Hydra configuration tree
```
