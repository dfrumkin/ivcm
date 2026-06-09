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
the single switch between them. Inputs are single-channel `384×384` images. At present
the data is **synthetic** (auto-generated, with a learnable signal) so the whole
pipeline can be smoke-tested without real data.

```bash
uv run train task=classification     # classification pipeline
uv run train task=segmentation       # segmentation pipeline
```

Anything in the config tree can be overridden from the command line, e.g.:

```bash
# More classes (kept consistent across data + model via a single value):
uv run train task=segmentation num_classes=4

# Shorter run / different device:
uv run train task=classification trainer.max_epochs=2 trainer.accelerator=cpu

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
data module. Augmentation is currently **D4** (90° rotations + flips) via Albumentations.

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
  train.yaml              # root: task switch, num_classes, seed
  task/                   # classification | segmentation (composes the groups below)
  data/ model/ loss/      # per-task components
  optimizer/ scheduler/   # shared components
  trainer/ logger/        # Lightning Trainer + W&B logger (logger=none to disable)
```

## Project layout

```
src/ivcm/
  data/        # synthetic datasets, datamodules, D4 transforms
  models/      # classification + segmentation LightningModules
  utils/       # logging setup
  train.py     # Hydra entry point (`train` console script)
tests/         # smoke tests (fast_dev_run for both tasks)
configs/       # Hydra configuration tree
```
