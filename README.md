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
uvx pyright                 # static type check (strict; see docs/coding-standards.md)
uv run pre-commit run --all-files
```

The codebase is clean under Pyright/Pylance **strict** mode. See
[docs/coding-standards.md](docs/coding-standards.md) for the typing conventions and how to
handle type-checker diagnostics.

The desktop UI (below) lives under `src/ivcm/app/` and imports **PySide6**, which is an
**optional** dependency (the `ui` extra) so headless training/inference runtimes need not install
Qt. Because Pyright (strict) scans all of `src`, run linting/type-checking/tests with the extra
synced so the Qt modules resolve:

```bash
uv sync --extra ui          # install PySide6 for the dev/typecheck environment
uv run ruff check .
uvx pyright
uv run pytest
```

`tests/test_app.py` itself only imports the Qt-free modules, so it also passes in the base
environment.

## Clinical review UI

A simple PySide6 desktop app for reviewing a directory of images. The doctor starts with **Load**
(highlighted until data is loaded): choose an image directory, then answer **"Load or create
labels?"** — **Create** runs the pipeline afresh, **Load** re-opens results saved earlier for that
directory (see [Saving and loading labels](#saving-and-loading-labels) below), and **Cancel** steps
back. When creating, the app classifies each image, segments the predicted-"nerve" ones, runs the
existing morphometry (`ivcm.morphometry`), and shows a folder-like table — **name, density, area,
perimeter** (in
physical units derived from the configured `physical_image_size`, not raw pixels) — with a per-row
**Use** checkbox (the manual nerve / not-nerve label), a **Pin** checkbox before it, and mask
overlays. Pinned rows sort to the very top (before the other checked rows); pinning an image also
checks **Use** (segmenting it if needed), and at most three images may be pinned (a fourth attempt
just warns). Unchecking **Use** also unpins. Selecting a row previews the overlay on the side;
above that preview a compact panel summarises the checked images — the count, plus a small
distribution strip per metric (over a `0`-to-max axis) marking the mean, ±1 SD, and min/max, with
the exact numbers on hover. Double-click / Enter opens an image large in a blocking window. A
second window stays in sync showing the top-3 checked images (pinned first) with their metrics and,
per image, a pin toggle (**Unpin** when pinned, **Pin** otherwise — pinning here obeys the same
three-pin limit) and a **Don't Use** button. Checking a non-nerve
image segments it on the spot. **Edit** under the preview opens an in-place mask
editor — paint with the left mouse button, erase with the right, adjust the brush size, and undo/redo
per stroke, with **Keep** / **Discard** icons (the rest of the UI is frozen during an edit). Over the
image the cursor becomes a brush circle sized to the brush and colored by function (pink to paint,
cyan to erase). Keep
recomputes that image's metrics, marks it a nerve, and re-sorts its row; Discard reverts (asking to
confirm if the mask changed). Everything is preconfigured except the directory.

```bash
uv sync --extra ui          # install the UI dependency
uv run ivcm-ui              # launch the app (from this checkout)
```

### Saving and loading labels

**Save** (or Cmd+S) writes the current review — the table and the masks — to a *label directory*.
The first save this session is a "Save As": pick where the new directory goes and type its name
(suggested `<image-dir>_labels`, or the loaded directory's location/name when labels were loaded —
the first save always asks in case the load source was read-only). Later saves go straight back to
that directory. If the chosen target already exists and is a label directory the app made, it offers
**Overwrite**/**Cancel**; if it holds anything else it is left untouched (a single **OK**, then pick
again). A label directory contains:

- `summary.csv` — one row per reviewed image: `Pin,Use,Name,Density,Area,Perimeter` (images that
  failed to load/classify are skipped and have no row).
- `masks/` — one single-channel PNG per segmented image (`0` background, `255` foreground), named
  after the image (`foo.tif` → `masks/foo.tif.png`).

**Load** → **Load labels** re-opens such a directory against the chosen image directory. The names
and mask resolutions must correspond to the images; a structurally invalid directory reports
**"Bad Format"** and a non-corresponding one reports an error, each returning to the chooser. Loaded
metrics come straight from `summary.csv` (no re-segmentation), so manual mask edits are preserved.

Closing the app (window close or **Quit**), or starting a new **Load**, while there are unsaved
changes — new or edited masks, `Pin`/`Use` changes, or an in-progress edit — prompts
**"Save changes?"** with **Save** (finishing any open edit first), **Discard**, or **Cancel** (stay
where you are). The status strip along the bottom shows the last save (with a timestamp) for the
current dataset, and is cleared when a new dataset is loaded.

**Checkpoints are preconfigured, not committed.** Trained checkpoints and `outputs/` are gitignored
local artifacts, so the app is **not runnable on a clean checkout** — edit the two `(run_dir, ckpt)`
pairs in [`src/ivcm/app/config.py`](src/ivcm/app/config.py) to point at this machine's trained runs
(and tune the morphometry parameters `physical_image_size`, `sigma`, and `min_size` there if needed):

```python
CONFIG = AppConfig(
    classification=ModelConfig(
        run_dir="outputs/<your-classification-run>",                 # holds .hydra/config.yaml
        ckpt="outputs/<your-classification-run>/checkpoints/<ckpt>.ckpt",
    ),
    segmentation=ModelConfig(
        run_dir="outputs/<your-segmentation-run>",                   # holds .hydra/config.yaml
        ckpt="outputs/<your-segmentation-run>/fold_0/checkpoints/<ckpt>.ckpt",
    ),
)
```

Each `run_dir` must contain `.hydra/config.yaml` (the saved Hydra config, used to rebuild the
model), and each `ckpt` must point at a real checkpoint under that run. If a path is missing — or
the configured models are miswired — the app shows a clear error dialog at startup and exits.

#### Using a trained model in the UI

When you train on a remote machine (e.g. a CUDA box) but run the UI elsewhere (e.g. a Mac), the
easiest path is the **weights-only W&B artifact** each run uploads (see
[Weights-only best-checkpoint artifact](#outputs-logging-and-tracking)). It is self-contained, so
**nothing else needs to be copied** from the training machine:

1. Pick the runs from W&B — the classification run, and (from a CV) the fold with the best
   `val/auprc` (the run **summary** is only for this choice; the UI does not read it).
2. Download each run's `weights-<run id>` artifact. A downloaded artifact directory holds
   `best-weights.ckpt` and `.hydra/config.yaml`.
3. Point `config.py` at those two directories, using **absolute (or full repo-relative) paths for
   both `run_dir` and `ckpt`** — `ModelConfig` resolves a relative path against the repo root, not
   against `run_dir`, so a bare `ckpt="best-weights.ckpt"` would look under the repo root:

   ```python
   segmentation=ModelConfig(
       run_dir="/abs/path/to/weights-<seg-run-id>",              # holds .hydra/config.yaml
       ckpt="/abs/path/to/weights-<seg-run-id>/best-weights.ckpt",
   ),
   ```

The weights are stored on CPU, so a model trained on CUDA loads directly on the Mac's MPS/CPU. The
UI machine still needs a compatible `ivcm` checkout: the artifact carries the weights and Hydra
config, but the model *class definitions* come from the installed package.

### App icon

The app sets a window/taskbar icon (the dendritic-cell artwork) on every platform via
`QApplication.setWindowIcon`. The assets live under
[`src/ivcm/app/assets/icons/`](src/ivcm/app/assets/icons/): the committed `icon.png` (1024×1024) is
the single source of truth, and [`scripts/make_icons.py`](scripts/make_icons.py) regenerates the
derived per-platform deliverables from it (run `uv run python scripts/make_icons.py` when the
artwork changes; the generated files are committed). The `.icns` step needs macOS `iconutil` and is
skipped with a warning elsewhere.

A future bundler/installer (none exists today) uses, per platform: `icon.icns` (macOS),
`icon.ico` (Windows), and the freedesktop `linux/hicolor/<size>x<size>/apps/ivcm-ui.png` tree plus
`ivcm-ui.desktop` (Linux). To install on Linux by hand, copy the `hicolor` tree under
`~/.local/share/icons` (or `/usr/share/icons`) and `ivcm-ui.desktop` into the matching
`applications/` directory (adjusting its `Exec=` if `ivcm-ui` is not on `PATH`).

## Training

Training is built on **PyTorch Lightning** with **Hydra** configuration. Both a
classification and a segmentation pipeline are supported; the `task` config group is
the single switch between them. Inputs are single-channel images. **Classification** resizes
each image to `${image_size}` (default `384×384`). **Segmentation** trains on native-resolution
random `${crop_size}` crops (default `192×192`) and runs inference with **sliding-window
inference** over `${crop_size}` windows, so no resizing is applied to segmentation data
(`crop_size` must be divisible by the encoder's downsampling stride, 32 for the resnet
encoders).

### Data

The real data is a **proprietary dataset** under `data/` (gitignored, not committed):

- **Classification** — 5 classes (`endothelium`, `epithelium`, `nerves`, `stroma`, `gel`) with a
  fixed `train/`/`val/` split and no test set. Reports per-class **sensitivity, specificity, and
  AUC** on the validation set, plus a **confusion matrix** printed to the run log each validation
  epoch. **Class imbalance is handled automatically on real data**: inverse-frequency class weights
  are derived from the training set's per-class counts and applied to the cross-entropy loss (a
  no-op for the balanced synthetic default). Disable with `data.weighted_loss=false`.
- **Segmentation** — binary dendritic-cell foreground vs. background on subbasal-nerve-plexus
  slices, trainable in **three cross-validation modes** (all with the soft Dice + cross-entropy
  loss):
  - `data=segmentation_real` **(default)** — the single `data/segmentation` volume pair, split into
    **5 contiguous slice folds** (4 train / 1 val).
  - `data=segmentation2_real` — the per-patient `data/segmentation2` set (two-annotator masks),
    **5-fold cross-validation grouped by patient** (whole patients held out per fold).
  - `data=segmentation_combined_real` — both: `data/segmentation` is kept **entirely in training**
    in every fold (it has no patient ids to split on) while validation is only the held-out
    `data/segmentation2` patients, so reported metrics are patient-held-out.

  The `data/segmentation2` ground truth fuses two annotators: `1`/`0` where they agree, `0.5` where
  they disagree; a single-annotator instance (no `gt2.png`) is `0`/`1` as if the second agreed. For
  metrics the soft mask is thresholded at `>= 0.5` (union). All modes report per-fold and **mean**
  Dice/IoU, AUPRC/AUC, and sensitivity/specificity. `data.batch_size` governs **training only** for
  the patient modes; validation runs one native-resolution instance at a time.

For each task the **default data is synthetic** (auto-generated, learnable signal), so the
whole pipeline runs and is smoke-tested with **no data present**. Real data is opt-in.

```bash
# Synthetic defaults — no data required (smoke/sanity):
uv run train task=classification
uv run train task=segmentation

# Real classification (fixed train/val split; per-class metrics + confusion matrix; auto weighting):
uv run train task=classification data=classification_real

# Real segmentation 5-fold cross-validation (per-fold + mean metrics). Pick a mode with `data=`:
# Bare command = 3-epoch smoke default; for a real run see "Baseline runs" below:
uv run cross-validate                                   # (a) single volume  [default]
uv run cross-validate data=segmentation2_real           # (b) data/segmentation2, patient-grouped
uv run cross-validate data=segmentation_combined_real   # (c) both (segmentation in training only)
```

`task` has **no default** — `uv run train` errors until a task is chosen. The proprietary
data lives at `./data` by default; set `IVCM_DATA_DIR` to point elsewhere (absolute path).

The commands above inherit the trainer's default **`max_epochs: 3`** — enough to exercise the
whole pipeline (a smoke run), not to train a usable model. For real training on the proprietary
data, raise `trainer.max_epochs` (and `data.num_workers`); the **Baseline runs** section below
is the canonical way to run classification and segmentation on real data.

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
count is derived as `${len:${class_names}}`). Training augmentation is **D4** (90°
rotations + flips) via Albumentations — combined with a resize to `${image_size}` for
classification and a random `${crop_size}` crop for segmentation — plus **image-only photometric
jitter** (brightness / contrast / gamma), **on by default**. Photometric jitter is configured under
`augmentation.photometric` (global default in [configs/common.yaml](configs/common.yaml)); the
**segmentation** task overrides it to mild **brightness only** (contrast/gamma off) because those
warps hurt nerve segmentation while helping classification (see
[configs/task/segmentation.yaml](configs/task/segmentation.yaml)). Turn it off entirely with
`augmentation.photometric.enabled=false` or tune it, e.g.:

```bash
uv run train task=classification augmentation.photometric.enabled=false   # disable photometric aug
uv run train task=segmentation augmentation.photometric.brightness_limit=0.3 augmentation.photometric.p=0.7
```

Evaluation/inference transforms stay deterministic (no augmentation).

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

- **Best-checkpoint saving** is opt-in via the `callbacks` group (`callbacks=default`); the
  default `callbacks=none` adds no explicit callbacks. (Lightning still auto-injects its own
  unmonitored checkpoint unless `trainer.enable_checkpointing=false` — `callbacks=none` only
  omits *our* monitored callback.) Selecting `callbacks=default` saves the **best** checkpoint
  — by `val/auc_macro` (classification) or `val/auprc` (segmentation), both maximized — plus
  `last`, and logs the learning rate. Checkpoints land in `<run-dir>/checkpoints/` —
  `outputs/<date>/<time>/checkpoints/` for `train`, `fold_<n>/checkpoints/` for each CV fold.
  With a monitored best checkpoint present, the **reported validation metrics are computed
  from that best checkpoint** (`trainer.validate(ckpt_path="best")`); the full per-epoch curve
  stays visible in W&B, and the W&B run **summary** `val/*` is overwritten with that
  best-checkpoint snapshot (a run summary otherwise holds each metric's *last-epoch* value).
  The summary also gets a **`best_epoch`** field naming the epoch that snapshot came from — the
  summary's `epoch`/`global_step` stay at the last training step (training length), so
  `best_epoch` is what tells you which epoch the `val/*` values belong to. Without a monitored
  best checkpoint (the default), the final-epoch model is validated as before.

- **Weights-only best-checkpoint artifact (W&B).** When a best checkpoint is saved *and* a W&B
  run is active, the run also uploads a `model` artifact named `weights-<run id>` (one per run;
  each CV fold is its own run, so its own artifact). It contains just two files: a **weights-only**
  `best-weights.ckpt` (the model `state_dict` only — no optimizer state, ~⅓ the full checkpoint's
  size — with all tensors on **CPU**) and the run's `.hydra/config.yaml`. This makes the artifact
  **self-contained and cross-platform**: trained on a CUDA machine, it loads directly on a Mac
  (MPS/CPU). The upload is best-effort — a failure only warns and never fails the run — and the
  local `best-weights.ckpt` is also left in the checkpoint dir. See
  [Using a trained model in the UI](#using-a-trained-model-in-the-ui) for how the app consumes it.

### Baseline runs

The current "base setup" (no architecture/loss tuning) run for real on the proprietary data,
with best-checkpoint saving and live W&B (requires `wandb login` / `WANDB_API_KEY`):

```bash
# Classification baseline (real data, ~50 epochs, best-AUC checkpoint; class weighting is automatic)
uv run train task=classification data=classification_real \
  callbacks=default logger.mode=online \
  trainer.max_epochs=50 data.num_workers=4

# Segmentation baseline — 5-fold cross-validation (best-AUPRC checkpoint per fold). Add
# `data=segmentation2_real` or `data=segmentation_combined_real` to select mode (b) or (c).
uv run cross-validate \
  callbacks=default logger.mode=online \
  trainer.max_epochs=50 data.num_workers=4
```

**Resuming an interrupted `train` run.** With `callbacks=default` (which writes `last.ckpt`),
point `ckpt_path` at a saved checkpoint to continue; Lightning restores the epoch, optimizer,
and scheduler state and trains on to `max_epochs`:

```bash
uv run train task=classification data=classification_real callbacks=default \
  trainer.max_epochs=50 \
  ckpt_path=outputs/<date>/<time>/checkpoints/last.ckpt
```

`ckpt_path` defaults to `null` (fresh run). When set, new checkpoints continue in the
**original** checkpoint directory (the resumed checkpoint's own folder), so best-checkpoint
tracking carries across the whole training history even though the resumed invocation gets a
new `outputs/<date>/<time>/` for its logs.

The resumed run also **continues the original W&B run** rather than starting a new one (the
same continuation that `cross-validate` folds get, below). A run's W&B identity is tied to its
checkpoint directory: a fresh `callbacks=default` run is tagged with a stable id under
`<run_dir>/checkpoints/`, and resuming from a checkpoint in that directory reconnects to that
run. This relies on a stable pinned checkpoint dir, so it covers `callbacks=default` runs (and
any resume from such a checkpoint). A fresh `callbacks=none` run is **not** tagged (Lightning's
auto-injected checkpoint has no stable location), so the first resume from its checkpoint
starts a new W&B run and tags that directory — subsequent resumes from the same directory then
continue that run.

**Resuming an interrupted `cross-validate` run.** CV resumes **per fold**: re-run pointing
Hydra at the interrupted run's output directory (a global `ckpt_path` is rejected — one
checkpoint can't map to five folds):

```bash
uv run cross-validate callbacks=default trainer.max_epochs=50 \
  hydra.run.dir=outputs/<date>/<time>
```

Each fold then **skips** if it already saved `metrics.json`, **resumes** from its `last.ckpt`
if it was interrupted mid-training, or **runs fresh** otherwise. Requires `callbacks=default`
(for `last.ckpt`). A resumed fold **continues its original W&B run** and **keeps tracking the
best checkpoint across the whole fold history** (the stable `fold_<n>/checkpoints/` lets
Lightning restore the checkpoint state on resume).

### Predicting / visual comparison

After a finished cross-validation run, build a single out-of-fold prediction volume for
slice-by-slice visual comparison against the ground truth (e.g. in ITK-SNAP):

```bash
uv run predict-cv outputs/<date>/<time>
```

This takes the **finished CV output directory** as its only argument and writes
`<cv_dir>/predictions.nii`: a combined binary nerve mask (`0`/`1`, `uint8`) at the GT
resolution and affine, where **each slice is predicted by the fold's best model that did not
train on it** (the validation blocks partition every slice, so the volume is directly
overlayable on `mask_gt.nii`). Each slice is predicted at its native resolution via
sliding-window inference, so no resize is involved. It requires the CV to have been run with
`callbacks=default` (so a monitored best checkpoint exists per fold).

`predict-cv` supports **only the single-volume CV (mode (a), `data=segmentation_real`)** — the
patient-grouped `segmentation2` / `combined` modes have no single out-of-fold volume to assemble, so
it rejects those runs with a clear error.

Unlike `train`/`cross-validate`, `predict-cv` is **not** a Hydra command: it reads the run's
own saved `.hydra/config.yaml` (so the model, paths, `n_folds`, and `crop_size` match what was
trained) and writes nothing into the run dir except `predictions.nii`. Two data-location
cases:

- **Default (relative) `data_dir`** — the volumes are recovered automatically from the run's
  saved working directory, so the command works from any cwd with no extra setup.
- **`IVCM_DATA_DIR` was set** — it re-resolves against the current environment, so run
  `predict-cv` with the same value the CV used (an absolute path is reused as-is; a relative
  one is still resolved against the saved cwd). The resolved image/mask paths are logged on
  every run so a mismatch is obvious.

## Configuration layout

```
configs/
  common.yaml             # shared root: seed, image_size, crop_size, data_dir, trainer/logger defaults
  train.yaml              # root for `train`: includes common, mandatory `task: ???`
  cv.yaml                 # root for `cross-validate`: segmentation + real CV data
  task/                   # classification | segmentation (class_names + composes groups)
  data/                   # <task> = synthetic (default); classification_real; segmentation_real (a),
                          #   segmentation2_real (b), segmentation_combined_real (c)
  model/ loss/            # per-task components (classification: cross_entropy; segmentation: dice_ce)
  optimizer/ scheduler/   # shared components
  trainer/ logger/        # Lightning Trainer + W&B logger (logger=none to disable)
  callbacks/              # none (default) | default = best-metric ModelCheckpoint + LR monitor
```

**Compatibility note:** the segmentation loss is now the soft **Dice + cross-entropy** combo
(`loss=dice_ce`); the old `loss=dice` (SMP multiclass Dice) config is **removed** — it consumed
integer targets, incompatible with the new soft `(logits, p)` criterion contract. An old
`loss=dice` override now errors rather than silently using the retired loss.

## Project layout

```
src/ivcm/
  data/        # real (.tif/.nii) + synthetic datasets, datamodules, D4 + photometric + resize/crop transforms
  models/      # classification + segmentation LightningModules (incl. metrics)
  utils/       # logging setup, device selection (shared by predict + UI)
  app/         # PySide6 clinical-review UI (`ivcm-ui` console script; optional `ui` extra)
  train.py     # Hydra entry point (`train` console script)
  cv.py        # segmentation cross-validation (`cross-validate` console script)
  predict.py   # out-of-fold prediction volume from a finished CV run (`predict-cv` script)
  morphometry.py  # dendritic-cell density/area/perimeter from a 2D mask (smoothed-contour)
tests/         # fold + config + metric-key + smoke + predict tests; real-data tests (skip if no data)
configs/       # Hydra configuration tree
```
