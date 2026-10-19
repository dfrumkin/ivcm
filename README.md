# ivcm

Medical image classification and segmentation, with a desktop review UI for clinicians.

## Setup

Requires [uv](https://docs.astral.sh/uv/), which also provisions the pinned Python version.

```bash
uv sync                     # create .venv and install dependencies
uv run pre-commit install   # enable git pre-commit hooks
```

The desktop UI needs the optional `ui` extra (PySide6/Qt):

```bash
uv sync --extra ui
```

## Development

```bash
uv run pytest               # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
uvx pyright                 # static type check (strict; see docs/coding-standards.md)
uv run pre-commit run --all-files
```

Pyright scans the whole `src` tree, including the UI, so sync the `ui` extra before
type-checking or linting. `tests/test_app.py` only imports Qt-free modules and passes without it.

## Project layout

```
src/ivcm/
  data/        # real (.tif/.nii) + synthetic datasets, datamodules, transforms
  models/      # classification + segmentation LightningModules
  utils/       # logging, device selection
  app/         # PySide6 clinical-review UI (`ivcm-ui`, optional `ui` extra)
  train.py     # Hydra entry point (`train`)
  cv.py        # segmentation cross-validation (`cross-validate`)
  predict.py   # out-of-fold prediction volume from a finished CV run (`predict-cv`)
  morphometry.py  # cell count/density/area/perimeter from a 2D mask
configs/       # Hydra configuration tree (see below)
tests/         # unit, config, and smoke tests; real-data tests skip if no data
```

## Clinical review UI

A PySide6 app for reviewing a directory of images: it classifies each image, segments the ones
predicted to be nerves, and shows their morphometry (count, density, area, perimeter) in a
table. Reviewers can accept/reject each image and hand-edit masks. A review can be saved to and
reloaded from a label directory.

Up to three images are pinned; the pinned images are the ones shown, ordered by cell count, in the
"Sample Images" window (opened with the "Sample" toolbar button). A fresh review pins
the images whose counts are closest to the mean − SD, the mean, and the mean + SD. Unpinning an
image (or marking it "Don't Use") in that window replaces it with the nerve image closest to that tile's target that has not been pinned yet in
this session. Unpinning in the main table leaves the tile empty, so another image can be pinned
by hand.

```bash
uv run ivcm-ui
```

The app ships with no trained models. Point it at yours by editing the two checkpoint paths in
[`src/ivcm/app/config.py`](src/ivcm/app/config.py):

```python
CONFIG = AppConfig(
    classification=ModelConfig(run_dir="outputs/<run>", ckpt="outputs/<run>/checkpoints/<ckpt>.ckpt"),
    segmentation=ModelConfig(run_dir="outputs/<run>", ckpt="outputs/<run>/fold_0/checkpoints/<ckpt>.ckpt"),
)
```

Each `run_dir` needs a `.hydra/config.yaml` (to rebuild the model) and `ckpt` a matching
checkpoint. Training a model locally produces both (see [Training](#training) below); to use a
model trained elsewhere, download its `weights-<run id>` W&B artifact and point `config.py` at
that directory instead — it's self-contained (CPU weights + config) and works across platforms.

## Training

Built on **PyTorch Lightning** + **Hydra**. `task=classification` or `task=segmentation` selects
the pipeline; there's no default, so it must be chosen explicitly. Real data is a proprietary
dataset under `data/` (gitignored); every task also has a synthetic default so the pipeline runs
with no data present.

```bash
# Synthetic smoke runs, no data required:
uv run train task=classification
uv run train task=segmentation

# Real data:
uv run train task=classification data=classification_real
uv run cross-validate                                   # 5-fold CV, single volume
uv run cross-validate data=segmentation2_real           # patient-grouped CV
uv run cross-validate data=segmentation_combined_real   # both, patient-held-out validation
```

Anything in the config tree can be overridden on the command line:

```bash
uv run train task=classification trainer.max_epochs=50 data.num_workers=4 logger.mode=online
```

- Each run writes to `outputs/<date>/<time>/`, logged with **Rich** and tracked with **W&B**
  (offline by default; `logger.mode=online` syncs live, `logger=none` disables it).
- `callbacks=default` saves the best checkpoint (by validation AUC/AUPRC) plus `last.ckpt`, and
  reports metrics from the best epoch. It also uploads a small weights-only W&B artifact per run,
  used by the UI (see above).
- Resume a run with `ckpt_path=<checkpoint>` (`train`) or by pointing `cross-validate` at the
  original `hydra.run.dir` — each fold resumes, skips, or restarts as needed.
- After a finished single-volume CV run, `uv run predict-cv <cv_dir>` builds a combined
  out-of-fold prediction volume for visual comparison against ground truth.

### Configuration layout

```
configs/
  common.yaml   # shared root: seed, image size, data dir, trainer/logger defaults
  train.yaml    # root for `train`
  cv.yaml       # root for `cross-validate`
  task/         # classification | segmentation
  data/         # synthetic (default), classification_real, segmentation*_real
  model/ loss/  # per-task components
  optimizer/ scheduler/ trainer/ logger/ callbacks/
```

## Packaging a Windows build

The UI can be frozen into a self-contained Windows folder (PyInstaller, no Python required) with
trained models bundled in, built on GitHub Actions and published as a release asset.

```bash
uv run python scripts/publish_models.py   # publish the two checkpoints as a release asset (once per model revision)
git tag v1.0.0                            # tagging and pushing triggers the build...
git push origin v1.0.0                    # ...which attaches the zip to the release
```

The customer unzips the release asset and runs `ivcm-ui.exe`; on first run they'll need to click
through a Windows SmartScreen warning (the app isn't code-signed). See
[`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml) for build details.
