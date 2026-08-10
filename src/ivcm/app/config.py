"""Hardcoded application configuration for the clinical-review UI.

A single ``AppConfig`` holds the two model ``(run_dir, ckpt)`` pairs, the morphometry
parameters, the overlay appearance, and the accepted image extensions. The doctor never edits
these; they only choose an image directory.

The checkpoint paths are **local placeholders** (resolved relative to the repo root, derived
from this file's location — not the process cwd). Checkpoints and ``outputs/`` are gitignored
local artifacts, so the app is not runnable on a clean checkout: edit ``CONFIG`` below to point
at this machine's trained runs. A missing path produces a clear startup error (see
``ivcm.app.inference``).

To consume a model trained on another machine, download that run's weights-only W&B artifact
(``weights-<run id>``, containing ``best-weights.ckpt`` + ``.hydra/config.yaml``) and set
``run_dir`` to the downloaded directory and ``ckpt`` to ``<dir>/best-weights.ckpt`` — using
**absolute (or full repo-relative) paths for both**, since a relative path resolves against the
repo root, not against ``run_dir``. The weights are CPU-portable (trained on CUDA, load on
MPS/CPU). See the README's "Using a trained model in the UI".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

# Repo root: ``.../ivcm_vibe`` from ``.../ivcm_vibe/src/ivcm/app/config.py`` (parents[3]).
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str) -> Path:
    """Resolve a configured path against the repo root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class ModelConfig:
    """One model: the Hydra run root (holding ``.hydra/config.yaml``) and a checkpoint under it.

    ``run_dir`` is the run root; ``ckpt`` may live anywhere inside the run (``checkpoints/`` for
    a ``train`` run, ``fold_<n>/checkpoints/`` for a CV run) and is given as a full path.
    """

    run_dir: str
    ckpt: str

    def run_path(self) -> Path:
        return _resolve(self.run_dir)

    def ckpt_path(self) -> Path:
        return _resolve(self.ckpt)

    def config_path(self) -> Path:
        return self.run_path() / ".hydra" / "config.yaml"


@dataclass(frozen=True)
class AppConfig:
    """Everything the UI needs that is not the chosen directory."""

    classification: ModelConfig
    segmentation: ModelConfig

    # Morphometry parameters forwarded to ``ivcm.morphometry.cell_morphometry``.
    # ``physical_image_size`` is the physical side length of the (square) input image in µm
    # (default 400 -- the HRT / Rostock Cornea Module field): density is reported as cells/mm²
    # (``count / (physical_image_size / 1000)**2``), area in µm² and perimeter in µm (scaling the
    # raw pixel measurements by ``physical_image_size / resolution``, squared for area). It is
    # unrelated to the pixel-space ``image_size`` in the training/model config.
    sigma: float = 1.0
    physical_image_size: float = 400.0
    min_size: int = 3

    # Overlay appearance: RGB mask colour and blend alpha in [0, 1].
    mask_color: tuple[int, int, int] = (255, 0, 0)
    mask_alpha: float = 0.45

    accepted_extensions: tuple[str, ...] = field(default_factory=lambda: (".tif", ".tiff", ".png", ".jpg", ".jpeg"))

    # Directory-batch throughput: ``batch_size`` is both the inference sub-batch size and the
    # file-chunk size of the pipeline; ``num_workers`` is the thread count for parallel decode /
    # preprocess / morphometry. Both must be >= 1 (a non-positive batch breaks chunking/forward
    # splitting and ThreadPoolExecutor rejects ``max_workers <= 0``).
    batch_size: int = 16
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}.")
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1; got {self.num_workers}.")
        if self.physical_image_size <= 0:
            raise ValueError(f"physical_image_size must be > 0; got {self.physical_image_size}.")


# Edit these to this machine's trained runs. Both ``run_dir`` values must contain
# ``.hydra/config.yaml``; both ``ckpt`` values must point at a real checkpoint under the run.
_LOCAL_CONFIG = AppConfig(
    classification=ModelConfig(
        run_dir="/Users/dmitry/ivcm-models/cls",
        ckpt="/Users/dmitry/ivcm-models/cls/best-weights.ckpt",
    ),
    segmentation=ModelConfig(
        run_dir="/Users/dmitry/ivcm-models/seg",
        ckpt="/Users/dmitry/ivcm-models/seg/best-weights.ckpt",
    ),
)


def _frozen_model_config(base: Path, name: str) -> ModelConfig:
    """A ``ModelConfig`` pointing at the bundled ``models/<name>`` directory under ``base``.

    Matches the layout the PyInstaller spec bundles: ``models/<name>/best-weights.ckpt`` and
    ``models/<name>/.hydra/config.yaml`` (the latter via ``ModelConfig.config_path()``).
    """
    run_dir = base / "models" / name
    return ModelConfig(run_dir=str(run_dir), ckpt=str(run_dir / "best-weights.ckpt"))


def build_config() -> AppConfig:
    """Build the application configuration for the current runtime.

    In a normal checkout this is the machine-specific ``_LOCAL_CONFIG`` above (edited to point at
    this machine's trained runs). In a PyInstaller one-folder bundle (``sys.frozen``), the two model
    ``(run_dir, ckpt)`` pairs are redirected to the ``models/cls`` and ``models/seg`` directories
    bundled under ``sys._MEIPASS``; every other field is carried over unchanged via
    :func:`dataclasses.replace`, so operational settings can never be dropped by the redirect.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", ""))  # PyInstaller sets _MEIPASS to the bundle root
        return replace(
            _LOCAL_CONFIG,
            classification=_frozen_model_config(base, "cls"),
            segmentation=_frozen_model_config(base, "seg"),
        )
    return _LOCAL_CONFIG


CONFIG = build_config()
