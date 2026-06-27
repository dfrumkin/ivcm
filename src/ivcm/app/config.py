"""Hardcoded application configuration for the clinical-review UI.

A single ``AppConfig`` holds the two model ``(run_dir, ckpt)`` pairs, the morphometry
constants, the overlay appearance, and the accepted image extensions. The doctor never edits
these; they only choose an image directory.

The checkpoint paths are **local placeholders** (resolved relative to the repo root, derived
from this file's location — not the process cwd). Checkpoints and ``outputs/`` are gitignored
local artifacts, so the app is not runnable on a clean checkout: edit ``CONFIG`` below to point
at this machine's trained runs. A missing path produces a clear startup error (see
``ivcm.app.inference``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    # Morphometry constants forwarded to ``ivcm.morphometry.cell_morphometry``.
    sigma: float = 1.0
    constant: float = 1.0
    min_size: int = 0

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


# Edit these to this machine's trained runs. Both ``run_dir`` values must contain
# ``.hydra/config.yaml``; both ``ckpt`` values must point at a real checkpoint under the run.
CONFIG = AppConfig(
    classification=ModelConfig(
        run_dir="outputs/2026-06-10/12-09-43",
        ckpt="outputs/2026-06-10/12-09-43/checkpoints/best-epoch38.ckpt",
    ),
    segmentation=ModelConfig(
        run_dir="outputs/2026-06-10/15-42-08",
        ckpt="outputs/2026-06-10/15-42-08/fold_1/checkpoints/best-epoch38.ckpt",
    ),
)
