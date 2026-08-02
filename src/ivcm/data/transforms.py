"""Albumentations transform builders.

Two families share the same D4 augmentation core (the dihedral group of the square:
90-degree rotations and h/v flips — a deliberately small, label-preserving set):

- **Classification** resizes every input to a fixed ``size`` (``${image_size}``, default 384)
  so it runs at one resolution: :func:`build_d4_transform` (train) and
  :func:`build_eval_transform` (eval).
- **Segmentation** trains on native-resolution random crops of ``size`` (``${crop_size}``,
  default 192) via :func:`build_crop_transform`, and evaluates on the full native-resolution
  image (no resize, no crop) via :func:`build_seg_eval_transform`, feeding it to sliding-window
  inference.

The two **training** builders additionally apply image-only **photometric** jitter (brightness /
contrast / gamma), on by default and driven by the ``augmentation.photometric`` config node — see
:func:`_photometric_transforms`. The eval builders stay purely deterministic (no photometric).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

# albumentations ships no type stubs (see the Transform note below), so these imports are
# flagged reportMissingTypeStubs; suppress at the boundary.
import albumentations as A  # pyright: ignore[reportMissingTypeStubs]
from albumentations.pytorch import ToTensorV2  # pyright: ignore[reportMissingTypeStubs]

# Albumentations transforms ship no type stubs; model them at our boundary as a callable
# returning the augmented ``{"image": ..., "mask": ...}`` dict. Datasets annotate their
# ``transform`` with this so downstream values are explicitly ``Any`` (a known type) rather
# than ``Unknown``, stopping the unknown-type cascade at the boundary.
Transform = Callable[..., dict[str, Any]]

# Default photometric augmentation (used when a builder is called with ``photometric=None``, i.e.
# bare code callers / tests): on, with moderate ranges suited to uncalibrated IVCM grayscale.
# Configured runs pass the resolved ``augmentation.photometric`` node instead. Kept as a module
# constant (not a mutable default argument) so callers never share a mutable default object.
_DEFAULT_PHOTOMETRIC: dict[str, Any] = {
    "enabled": True,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "gamma_limit": (80, 120),
    "p": 0.5,
}


def _photometric_transforms(photometric: Mapping[str, Any] | None) -> list[Any]:
    """Freshly-constructed image-only photometric transforms for the training pipelines.

    ``photometric`` is the ``augmentation.photometric`` config node (an OmegaConf mapping or plain
    dict) with keys ``enabled``/``brightness_limit``/``contrast_limit``/``gamma_limit``/``p``, or
    ``None`` to use :data:`_DEFAULT_PHOTOMETRIC`. Returns fresh transform instances on every call
    (Albumentations transforms carry per-instance state, so they must not be shared across
    ``Compose``s), or an empty list when disabled.

    ``enabled`` is checked first and is authoritative: a false value short-circuits before any range
    field is read, so the master switch disables entirely even if the range fields are absent or
    malformed. Applies to the image only — Albumentations never routes photometric ops to ``mask``.
    """
    cfg: Mapping[str, Any] = _DEFAULT_PHOTOMETRIC if photometric is None else photometric
    if not cfg.get("enabled", True):
        return []
    p = cfg.get("p", _DEFAULT_PHOTOMETRIC["p"])
    return [
        A.RandomBrightnessContrast(
            brightness_limit=cfg.get("brightness_limit", _DEFAULT_PHOTOMETRIC["brightness_limit"]),
            contrast_limit=cfg.get("contrast_limit", _DEFAULT_PHOTOMETRIC["contrast_limit"]),
            p=p,
        ),
        A.RandomGamma(gamma_limit=tuple(cfg.get("gamma_limit", _DEFAULT_PHOTOMETRIC["gamma_limit"])), p=p),
    ]


def build_d4_transform(size: int = 384, photometric: Mapping[str, Any] | None = None) -> A.Compose:
    """Training transform: resize to ``size``, random D4 symmetry, photometric jitter, CHW tensor.

    Applied jointly to image and (when present) mask, so the resize and geometric ops
    stay aligned. The mask is resized with nearest-neighbour interpolation (Albumentations
    applies nearest to masks automatically), the image bilinear. ``photometric`` (see
    :func:`_photometric_transforms`) adds image-only brightness/contrast/gamma jitter after the
    geometric ops; ``None`` uses the on-by-default settings, and an ``enabled=false`` config removes
    it entirely (leaving the classic D4-only pipeline).
    """
    return A.Compose(
        [
            A.Resize(size, size),
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            *_photometric_transforms(photometric),
            ToTensorV2(),
        ]
    )


def build_eval_transform(size: int = 384) -> A.Compose:
    """Evaluation transform: resize to ``size`` then convert to a CHW tensor."""
    return A.Compose([A.Resize(size, size), ToTensorV2()])


def build_crop_transform(size: int = 192, photometric: Mapping[str, Any] | None = None) -> A.Compose:
    """Segmentation training transform: random ``size`` crop, D4 symmetry, photometric jitter, tensor.

    Applied jointly to image and mask so the crop and geometric ops stay aligned. No resize —
    the crop is taken at the slice's native resolution. ``pad_if_needed`` pads (reflect) any
    image/mask smaller than ``size`` up to ``size`` so the crop is always valid. The parameter
    is named ``size`` to match the other builders and the ``size: ...`` config convention.
    ``photometric`` (see :func:`_photometric_transforms`) adds image-only brightness/contrast/gamma
    jitter after the geometric ops — image only, so soft masks are unaffected; ``None`` uses the
    on-by-default settings and an ``enabled=false`` config removes it entirely.
    """
    return A.Compose(
        [
            A.RandomCrop(size, size, pad_if_needed=True),
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            *_photometric_transforms(photometric),
            ToTensorV2(),
        ]
    )


def build_seg_eval_transform() -> A.Compose:
    """Segmentation evaluation transform: convert to a CHW tensor at native resolution.

    No resize and no crop — the full-resolution image (and mask, when present) is fed straight
    to sliding-window inference, which tiles it into ``roi_size`` windows.
    """
    return A.Compose([ToTensorV2()])
