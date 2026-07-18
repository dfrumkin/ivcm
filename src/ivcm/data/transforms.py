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
"""

from __future__ import annotations

from collections.abc import Callable
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


def build_d4_transform(size: int = 384) -> A.Compose:
    """Training transform: resize to ``size``, random D4 symmetry, then CHW tensor.

    Applied jointly to image and (when present) mask, so the resize and geometric ops
    stay aligned. The mask is resized with nearest-neighbour interpolation (Albumentations
    applies nearest to masks automatically), the image bilinear.
    """
    return A.Compose(
        [
            A.Resize(size, size),
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )


def build_eval_transform(size: int = 384) -> A.Compose:
    """Evaluation transform: resize to ``size`` then convert to a CHW tensor."""
    return A.Compose([A.Resize(size, size), ToTensorV2()])


def build_crop_transform(size: int = 192) -> A.Compose:
    """Segmentation training transform: random ``size`` crop, random D4 symmetry, CHW tensor.

    Applied jointly to image and mask so the crop and geometric ops stay aligned. No resize —
    the crop is taken at the slice's native resolution. ``pad_if_needed`` pads (reflect) any
    image/mask smaller than ``size`` up to ``size`` so the crop is always valid. The parameter
    is named ``size`` to match the other builders and the ``size: ...`` config convention.
    """
    return A.Compose(
        [
            A.RandomCrop(size, size, pad_if_needed=True),
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )


def build_seg_eval_transform() -> A.Compose:
    """Segmentation evaluation transform: convert to a CHW tensor at native resolution.

    No resize and no crop — the full-resolution image (and mask, when present) is fed straight
    to sliding-window inference, which tiles it into ``roi_size`` windows.
    """
    return A.Compose([ToTensorV2()])
