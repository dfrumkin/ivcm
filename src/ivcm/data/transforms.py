"""Albumentations transform builders.

Every input is resized to a fixed ``size`` (a single ``${image_size}`` config value,
default 384) so the whole pipeline runs at one resolution. Beyond the resize we only use
D4 augmentations (the dihedral group of the square: 90-degree rotations and h/v flips):
a deliberately small, label-preserving set appropriate for both classification and
segmentation, easily extended later.
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
