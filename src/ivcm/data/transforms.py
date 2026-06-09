"""Albumentations transform builders.

Every input is resized to a fixed ``size`` (a single ``${image_size}`` config value,
default 384) so the whole pipeline runs at one resolution. Beyond the resize we only use
D4 augmentations (the dihedral group of the square: 90-degree rotations and h/v flips):
a deliberately small, label-preserving set appropriate for both classification and
segmentation, easily extended later.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


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
