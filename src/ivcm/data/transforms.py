"""Albumentations transform builders.

For now we only use D4 augmentations (the dihedral group of the square: 90-degree
rotations and horizontal/vertical flips). This is a deliberately small, label-preserving
set appropriate for both classification and segmentation; it can be extended later.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_d4_transform() -> A.Compose:
    """Training transform: random D4 symmetry, then convert to a CHW float tensor.

    Applied jointly to image and (when present) mask, so geometric ops stay aligned.
    """
    return A.Compose(
        [
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )


def build_eval_transform() -> A.Compose:
    """Evaluation transform: no augmentation, just tensor conversion."""
    return A.Compose([ToTensorV2()])
