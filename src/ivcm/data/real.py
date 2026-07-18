"""Datasets for the real (proprietary) data.

Classification reads single-channel ``.tif`` images from per-class folders; segmentation
reads single 2D slices out of pre-loaded ``.nii`` volumes. Both yield the same tensor
shapes as the synthetic datasets so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ivcm.data.transforms import Transform


class NonSquareImageError(ValueError):
    """A pipeline input image is not square (pixel height != width).

    A specific type so callers (the UI directory pre-scan in particular) can distinguish a
    non-square input — a fatal, guaranteed-invariant violation — from ordinary load/decode
    failures, which are handled separately.
    """


def ensure_square(shape: Sequence[int], *, source: str) -> None:
    """Raise :class:`NonSquareImageError` unless ``shape``'s first two dims (H, W) are equal.

    Takes a shape-like (an ``(H, W[, ...])`` tuple), not an array, so the UI pre-scan's
    header-only dimension read stays cheap for large images. Array callers pass ``image.shape``.
    """
    height, width = int(shape[0]), int(shape[1])
    if height != width:
        raise NonSquareImageError(
            f"Image {source} is not square: {height}×{width}. All inputs must be square (height == width)."
        )


def contiguous_folds(n_slices: int, n_folds: int) -> list[np.ndarray]:
    """Partition ``range(n_slices)`` into ``n_folds`` contiguous index blocks.

    Uses ``np.array_split`` so the split is deterministic and contiguous: the first
    ``n_slices % n_folds`` folds get one extra index. Adjacent slices are correlated, so
    contiguous blocks (rather than interleaving) keep a fold's val slices away from its
    train slices.
    """
    return np.array_split(np.arange(n_slices), n_folds)


class ClassificationImageDataset(Dataset[tuple[Any, int]]):
    """Single-channel ``.tif`` images under ``root/<class_name>/``.

    Class -> index order is fixed by ``class_names`` (the single source of truth, supplied
    from config), so per-class metrics line up across runs.
    """

    def __init__(self, root: str | Path, class_names: Sequence[str], transform: Transform | None = None) -> None:
        self.root = Path(root)
        self.class_names = list(class_names)
        self.transform = transform

        found = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if found != sorted(self.class_names):
            raise ValueError(f"Class folders under {self.root} {found} do not match class_names {self.class_names}")

        self.samples: list[tuple[Path, int]] = []
        for label, name in enumerate(self.class_names):
            for path in sorted((self.root / name).glob("*.tif")):
                self.samples.append((path, label))
        if not self.samples:
            raise ValueError(f"No .tif images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        ensure_square(image.shape, source=path.name)
        image = image[..., None]  # HWC, single channel
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class NiiSegmentationDataset(Dataset[tuple[Any, Any]]):
    """2D slices taken from pre-loaded ``.nii`` volumes.

    ``images``/``masks`` are full volumes of shape ``(H, W, n_slices)`` (loaded once by the
    datamodule and shared between the train/val datasets). ``slice_indices`` selects which
    slices along the last axis belong to this split. The mask is binary (background vs.
    nerve).
    """

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        slice_indices: Sequence[int] | np.ndarray,
        transform: Transform | None = None,
    ) -> None:
        self.images = images
        self.masks = masks
        self.slice_indices = list(slice_indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slice_indices)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        idx = self.slice_indices[index]
        image = self.images[:, :, idx].astype(np.float32) / 255.0
        ensure_square(image.shape, source=f"slice {idx}")
        image = image[..., None]  # HWC, single channel
        mask = self.masks[:, :, idx].astype(np.int64)
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask
