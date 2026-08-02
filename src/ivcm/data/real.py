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
import torch
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


def inverse_frequency_weights(counts: Sequence[int]) -> torch.Tensor:
    """Per-class cross-entropy weights = normalized inverse frequency.

    ``w_c = total / (num_classes * count_c)``. The *sample-frequency-weighted* mean of the weights is
    exactly 1 (``Σ count_c · w_c / total = 1``), so the overall loss scale stays comparable to the
    unweighted case (the plain arithmetic mean of the weights is not necessarily 1). Raises a clear
    error if any class has zero samples (its weight would divide by zero).
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    if counts_arr.ndim != 1 or counts_arr.size == 0:
        raise ValueError(f"counts must be a non-empty 1-D sequence of ints; got shape {counts_arr.shape}.")
    if (counts_arr <= 0).any():
        raise ValueError(f"every class must have at least one sample; got counts {list(counts)}.")
    num_classes = counts_arr.size
    weights = counts_arr.sum() / (num_classes * counts_arr)
    return torch.tensor(weights, dtype=torch.float32)


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

    def class_counts(self, num_classes: int) -> list[int]:
        """Per-class sample counts by label index, read from ``samples`` (no image decode).

        Used by the datamodule to derive inverse-frequency loss weights. ``num_classes`` fixes the
        output length so a class present in ``class_names`` but empty on disk reads as ``0``.
        """
        counts = [0] * num_classes
        for _path, label in self.samples:
            counts[label] += 1
        return counts

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
    slices along the last axis belong to this split. The mask is returned as a **float32 foreground
    probability** in ``{0.0, 1.0}`` (binarised background vs. nerve) so it stacks with the soft
    ``data/segmentation2`` masks in a combined ``ConcatDataset`` and feeds the soft-target seg loss.
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
        # /255 bakes in the pipeline's 8-bit source assumption; clip guarantees the [0, 1] range the
        # (image-only) photometric train augmentation relies on. A no-op for valid 8-bit volumes.
        image = np.clip(self.images[:, :, idx].astype(np.float32) / 255.0, 0.0, 1.0)
        ensure_square(image.shape, source=f"slice {idx}")
        image = image[..., None]  # HWC, single channel
        mask = (self.masks[:, :, idx] > 0).astype(np.float32)  # float32 foreground prob {0.0, 1.0}
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask


def discover_segmentation2(root: str | Path) -> list[tuple[Path, str]]:
    """Enumerate ``data/segmentation2`` instances as sorted ``(instance_dir, patient)`` pairs.

    The tree is exactly ``<root>/<patient>/<instance>/``; ``patient`` is the instance dir's parent
    folder name. Fails loudly (raises) on an instance directory missing ``image.png`` or ``gt.png``
    (a missing label must surface, not silently shrink the dataset) and on an empty tree. ``gt2.png``
    is optional (single-annotator instances).
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"segmentation2 root {root} is not a directory.")
    pairs: list[tuple[Path, str]] = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for instance_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            if not (instance_dir / "image.png").is_file() or not (instance_dir / "gt.png").is_file():
                raise ValueError(
                    f"instance {instance_dir} is missing image.png or gt.png "
                    "(every data/segmentation2 instance needs both)."
                )
            pairs.append((instance_dir, patient_dir.name))
    if not pairs:
        raise ValueError(f"no instances found under {root} (expected <root>/<patient>/<instance>/).")
    return pairs


def patient_folds(patients: Sequence[str], n_folds: int) -> list[np.ndarray]:
    """Partition the *unique* patients into ``n_folds`` deterministic blocks (sorted, then split).

    Mirrors :func:`contiguous_folds` but over patient ids: one block is a fold's validation patients,
    the rest are training, so a whole patient stays on one side of the split (no per-frame leakage).
    """
    unique = np.array(sorted(set(patients)))
    return np.array_split(unique, n_folds)


class Segmentation2Dataset(Dataset[tuple[Any, Any]]):
    """Per-instance two-annotator segmentation samples from ``data/segmentation2``.

    Built from a list of instance directories (not the root), each holding ``image.png`` + ``gt.png``
    (reader 1) and optionally ``gt2.png`` (reader 2). The target is a float32 foreground probability
    ``p``: ``(r1 + r2) / 2`` when both readers are present (values ``{0, 0.5, 1}`` — ``0.5`` where they
    disagree) or ``r1`` alone (``{0, 1}`` — a single annotator is treated as full agreement).
    """

    def __init__(self, instance_dirs: Sequence[Path], transform: Transform | None = None) -> None:
        self.instance_dirs = list(instance_dirs)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.instance_dirs)

    @staticmethod
    def _binary_mask(path: Path) -> np.ndarray:
        """Load a mask PNG as a 2-D ``{0.0, 1.0}`` float32 array.

        ``convert("L")`` first so a palette/RGB PNG collapses to a single 2-D channel (keeping the
        shape check meaningful); cast to float32 *before* averaging so ``r1 + r2`` counts annotators
        (boolean ``+`` would OR, not add).
        """
        return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.float32)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        instance = self.instance_dirs[index]
        image = np.asarray(Image.open(instance / "image.png").convert("L"), dtype=np.float32) / 255.0
        ensure_square(image.shape, source=str(instance / "image.png"))
        r1 = self._binary_mask(instance / "gt.png")
        gt2_path = instance / "gt2.png"
        r2 = self._binary_mask(gt2_path) if gt2_path.is_file() else None

        shapes: list[tuple[int, ...]] = [image.shape, r1.shape]
        if r2 is not None:
            shapes.append(r2.shape)
        if len(set(shapes)) != 1:
            raise ValueError(
                f"image/gt/gt2 sizes disagree in {instance}: image {image.shape}, gt {r1.shape}"
                + (f", gt2 {r2.shape}" if r2 is not None else "")
            )

        mask = (r1 + r2) / 2.0 if r2 is not None else r1  # {0, 0.5, 1} or {0, 1}
        image = image[..., None]  # HWC, single channel
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask
