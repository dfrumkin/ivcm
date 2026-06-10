"""Lightning data modules.

``BaseDataModule`` is the generic case: it holds two already-instantiated datasets (built
by Hydra from config) and owns only the DataLoader plumbing. The real/synthetic
distinction therefore lives entirely in the dataset configs, not in datamodule classes.

``SegmentationCVDataModule`` is the one exception: 5-fold cross-validation couples the
train/val split (one volume load, one fold index decides which slices are validation), so
it builds its datasets internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import lightning as L
import nibabel as nib
import numpy as np
from nibabel.dataobj_images import DataobjImage
from torch.utils.data import DataLoader, Dataset

from ivcm.data.real import NiiSegmentationDataset, contiguous_folds
from ivcm.data.transforms import build_d4_transform, build_eval_transform

# A dataset sample is an (image, target) pair; the image is `Any` because it crosses the
# untyped albumentations boundary, and the target is an int (classification) or array/tensor
# mask (segmentation). DataLoaders are parameterised by this so their item type is known.
Sample = tuple[Any, Any]


def _load_volume(path: Path) -> np.ndarray:
    """Load a ``.nii`` volume as a numpy array.

    ``nib.load`` is typed to return the base ``FileBasedImage``, which exposes no array
    access; every volume we load is a ``DataobjImage``, so cast to reach ``dataobj``.
    """
    image = cast(DataobjImage, nib.load(path))  # pyright: ignore[reportUnknownMemberType]  # nib.load has untyped **kwargs
    return np.asarray(image.dataobj)


class BaseDataModule(L.LightningDataModule):
    """Generic train/val DataLoader plumbing around two pre-built datasets."""

    def __init__(
        self,
        train_dataset: Dataset[Sample],
        val_dataset: Dataset[Sample],
        batch_size: int = 8,
        num_workers: int = 0,
        class_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = list(class_names) if class_names is not None else None

    def train_dataloader(self) -> DataLoader[Sample]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Sample]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class SegmentationCVDataModule(L.LightningDataModule):
    """5-fold cross-validation over the slices of a single ``.nii`` volume pair.

    Loads both volumes once in :meth:`setup`, partitions the slices into ``n_folds``
    contiguous blocks, uses block ``fold`` for validation and the rest for training, and
    shares the loaded arrays between the train/val datasets.
    """

    def __init__(
        self,
        images_path: str | Path,
        masks_path: str | Path,
        n_folds: int = 5,
        fold: int = 0,
        batch_size: int = 4,
        num_workers: int = 0,
        size: int = 384,
    ) -> None:
        super().__init__()
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold {fold} out of range for n_folds {n_folds}")
        self.images_path = Path(images_path)
        self.masks_path = Path(masks_path)
        self.n_folds = n_folds
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.train_dataset: NiiSegmentationDataset | None = None
        self.val_dataset: NiiSegmentationDataset | None = None
        # Exposed for tests / inspection.
        self.train_indices: np.ndarray | None = None
        self.val_indices: np.ndarray | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        images = _load_volume(self.images_path)
        masks = _load_volume(self.masks_path)
        n_slices = images.shape[2]

        folds = contiguous_folds(n_slices, self.n_folds)
        self.val_indices = folds[self.fold]
        self.train_indices = np.concatenate([f for i, f in enumerate(folds) if i != self.fold])

        self.train_dataset = NiiSegmentationDataset(
            images, masks, self.train_indices, transform=build_d4_transform(self.size)
        )
        self.val_dataset = NiiSegmentationDataset(
            images, masks, self.val_indices, transform=build_eval_transform(self.size)
        )

    def train_dataloader(self) -> DataLoader[Sample]:
        assert self.train_dataset is not None, "call setup() before requesting a dataloader"
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[Sample]:
        assert self.val_dataset is not None, "call setup() before requesting a dataloader"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
