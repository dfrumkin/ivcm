"""Lightning data modules.

``BaseDataModule`` is the generic case: it holds two already-instantiated datasets (built
by Hydra from config) and owns only the DataLoader plumbing. The real/synthetic
distinction therefore lives entirely in the dataset configs, not in datamodule classes.

``SegmentationCVDataModule`` is the one exception: 5-fold cross-validation couples the
train/val split (one volume load, one fold index decides which slices are validation), so
it builds its datasets internally.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import lightning as L
import nibabel as nib
import numpy as np
import torch
from nibabel.dataobj_images import DataobjImage
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ivcm.data.real import (
    NiiSegmentationDataset,
    Segmentation2Dataset,
    contiguous_folds,
    discover_segmentation2,
    inverse_frequency_weights,
    patient_folds,
)
from ivcm.data.transforms import build_crop_transform, build_seg_eval_transform

log = logging.getLogger(__name__)


@runtime_checkable
class _HasClassCounts(Protocol):
    """A dataset that can report per-class sample counts (for inverse-frequency loss weights)."""

    def class_counts(self, num_classes: int) -> list[int]: ...


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
        weighted_loss: bool = True,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = list(class_names) if class_names is not None else None

        # Inverse-frequency class weights from the train set's per-class counts, applied automatically
        # (real classification data). `None` when disabled, no class_names, or a dataset that can't
        # report counts (e.g. the synthetic/segmentation datasets) — so the data-free path is a no-op.
        self.class_weights: torch.Tensor | None = None
        if weighted_loss and self.class_names is not None and isinstance(train_dataset, _HasClassCounts):
            self.class_weights = inverse_frequency_weights(train_dataset.class_counts(len(self.class_names)))

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
        crop_size: int = 192,
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
        self.crop_size = crop_size
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

        # Train on native-resolution crops; validate on full native-resolution slices (fed to
        # sliding-window inference). All slices in one volume share H×W, so val still batches.
        self.train_dataset = NiiSegmentationDataset(
            images, masks, self.train_indices, transform=build_crop_transform(self.crop_size)
        )
        self.val_dataset = NiiSegmentationDataset(images, masks, self.val_indices, transform=build_seg_eval_transform())

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


class Segmentation2CVDataModule(L.LightningDataModule):
    """Patient-grouped cross-validation over ``data/segmentation2`` (two-annotator soft masks).

    ``images_path``/``masks_path`` given together ⇒ combined mode: the whole ``data/segmentation``
    volume is added to training in every fold (it has no patient ids to split on), while validation
    stays only the held-out ``data/segmentation2`` patients. ``batch_size`` governs **training only**;
    validation uses ``batch_size = 1`` because per-instance PNGs need not share a size (and
    sliding-window inference needs no batching). Exposes ``n_folds``/``fold`` so ``cv.py`` drives it
    like :class:`SegmentationCVDataModule`.
    """

    def __init__(
        self,
        root: str | Path,
        images_path: str | Path | None = None,
        masks_path: str | Path | None = None,
        n_folds: int = 5,
        fold: int = 0,
        batch_size: int = 4,
        num_workers: int = 0,
        crop_size: int = 192,
    ) -> None:
        super().__init__()
        if (images_path is None) != (masks_path is None):
            raise ValueError(
                "images_path and masks_path must be given together (combined mode) or both omitted "
                f"(segmentation2-only); got images_path={images_path!r}, masks_path={masks_path!r}."
            )
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2 (a single fold would leave training empty); got {n_folds}.")
        if not 0 <= fold < n_folds:
            raise ValueError(f"fold {fold} out of range for n_folds {n_folds}")
        self.root = Path(root)
        self.images_path = Path(images_path) if images_path is not None else None
        self.masks_path = Path(masks_path) if masks_path is not None else None
        self.n_folds = n_folds
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.train_dataset: Dataset[Sample] | None = None
        self.val_dataset: Segmentation2Dataset | None = None
        # Exposed for tests / inspection.
        self.train_patients: list[str] | None = None
        self.val_patients: list[str] | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        instances = discover_segmentation2(self.root)
        patients = [patient for _dir, patient in instances]
        n_patients = len(set(patients))
        if self.n_folds > n_patients:
            raise ValueError(f"n_folds {self.n_folds} exceeds the number of patients {n_patients} under {self.root}.")

        folds = patient_folds(patients, self.n_folds)
        val_patients: set[str] = set(folds[self.fold].tolist())
        self.val_patients = sorted(val_patients)
        self.train_patients = sorted(set(patients) - val_patients)

        ds2_train_dirs = [d for d, p in instances if p not in val_patients]
        ds2_val_dirs = [d for d, p in instances if p in val_patients]
        # patient_folds balances by patient count, not instance count; log both so an
        # instance-per-patient imbalance (which can make a fold's metrics noisier) is visible.
        log.info(
            "segmentation2 fold %d: train %d patients / %d instances; val %d patients / %d instances",
            self.fold,
            len(self.train_patients),
            len(ds2_train_dirs),
            len(self.val_patients),
            len(ds2_val_dirs),
        )

        ds2_train = Segmentation2Dataset(ds2_train_dirs, transform=build_crop_transform(self.crop_size))
        self.val_dataset = Segmentation2Dataset(ds2_val_dirs, transform=build_seg_eval_transform())

        if self.images_path is not None and self.masks_path is not None:
            images = _load_volume(self.images_path)
            masks = _load_volume(self.masks_path)
            n_slices = images.shape[2]
            ds1 = NiiSegmentationDataset(
                images, masks, list(range(n_slices)), transform=build_crop_transform(self.crop_size)
            )
            log.info("combined mode: %d ds1 volume slices added to training (all folds)", n_slices)
            self.train_dataset = ConcatDataset([cast("Dataset[Sample]", ds2_train), cast("Dataset[Sample]", ds1)])
        else:
            self.train_dataset = ds2_train

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
        # batch_size=1: native-resolution instances may differ in size, so they can't be collated
        # into a batch; sliding-window inference validates each image on its own.
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
