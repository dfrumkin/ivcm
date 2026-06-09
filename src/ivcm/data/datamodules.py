"""Lightning data modules wrapping the synthetic datasets."""

from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from ivcm.data.synthetic import SyntheticClassificationDataset, SyntheticSegmentationDataset
from ivcm.data.transforms import build_d4_transform, build_eval_transform


class _SyntheticDataModule(L.LightningDataModule):
    """Shared train/val dataloader plumbing for the synthetic datasets."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 384,
        train_samples: int = 64,
        val_samples: int = 32,
        batch_size: int = 8,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def _make_dataset(self, num_samples: int, seed: int, transform):  # pragma: no cover - overridden
        raise NotImplementedError

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = self._make_dataset(self.train_samples, seed=0, transform=build_d4_transform())
        if self.val_dataset is None:
            # Disjoint seed range from training so val samples differ.
            self.val_dataset = self._make_dataset(self.val_samples, seed=10_000, transform=build_eval_transform())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SyntheticClassificationDataModule(_SyntheticDataModule):
    def _make_dataset(self, num_samples: int, seed: int, transform):
        return SyntheticClassificationDataset(
            num_samples=num_samples,
            num_classes=self.num_classes,
            image_size=self.image_size,
            seed=seed,
            transform=transform,
        )


class SyntheticSegmentationDataModule(_SyntheticDataModule):
    def _make_dataset(self, num_samples: int, seed: int, transform):
        return SyntheticSegmentationDataset(
            num_samples=num_samples,
            num_classes=self.num_classes,
            image_size=self.image_size,
            seed=seed,
            transform=transform,
        )
