"""Synthetic datasets for smoke-testing the training pipeline.

No real data is used. Each sample is generated on the fly from a per-index seed so
the datasets are deterministic and reproducible. Crucially, the **target is correlated
with the image intensity** so the models can actually learn it: a smoke run should show
a decreasing loss and an improving metric rather than pure noise.

The signal is intensity-based (not position-based) so it is invariant under the D4
augmentations (rotations/flips) applied downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from ivcm.data.transforms import Transform


def _class_intensity(class_idx: int, num_classes: int) -> float:
    """Map a class index to a distinct base intensity in ``[0.1, 0.9]``."""
    if num_classes <= 1:
        return 0.5
    return 0.1 + 0.8 * (class_idx / (num_classes - 1))


class SyntheticClassificationDataset(Dataset[tuple[Any, int]]):
    """Single-channel images whose class is encoded by a radial ring frequency.

    The class controls the spatial frequency of a concentric-ring pattern. This signal
    is invariant under the D4 augmentations (it depends only on radius) and, unlike a
    flat intensity offset, survives BatchNorm normalisation, so a CNN can actually learn
    and generalise it.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        image_size: int = 384,
        noise: float = 0.05,
        seed: int = 0,
        transform: Transform | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.noise = noise
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Any, int]:
        rng = np.random.default_rng(self.seed + index)
        label = int(rng.integers(self.num_classes))

        size = self.image_size
        yy, xx = np.mgrid[0:size, 0:size]
        center = (size - 1) / 2.0
        radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2) / (size / 2.0)
        freq = label + 1.0  # distinct ring frequency per class
        image = 0.5 + 0.4 * np.sin(2.0 * np.pi * freq * radius)
        image = image + rng.normal(0.0, self.noise, size=(size, size))
        image = np.clip(image, 0.0, 1.0).astype(np.float32)[..., None]

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class SyntheticSegmentationDataset(Dataset[tuple[Any, Any]]):
    """Single-channel images with random disks; per-pixel intensity encodes the class.

    Background is class ``0``; foreground disks take classes ``1..num_classes-1``. The
    mask is therefore recoverable from intensity, making the segmentation learnable.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        image_size: int = 384,
        max_objects: int = 4,
        noise: float = 0.05,
        seed: int = 0,
        transform: Transform | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.max_objects = max_objects
        self.noise = noise
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        rng = np.random.default_rng(self.seed + index)
        size = self.image_size
        mask = np.zeros((size, size), dtype=np.int64)

        yy, xx = np.mgrid[0:size, 0:size]
        num_objects = int(rng.integers(1, self.max_objects + 1))
        for _ in range(num_objects):
            cls = int(rng.integers(1, max(2, self.num_classes)))
            cy, cx = rng.integers(0, size, size=2)
            radius = int(rng.integers(size // 12, size // 5))
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            mask[disk] = cls

        # Intensity = class base intensity + noise, so the mask is learnable from pixels.
        def intensity(c: int) -> float:
            return _class_intensity(int(c), self.num_classes)

        image = np.asarray(np.vectorize(intensity)(mask), dtype=np.float32)
        image = image + rng.normal(0.0, self.noise, size=(size, size))
        image = np.clip(image, 0.0, 1.0).astype(np.float32)[..., None]

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask
