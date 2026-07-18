"""Square-input enforcement across the validation helper, the datasets, ``load_image``, and the
UI directory pre-scan.

All classification/segmentation inputs are guaranteed square (pixel height == width); a non-square
input is a fatal invariant violation. These tests pin that the enforcement fires at each input
boundary and that the UI pre-scan makes *only* non-square images fatal (unreadable files stay the
batch's ``skipped`` responsibility).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ivcm.app.batch import ensure_all_square
from ivcm.app.inference import load_image
from ivcm.data import (
    ClassificationImageDataset,
    NiiSegmentationDataset,
    NonSquareImageError,
    ensure_square,
)


def _write_image(path: Path, height: int, width: int) -> None:
    """Write a grayscale image of the given pixel shape (H, W) to ``path``."""
    Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L").save(path)


def test_ensure_square_accepts_square_and_rejects_non_square() -> None:
    ensure_square((16, 16), source="ok")  # square: no raise
    ensure_square((16, 16, 1), source="ok-hwc")  # leading (H, W) equal: no raise
    with pytest.raises(NonSquareImageError):
        ensure_square((4, 8), source="bad")


def test_segmentation_dataset_rejects_non_square_slice() -> None:
    """A non-square ``.nii`` slice fails loudly at access time (covers training + OOF prediction)."""
    non_square = np.zeros((4, 8, 2), dtype=np.uint8)  # (H, W, n_slices), H != W
    dataset = NiiSegmentationDataset(non_square, non_square, [0])
    with pytest.raises(NonSquareImageError):
        _ = dataset[0]

    square = np.zeros((4, 4, 2), dtype=np.uint8)
    ok = NiiSegmentationDataset(square, square, [0])
    _ = ok[0]  # square slice: no raise


def test_classification_dataset_rejects_non_square_image(tmp_path: Path) -> None:
    """A non-square ``.tif`` under a class folder fails loudly at access time (training path)."""
    class_dir = tmp_path / "cells"
    class_dir.mkdir()
    _write_image(class_dir / "wide.tif", height=8, width=16)
    dataset = ClassificationImageDataset(tmp_path, class_names=["cells"])
    with pytest.raises(NonSquareImageError):
        _ = dataset[0]

    square_dir = tmp_path / "square"
    class_dir_sq = square_dir / "cells"
    class_dir_sq.mkdir(parents=True)
    _write_image(class_dir_sq / "ok.tif", height=12, width=12)
    ok = ClassificationImageDataset(square_dir, class_names=["cells"])
    _ = ok[0]  # square image: no raise


def test_load_image_rejects_non_square(tmp_path: Path) -> None:
    """``load_image`` enforces the invariant so the inference engine only yields square images."""
    bad = tmp_path / "bad.png"
    _write_image(bad, height=8, width=16)
    with pytest.raises(NonSquareImageError):
        load_image(bad)

    good = tmp_path / "good.png"
    _write_image(good, height=12, width=12)
    image = load_image(good)
    assert image.shape == (12, 12, 1)


def test_ensure_all_square_fatal_only_for_non_square(tmp_path: Path) -> None:
    """The UI pre-scan raises for a non-square file, passes when all are square, and does not treat
    an unreadable file as fatal (that stays ``process_directory``'s skipped responsibility)."""
    square_a = tmp_path / "a.png"
    square_b = tmp_path / "b.png"
    _write_image(square_a, 10, 10)
    _write_image(square_b, 20, 20)
    ensure_all_square([square_a, square_b])  # all square: no raise

    non_square = tmp_path / "wide.png"
    _write_image(non_square, height=10, width=20)
    with pytest.raises(NonSquareImageError):
        ensure_all_square([square_a, non_square])

    # An unreadable/corrupt file is not fatal here — the pre-scan skips it (left to the batch).
    corrupt = tmp_path / "corrupt.tif"
    corrupt.write_bytes(b"not an image")
    ensure_all_square([square_a, corrupt, square_b])  # no raise
