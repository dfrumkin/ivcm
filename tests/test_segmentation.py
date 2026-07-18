"""Focused unit tests for the crop-training / sliding-window segmentation building blocks.

Checkpoint-free: they exercise the transform builders and the ``sliding_window_logits`` wrapper
directly, which is where the crop/native-resolution change is centered.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch

from ivcm.data import build_crop_transform, build_seg_eval_transform
from ivcm.models.segmentation import sliding_window_logits


def test_build_crop_transform_crops_image_and_mask_jointly() -> None:
    """The training transform yields a (crop, crop) image tensor and an aligned (crop, crop) mask."""
    crop = 32
    image = np.random.default_rng(0).random((48, 60, 1), dtype=np.float32)  # HWC, larger than crop
    mask = np.zeros((48, 60), dtype=np.int64)

    out = cast("dict[str, torch.Tensor]", build_crop_transform(crop)(image=image, mask=mask))

    assert out["image"].shape == (1, crop, crop)  # CHW, single channel
    assert out["mask"].shape == (crop, crop)  # cropped jointly with the image


def test_build_crop_transform_pads_when_smaller_than_crop() -> None:
    """pad_if_needed lets the crop succeed even when the image is smaller than the crop size."""
    crop = 32
    image = np.random.default_rng(1).random((20, 12, 1), dtype=np.float32)  # smaller than crop
    mask = np.zeros((20, 12), dtype=np.int64)

    out = cast("dict[str, torch.Tensor]", build_crop_transform(crop)(image=image, mask=mask))

    assert out["image"].shape == (1, crop, crop)
    assert out["mask"].shape == (crop, crop)


def test_build_seg_eval_transform_preserves_native_size() -> None:
    """The eval transform converts to a CHW tensor without resizing (native H x W preserved)."""
    image = np.random.default_rng(2).random((40, 50, 1), dtype=np.float32)
    mask = np.zeros((40, 50), dtype=np.int64)

    out = cast("dict[str, torch.Tensor]", build_seg_eval_transform()(image=image, mask=mask))

    assert out["image"].shape == (1, 40, 50)  # native H, W (no resize)
    assert out["mask"].shape == (40, 50)


class _PixelwisePredictor(torch.nn.Module):
    """(B, 1, H, W) -> (B, 2, H, W): a purely per-pixel map, so window stitching is exact."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, -x], dim=1)


def test_sliding_window_logits_full_roi_equals_direct_forward() -> None:
    """A ROI covering the whole image is a single window, so the result equals a direct forward."""
    model = _PixelwisePredictor().eval()
    images = torch.rand(2, 1, 24, 20)

    out = sliding_window_logits(model.forward, images, roi_size=24, overlap=0.25, sw_batch_size=4)

    assert out.shape == (2, 2, 24, 20)  # full-image logits, (B, num_classes, H, W)
    with torch.inference_mode():
        torch.testing.assert_close(out, model(images))


def test_sliding_window_logits_tiles_and_reconstructs_full_image() -> None:
    """With a ROI smaller than the image and overlap, the stitched logits still cover the full
    image and, for a per-pixel predictor, match the direct forward (overlap averages equal values)."""
    model = _PixelwisePredictor().eval()
    images = torch.rand(1, 1, 40, 30)

    out = sliding_window_logits(model.forward, images, roi_size=16, overlap=0.5, sw_batch_size=2)

    assert out.shape == (1, 2, 40, 30)  # reconstructed at the full native resolution
    with torch.inference_mode():
        torch.testing.assert_close(out, model(images))
