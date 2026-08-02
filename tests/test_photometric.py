"""Tests for the configurable, on-by-default photometric training augmentation.

Covers the two behavioral requirements — on by default (active for both tasks), fully disableable —
and the wiring that carries the config into both transform paths: the Hydra-instantiated
``BaseDataModule`` transforms and the segmentation CV datamodules (which build the crop transform in
code). Also guards the ``NiiSegmentationDataset`` ``[0, 1]`` clamp the ops depend on.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import albumentations as A  # pyright: ignore[reportMissingTypeStubs]
import nibabel as nib
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import ConcatDataset

from ivcm.data import NiiSegmentationDataset
from ivcm.data.transforms import build_crop_transform, build_d4_transform

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# A spatially uniform, mid-range constant: geometric ops (D4/crop) leave it identical, so any change
# in pixel value is attributable to the photometric ops. Mid-range (not 0/1) matters — gamma leaves
# 1.0 unchanged and brightness/contrast clip at the [0, 1] ends.
_MID = 0.5


def _photometric_names(compose_t: A.Compose) -> list[str]:
    return [type(t).__name__ for t in cast("Sequence[Any]", compose_t.transforms)]


def _has_photometric(compose_t: A.Compose) -> bool:
    names = _photometric_names(compose_t)
    return "RandomBrightnessContrast" in names and "RandomGamma" in names


def _const_image(size: int = 32) -> np.ndarray:
    return np.full((size, size, 1), _MID, dtype=np.float32)


def _changes_constant(compose_t: A.Compose, *, has_mask: bool) -> bool:
    """Does the transform alter a mid-range constant image under any of a few fixed seeds?

    Deterministic (fixed seeds via ``set_random_seed``), so non-flaky: with the default per-op
    ``p=0.5`` some seeds skip the ops, but across the fixed set at least one applies when enabled.
    """
    image = _const_image()
    for seed in range(8):
        compose_t.set_random_seed(seed)  # pyright: ignore[reportUnknownMemberType]  # albumentations untyped
        kwargs: dict[str, Any] = {"image": image}
        if has_mask:
            kwargs["mask"] = np.zeros((32, 32), dtype=np.float32)
        out = cast("dict[str, torch.Tensor]", compose_t(**kwargs))
        if not np.allclose(out["image"].numpy(), _MID):
            return True
    return False


# ------- On by default, active for BOTH builders -------


def test_photometric_on_by_default_classification() -> None:
    """build_d4_transform (default photometric=None) applies brightness/contrast/gamma."""
    t = build_d4_transform(32)
    assert _has_photometric(t)
    assert _changes_constant(t, has_mask=False)


def test_photometric_on_by_default_segmentation() -> None:
    """build_crop_transform (default photometric=None) applies brightness/contrast/gamma."""
    t = build_crop_transform(32)
    assert _has_photometric(t)
    assert _changes_constant(t, has_mask=True)


# ------- Disable switch works for BOTH builders (structural + behavioral) -------


def test_photometric_disabled_classification() -> None:
    disabled = {"enabled": False}
    t = build_d4_transform(32, disabled)
    assert not _has_photometric(t)  # structural: the ops are absent
    assert not _changes_constant(t, has_mask=False)  # behavioral: constant unchanged


def test_photometric_disabled_segmentation() -> None:
    disabled = {"enabled": False}
    t = build_crop_transform(32, disabled)
    assert not _has_photometric(t)
    assert not _changes_constant(t, has_mask=True)


def test_disabled_is_authoritative_without_range_fields() -> None:
    """enabled=false disables even when the range fields are absent (master switch is authoritative)."""
    t = build_crop_transform(32, {"enabled": False})  # no brightness/contrast/gamma keys
    assert not _has_photometric(t)


# ------- Mask values untouched; outputs stay in [0, 1] (incl. top of range) -------


def test_photometric_preserves_mask_values_and_output_range() -> None:
    """Photometric ops introduce no new mask values and never push the image outside [0, 1]."""
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[:20] = 0.5
    mask[20:] = 1.0
    input_values = {0.0, 0.5, 1.0}

    force_on = {"enabled": True, "brightness_limit": 0.2, "contrast_limit": 0.2, "gamma_limit": [80, 120], "p": 1.0}
    t = build_crop_transform(40, force_on)
    for seed in range(5):
        t.set_random_seed(seed)  # pyright: ignore[reportUnknownMemberType]  # albumentations untyped
        # Top-of-range image confirms brightness/contrast/gamma do not overshoot 1.0.
        out = cast("dict[str, torch.Tensor]", t(image=np.ones((40, 40, 1), dtype=np.float32), mask=mask))
        img = out["image"].numpy()
        assert np.isfinite(img).all()
        assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
        assert set(np.unique(out["mask"].numpy()).tolist()).issubset(input_values)


# ------- NiiSegmentationDataset clamps to [0, 1] -------


def test_nii_dataset_clamps_out_of_range_to_unit_interval() -> None:
    """A slice whose raw volume value exceeds 255 is clamped into [0, 1] before the transform."""
    images = np.full((8, 8, 1), 1000, dtype=np.int16)  # /255 -> ~3.92, must be clamped to 1.0
    masks = np.zeros((8, 8, 1), dtype=np.uint16)
    image, _mask = NiiSegmentationDataset(images, masks, [0])[0]
    arr = np.asarray(image)
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


# ------- Config/CLI disable reaches both wiring paths -------


def _instantiate_train_transform(task: str, *, disabled: bool) -> A.Compose:
    overrides = [f"task={task}"]
    if disabled:
        overrides.append("augmentation.photometric.enabled=false")
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    datamodule = instantiate(cfg.data)
    return cast("A.Compose", datamodule.train_dataset.transform)


def test_config_disable_reaches_basedatamodule_transforms() -> None:
    """`augmentation.photometric.enabled=false` removes photometric ops from the config-built
    BaseDataModule train transforms for both tasks (guards the transform-config wiring)."""
    for task in ("classification", "segmentation"):
        assert _has_photometric(_instantiate_train_transform(task, disabled=False))
        assert not _has_photometric(_instantiate_train_transform(task, disabled=True))


def _make_volume(tmp_path: Path) -> tuple[Path, Path]:
    """Write tiny paired image/mask NIfTI volumes; return their paths."""
    gen = np.random.default_rng(0)
    imgs = gen.integers(0, 256, (32, 32, 6)).astype(np.int16)
    masks = gen.integers(0, 2, (32, 32, 6)).astype(np.uint16)
    imgs_path, masks_path = tmp_path / "imgs.nii", tmp_path / "mask_gt.nii"
    nib.save(nib.Nifti1Image(imgs, np.eye(4)), imgs_path)  # pyright: ignore[reportUnknownMemberType]  # nibabel untyped
    nib.save(nib.Nifti1Image(masks, np.eye(4)), masks_path)  # pyright: ignore[reportUnknownMemberType]  # nibabel untyped
    return imgs_path, masks_path


def _make_seg2_root(tmp_path: Path, n_patients: int = 2) -> Path:
    """Build a tiny data/segmentation2 tree (<root>/<patient>/<instance>/image.png+gt.png)."""
    root = tmp_path / "segmentation2"
    for i in range(n_patients):
        inst = root / f"p{i}" / "inst0"
        inst.mkdir(parents=True)
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (32, 32)).astype(np.uint8)).save(inst / "image.png")
        gt = np.zeros((32, 32), np.uint8)
        gt[:16] = 255
        Image.fromarray(gt).save(inst / "gt.png")
    return root


def _train_transforms(train_dataset: object) -> list[A.Compose]:
    """The crop transform(s) behind a CV datamodule's train dataset (ConcatDataset in combined mode)."""
    if isinstance(train_dataset, ConcatDataset):
        subsets = cast("list[Any]", train_dataset.datasets)  # pyright: ignore[reportUnknownMemberType]  # torch untyped
        return [cast("A.Compose", d.transform) for d in subsets]
    return [cast("A.Compose", cast("Any", train_dataset).transform)]


@pytest.mark.parametrize("data_mode", ["segmentation_real", "segmentation2_real", "segmentation_combined_real"])
def test_cv_config_wires_photometric(tmp_path: Path, data_mode: str) -> None:
    """Every real segmentation CV config carries `photometric: ${augmentation.photometric}` into its
    datamodule, which builds the crop transform in setup() (no transform config field). Composed from
    the actual configs so a missing/typo'd wire in any of them fails; combined mode also covers the
    ds1 train crop. Driven by tiny temporary NIfTI/PNG data — no proprietary data needed."""
    imgs_path, masks_path = _make_volume(tmp_path)
    seg2_root = _make_seg2_root(tmp_path)

    per_mode: dict[str, list[str]] = {
        "segmentation_real": [f"data.images_path={imgs_path}", f"data.masks_path={masks_path}"],
        "segmentation2_real": ["data=segmentation2_real", f"data.root={seg2_root}", "data.n_folds=2"],
        "segmentation_combined_real": [
            "data=segmentation_combined_real",
            f"data.root={seg2_root}",
            f"data.images_path={imgs_path}",
            f"data.masks_path={masks_path}",
            "data.n_folds=2",
        ],
    }

    def train_transforms(*, disabled: bool) -> list[A.Compose]:
        overrides = ["crop_size=32", *per_mode[data_mode]]
        if disabled:
            overrides.append("augmentation.photometric.enabled=false")
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="cv", overrides=overrides)
        dm = instantiate(cfg.data)
        dm.setup()
        assert dm.train_dataset is not None
        return _train_transforms(dm.train_dataset)

    enabled = train_transforms(disabled=False)
    assert enabled and all(_has_photometric(t) for t in enabled)  # on by default, wire present
    disabled = train_transforms(disabled=True)
    assert disabled and not any(_has_photometric(t) for t in disabled)  # disabled config forwarded
