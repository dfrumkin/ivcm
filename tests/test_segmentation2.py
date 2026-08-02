"""Tests for the per-patient two-annotator segmentation data path (data/segmentation2)."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import cast

import lightning as L
import nibabel as nib
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from PIL import Image

from ivcm.data import Segmentation2CVDataModule, Segmentation2Dataset, build_crop_transform

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _make_instance(inst_dir: Path, size: int = 32, *, with_gt2: bool = True, seed: int = 0) -> None:
    inst_dir.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(seed)
    _save_png(inst_dir / "image.png", gen.integers(0, 256, (size, size)))
    gt = np.zeros((size, size), np.uint8)
    gt[: size // 2] = 255  # reader 1: top half
    _save_png(inst_dir / "gt.png", gt)
    if with_gt2:
        gt2 = np.zeros((size, size), np.uint8)
        gt2[: size // 4] = 255  # reader 2: top quarter -> disagreement band in [size//4, size//2)
        _save_png(inst_dir / "gt2.png", gt2)


def _make_volume(seg_dir: Path, n_slices: int = 6, size: int = 32) -> None:
    seg_dir.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(0)
    imgs = gen.integers(0, 256, (size, size, n_slices)).astype(np.int16)
    masks = gen.integers(0, 2, (size, size, n_slices)).astype(np.uint16)
    nib.save(nib.Nifti1Image(imgs, np.eye(4)), seg_dir / "imgs.nii")  # pyright: ignore[reportUnknownMemberType]  # nibabel untyped
    nib.save(nib.Nifti1Image(masks, np.eye(4)), seg_dir / "mask_gt.nii")  # pyright: ignore[reportUnknownMemberType]  # nibabel untyped


# ------- Test 1: two-annotator GT construction -------


def test_two_annotator_gt_construction(tmp_path: Path) -> None:
    both = tmp_path / "p" / "both"
    _make_instance(both, 32, with_gt2=True)
    _img, p = Segmentation2Dataset([both])[0]  # no transform -> raw numpy mask
    assert set(np.unique(np.asarray(p)).tolist()) == {0.0, 0.5, 1.0}

    # 0.5 survives the crop transform (whole-image crop keeps every pixel; ToTensorV2 keeps float).
    _img_t, p_t = Segmentation2Dataset([both], transform=build_crop_transform(32))[0]
    assert bool((p_t == 0.5).any()) and float(p_t.min()) == 0.0 and float(p_t.max()) == 1.0

    single = tmp_path / "p" / "single"
    _make_instance(single, 32, with_gt2=False)
    _img2, p2 = Segmentation2Dataset([single])[0]
    assert set(np.unique(np.asarray(p2)).tolist()) == {0.0, 1.0}


# ------- Test 2: patient-grouped split (mode b) -------


def _seg2_tree(root: Path) -> None:
    # 4 patients; pB has two instances, one 48px (mixed native sizes); pD is single-annotator.
    _make_instance(root / "pA" / "i1", 32, with_gt2=True, seed=1)
    _make_instance(root / "pB" / "i1", 32, with_gt2=True, seed=2)
    _make_instance(root / "pB" / "i2", 48, with_gt2=True, seed=3)
    _make_instance(root / "pC" / "i1", 32, with_gt2=True, seed=4)
    _make_instance(root / "pD" / "i1", 32, with_gt2=False, seed=5)


def test_patient_grouped_split_disjoint_and_deterministic(tmp_path: Path) -> None:
    _seg2_tree(tmp_path / "segmentation2")
    n_folds = 2
    all_val: set[str] = set()
    for fold in range(n_folds):
        dm = Segmentation2CVDataModule(tmp_path / "segmentation2", n_folds=n_folds, fold=fold, crop_size=32)
        dm.setup()
        assert dm.train_patients is not None and dm.val_patients is not None
        assert set(dm.train_patients).isdisjoint(dm.val_patients)
        all_val |= set(dm.val_patients)
        # Deterministic: a second identical module yields the same split.
        dm2 = Segmentation2CVDataModule(tmp_path / "segmentation2", n_folds=n_folds, fold=fold, crop_size=32)
        dm2.setup()
        assert dm2.val_patients == dm.val_patients
    assert all_val == {"pA", "pB", "pC", "pD"}


def test_segmentation2_config_wiring_and_val_batch(tmp_path: Path) -> None:
    _seg2_tree(tmp_path / "segmentation2")
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="cv",
            overrides=[f"data_dir={tmp_path}", "data=segmentation2_real", "data.n_folds=2", "crop_size=32"],
        )
    dm = instantiate(cfg.data)
    dm.setup()
    # Train collates crops fine; validation (batch_size 1) handles the mixed native sizes (32 and 48).
    img, mask = next(iter(dm.train_dataloader()))
    assert img.shape[-2:] == (32, 32) and mask.shape[-2:] == (32, 32)
    for vb_img, _vb_mask in dm.val_dataloader():
        assert vb_img.shape[0] == 1  # per-instance validation


# ------- Test 3: combined mode (mode c) -------


def test_combined_mode_ds1_in_train_val_only_ds2(tmp_path: Path) -> None:
    _make_instance(tmp_path / "segmentation2" / "pA" / "i1", 32, with_gt2=True, seed=1)
    _make_instance(tmp_path / "segmentation2" / "pB" / "i1", 32, with_gt2=True, seed=2)
    _make_volume(tmp_path / "segmentation", n_slices=4, size=32)

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="cv",
            overrides=[
                f"data_dir={tmp_path}",
                "data=segmentation_combined_real",
                "data.n_folds=2",
                "data.fold=0",
                "crop_size=32",
            ],
        )
    dm = instantiate(cfg.data)
    dm.setup()

    # fold 0 val = 1 patient (1 instance); train = the other ds2 patient (1) + all 4 ds1 slices.
    assert isinstance(dm.val_dataset, Segmentation2Dataset)
    assert len(cast(Sized, dm.val_dataset)) == 1
    assert len(cast(Sized, dm.train_dataset)) == 1 + 4  # ds2 train instance + every ds1 slice

    img, mask = next(iter(dm.train_dataloader()))
    assert img.shape[-2:] == (32, 32) and mask.dtype == torch.float32
    vb_img, _ = next(iter(dm.val_dataloader()))
    assert vb_img.shape[0] == 1


# ------- Test 9: mode (a) volume path under the combo loss -------


def test_mode_a_volume_trains_under_combo_loss(tmp_path: Path) -> None:
    _make_volume(tmp_path / "segmentation", n_slices=6, size=32)
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=segmentation",
                "data=segmentation_real",
                f"data_dir={tmp_path}",
                "image_size=32",
                "crop_size=32",
                "data.batch_size=2",
            ],
        )
    datamodule = instantiate(cfg.data)
    model = instantiate(
        cfg.model,
        criterion=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=instantiate(cfg.scheduler),
    )
    trainer = L.Trainer(
        fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False, enable_checkpointing=False
    )
    trainer.fit(model, datamodule=datamodule)
    loss = trainer.callback_metrics.get("train/loss_step", trainer.callback_metrics.get("train/loss"))
    assert loss is not None and torch.isfinite(loss)


# ------- Test 10 & 11: fail-loud integrity and split-contract validation -------


def test_discover_fails_on_missing_files_and_empty_tree(tmp_path: Path) -> None:
    from ivcm.data import discover_segmentation2

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        discover_segmentation2(empty)  # no instances

    broken = tmp_path / "broken" / "pA" / "i1"
    broken.mkdir(parents=True)
    _save_png(broken / "image.png", np.zeros((16, 16), np.uint8))  # gt.png missing
    with pytest.raises(ValueError):
        discover_segmentation2(tmp_path / "broken")


def test_dataset_rejects_size_mismatch(tmp_path: Path) -> None:
    inst = tmp_path / "pA" / "i1"
    inst.mkdir(parents=True)
    _save_png(inst / "image.png", np.zeros((32, 32), np.uint8))
    _save_png(inst / "gt.png", np.zeros((32, 32), np.uint8))
    _save_png(inst / "gt2.png", np.zeros((16, 16), np.uint8))  # wrong size
    with pytest.raises(ValueError):
        Segmentation2Dataset([inst])[0]


def test_split_contract_validation(tmp_path: Path) -> None:
    _seg2_tree(tmp_path / "segmentation2")
    root = tmp_path / "segmentation2"
    with pytest.raises(ValueError):
        Segmentation2CVDataModule(root, n_folds=1)  # < 2
    with pytest.raises(ValueError):
        Segmentation2CVDataModule(root, n_folds=3, fold=3)  # fold out of range
    with pytest.raises(ValueError):
        Segmentation2CVDataModule(root, images_path=tmp_path / "x.nii")  # partial volume pair
    with pytest.raises(ValueError):
        Segmentation2CVDataModule(root, n_folds=99).setup()  # n_folds > num_patients (checked in setup)
