"""Tests for the out-of-fold prediction command (``ivcm.predict`` / ``predict-cv``).

These build a *fake* finished CV directory (saved config + hydra.yaml + per-fold dummy
checkpoints + tiny .nii volumes) so the assembly, config, path-resolution, and device paths
are exercised without training. The private helpers are imported deliberately (white-box).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from nibabel.nifti1 import Nifti1Image
from omegaconf import DictConfig, OmegaConf
from torch import nn

from ivcm.models.segmentation import SegmentationModule
from ivcm.predict import (
    _find_best_ckpt,  # pyright: ignore[reportPrivateUsage]
    _load_run_cfg,  # pyright: ignore[reportPrivateUsage]
    _original_cwd,  # pyright: ignore[reportPrivateUsage]
    _predict_fold,  # pyright: ignore[reportPrivateUsage]
    _resolve_data_path,  # pyright: ignore[reportPrivateUsage]
    main,
    write_oof_predictions,
)

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")
# A non-identity affine so affine preservation is actually testable.
AFFINE = np.diag([1.5, 2.0, 3.0, 1.0]).astype(np.float64)


def _cpu() -> torch.device:
    """Typed stand-in for ``_select_device`` so tests stay on CPU regardless of hardware."""
    return torch.device("cpu")


def _write_volume(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(Nifti1Image(array, AFFINE), path)  # pyright: ignore[reportUnknownMemberType]  # nibabel untyped


def _read_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = cast(Nifti1Image, nib.load(path))  # pyright: ignore[reportUnknownMemberType]  # nib.load untyped **kwargs
    return np.asarray(img.dataobj), np.asarray(img.affine)


def _compose_cfg(images_path: Path | str, masks_path: Path | str, *, n_folds: int, image_size: int) -> DictConfig:
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(
            config_name="cv",
            overrides=[
                f"data.images_path={images_path}",
                f"data.masks_path={masks_path}",
                f"data.n_folds={n_folds}",
                "data.batch_size=2",
                f"image_size={image_size}",
            ],
        )


def _build_model(cfg: DictConfig) -> SegmentationModule:
    model: SegmentationModule = instantiate(
        cfg.model,
        criterion=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=instantiate(cfg.scheduler),
    )
    return model


def _make_cv_dir(cv_dir: Path, cfg: DictConfig, *, n_folds: int, cwd: Path) -> None:
    """Write a fake finished CV run: saved config + hydra.yaml + one best ckpt per fold."""
    hydra_dir = cv_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")
    OmegaConf.save(OmegaConf.create({"hydra": {"runtime": {"cwd": str(cwd)}}}), hydra_dir / "hydra.yaml")

    model = _build_model(cfg)
    for fold in range(n_folds):
        ckpt = cv_dir / f"fold_{fold}" / "checkpoints" / "best-epoch00.ckpt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, ckpt)


def test_assembles_full_oof_volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Produces a full-volume binary mask at GT shape/affine; saved n_folds (3, not the
    default 5) and the saved absolute paths are honoured (would fail on ambient defaults)."""
    monkeypatch.setattr("ivcm.predict._select_device", _cpu)
    h, w, n_slices, n_folds = 12, 10, 6, 3
    data_dir = tmp_path / "data"
    images_path, masks_path = data_dir / "imgs.nii", data_dir / "mask_gt.nii"
    rng = np.random.default_rng(0)
    _write_volume(images_path, rng.integers(0, 256, (h, w, n_slices), dtype=np.uint8))
    _write_volume(masks_path, rng.integers(0, 2, (h, w, n_slices), dtype=np.uint8))

    cfg = _compose_cfg(images_path, masks_path, n_folds=n_folds, image_size=16)
    cv_dir = tmp_path / "cv"
    _make_cv_dir(cv_dir, cfg, n_folds=n_folds, cwd=tmp_path)

    out_path = write_oof_predictions(_load_run_cfg(cv_dir), cv_dir)

    assert out_path == cv_dir / "predictions.nii"
    pred, affine = _read_volume(out_path)
    assert pred.shape == (h, w, n_slices)  # restored to GT resolution, not image_size (16)
    assert pred.dtype == np.uint8
    assert set(np.unique(pred).tolist()).issubset({0, 1})
    assert np.allclose(affine, AFFINE)  # GT affine preserved


def test_relative_data_paths_resolved_against_saved_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative data paths resolve against the run's saved hydra.runtime.cwd, so the command
    works when invoked from an unrelated working directory."""
    monkeypatch.setattr("ivcm.predict._select_device", _cpu)
    h, w, n_slices, n_folds = 8, 8, 4, 2
    data_root = tmp_path / "runroot"
    images_rel, masks_rel = Path("vols/imgs.nii"), Path("vols/mask_gt.nii")
    rng = np.random.default_rng(1)
    _write_volume(data_root / images_rel, rng.integers(0, 256, (h, w, n_slices), dtype=np.uint8))
    _write_volume(data_root / masks_rel, rng.integers(0, 2, (h, w, n_slices), dtype=np.uint8))

    assert _resolve_data_path(str(images_rel), data_root) == data_root / images_rel

    cfg = _compose_cfg(images_rel, masks_rel, n_folds=n_folds, image_size=8)
    cv_dir = tmp_path / "cv"
    _make_cv_dir(cv_dir, cfg, n_folds=n_folds, cwd=data_root)

    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)  # invoke from somewhere unrelated to the data

    out_path = write_oof_predictions(_load_run_cfg(cv_dir), cv_dir)
    assert out_path.is_file()


def test_cli_tolerates_hydra_only_interpolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The saved config carries logger.save_dir: ${hydra:runtime.output_dir} (no hydra:
    resolver outside Hydra). Driving the CLI main() must not resolve it -> no crash, and no
    Hydra metadata is written back into the finished run."""
    monkeypatch.setattr("ivcm.predict._select_device", _cpu)
    h, w, n_slices, n_folds = 8, 6, 4, 2
    data_dir = tmp_path / "data"
    images_path, masks_path = data_dir / "imgs.nii", data_dir / "mask_gt.nii"
    rng = np.random.default_rng(2)
    _write_volume(images_path, rng.integers(0, 256, (h, w, n_slices), dtype=np.uint8))
    _write_volume(masks_path, rng.integers(0, 2, (h, w, n_slices), dtype=np.uint8))

    cfg = _compose_cfg(images_path, masks_path, n_folds=n_folds, image_size=8)
    assert "${hydra:" in OmegaConf.to_yaml(cfg.logger, resolve=False)  # scenario is real
    cv_dir = tmp_path / "cv"
    _make_cv_dir(cv_dir, cfg, n_folds=n_folds, cwd=tmp_path)

    main([str(cv_dir)])  # must not raise

    assert str(cv_dir / "predictions.nii") in capsys.readouterr().out
    assert (cv_dir / "predictions.nii").is_file()
    # No Hydra metadata written into the finished run dir (only our two files remain).
    assert {p.name for p in (cv_dir / ".hydra").iterdir()} == {"config.yaml", "hydra.yaml"}


def test_missing_config_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a finished cross-validate run"):
        _load_run_cfg(tmp_path)


def test_missing_hydra_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a finished cross-validate run"):
        _original_cwd(tmp_path)


def test_hydra_yaml_without_cwd_raises(tmp_path: Path) -> None:
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    OmegaConf.save(OmegaConf.create({"hydra": {"runtime": {}}}), hydra_dir / "hydra.yaml")
    with pytest.raises(ValueError, match="no hydra.runtime.cwd"):
        _original_cwd(tmp_path)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="callbacks=default"):
        _find_best_ckpt(fold_dir)


class _SpyModel(nn.Module):
    """Records the device of each forward input and returns zero logits of the right shape."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seen: list[torch.device] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.device)
        b, _c, height, width = x.shape
        return torch.zeros(b, self.num_classes, height, width, device=x.device)


def test_predict_fold_moves_batch_to_device_and_returns_numpy() -> None:
    """The batch is moved to the passed device before the forward pass, and the result is a
    CPU uint8 NumPy array of shape (len(val_idx), H, W) at original resolution."""
    device = torch.device("cpu")
    h, w, n_slices = 8, 6, 4
    rng = np.random.default_rng(3)
    images = rng.integers(0, 256, (h, w, n_slices), dtype=np.uint8)
    masks = rng.integers(0, 2, (h, w, n_slices), dtype=np.uint8)
    val_idx = np.arange(n_slices)
    spy = _SpyModel()
    spy.to(device)

    preds = _predict_fold(cast(SegmentationModule, spy), images, masks, val_idx, size=5, batch_size=2, device=device)

    assert spy.seen and all(d == device for d in spy.seen)  # .to(device) happened pre-forward
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (n_slices, h, w)  # restored to original H, W (not size=5)
    assert preds.dtype == np.uint8
