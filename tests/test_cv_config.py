import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# These tests deliberately exercise cv.py's private helpers (white-box unit tests), so the
# cross-module private-use warnings are suppressed per imported name.
from ivcm.cv import (
    METRIC_KEYS,
    _completed_metrics,  # pyright: ignore[reportPrivateUsage]
    _find_resume_ckpt,  # pyright: ignore[reportPrivateUsage]
    _fold_wandb_id,  # pyright: ignore[reportPrivateUsage]
    _prepare_fold_cfg,  # pyright: ignore[reportPrivateUsage]
    _reject_resume,  # pyright: ignore[reportPrivateUsage]
    _save_fold_metrics,  # pyright: ignore[reportPrivateUsage]
)

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")

_METRICS = {k: 0.5 for k in METRIC_KEYS}


def test_cv_config_wiring():
    """cv.yaml composes the segmentation stack with the real CV datamodule (no default-task leak)."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv")

    assert cfg.model._target_.endswith("SegmentationModule")
    assert cfg.data._target_.endswith("SegmentationCVDataModule")
    assert cfg.data.n_folds == 5
    assert cfg.data.fold == 0  # default fold, before the cv loop mutates it
    assert cfg.seed is not None  # run() reads cfg.seed unconditionally


def test_cv_config_resolves():
    """Full resolve (logger disabled to dodge ${hydra:...}) catches resolver/root-field gaps."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv", overrides=["logger=none"])

    OmegaConf.to_yaml(cfg, resolve=True)  # raises if any interpolation/resolver is broken
    assert cfg.model.net.classes == 2  # ${len:${class_names}} over [background, nerve]
    assert cfg.model.num_classes == 2
    assert cfg.data.fold == 0
    assert "trainer" in cfg  # provided by the shared common.yaml


def test_cv_rejects_global_ckpt_path():
    """cross-validate must refuse a global ckpt_path: one checkpoint cannot map to per-fold runs."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv", overrides=["ckpt_path=/some/checkpoint.ckpt"])

    with pytest.raises(ValueError, match="does not accept a global ckpt_path"):
        _reject_resume(cfg)

    # The default (no ckpt_path) passes the guard.
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        _reject_resume(compose(config_name="cv"))


def test_completed_metrics_roundtrip(tmp_path: Path) -> None:
    """A finished fold's metrics persist and are read back; missing/corrupt -> not complete."""
    fold_dir = tmp_path / "fold_0"
    assert _completed_metrics(fold_dir) is None  # nothing saved yet

    _save_fold_metrics(fold_dir, _METRICS)
    assert _completed_metrics(fold_dir) == _METRICS

    # Missing a required key -> treated as not complete (fold should re-run).
    (fold_dir / "metrics.json").write_text(json.dumps({"val/auc": 0.5}))
    assert _completed_metrics(fold_dir) is None

    # Corrupt JSON -> not complete.
    (fold_dir / "metrics.json").write_text("{not json")
    assert _completed_metrics(fold_dir) is None


def test_find_resume_ckpt(tmp_path: Path) -> None:
    """Returns the fold's last.ckpt at the stable <fold_dir>/checkpoints path, else None."""
    fold_dir = tmp_path / "fold_1"
    assert _find_resume_ckpt(fold_dir) is None  # dir absent
    fold_dir.mkdir()
    assert _find_resume_ckpt(fold_dir) is None  # no checkpoint

    ckpt = fold_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("ckpt")
    assert _find_resume_ckpt(fold_dir) == str(ckpt)


def test_fold_wandb_id_is_stable(tmp_path: Path) -> None:
    """A fold's W&B id is generated once and reused so a resumed fold continues the same run."""
    fold_dir = tmp_path / "fold_2"
    first = _fold_wandb_id(fold_dir)
    assert first and (fold_dir / "wandb_run_id.txt").exists()
    assert _fold_wandb_id(fold_dir) == first  # reloaded, not regenerated


def test_prepare_fold_cfg_resumes_from_fold_checkpoint(tmp_path: Path) -> None:
    """The per-fold config selects the fold, routes outputs to fold_dir, and pins ckpt_path to
    the fold's last.ckpt when present (the wiring the CV loop relies on for resume)."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv", overrides=["logger=none", "callbacks=default"])
    fold_dir = tmp_path / "fold_2"

    fresh = _prepare_fold_cfg(cfg, 2, fold_dir, "grp")
    assert fresh.data.fold == 2
    assert fresh.trainer.default_root_dir == str(fold_dir)
    assert fresh.ckpt_path is None  # nothing to resume from yet

    ckpt = fold_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("ckpt")
    resumed = _prepare_fold_cfg(cfg, 2, fold_dir, "grp")
    assert resumed.ckpt_path == str(ckpt)


def test_prepare_fold_cfg_rejects_resume_without_callbacks(tmp_path: Path) -> None:
    """Resuming a fold checkpoint without callbacks=default would lose best-checkpoint tracking,
    so it must fail loudly rather than silently degrade to final-model metrics."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv", overrides=["logger=none"])  # callbacks=none (default)
    fold_dir = tmp_path / "fold_3"
    ckpt = fold_dir / "checkpoints" / "last.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("ckpt")

    with pytest.raises(ValueError, match="re-run with callbacks=default"):
        _prepare_fold_cfg(cfg, 3, fold_dir, "grp")
