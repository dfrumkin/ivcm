"""Unit tests for ``update_summary`` and the best-weights artifact helper's no-op gate."""

from __future__ import annotations

from pathlib import Path

import pytest
import wandb
from omegaconf import OmegaConf

from ivcm.utils import finish_wandb, update_summary
from ivcm.utils.wandb import log_best_weights_artifact


def test_update_summary_overwrites_active_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``update_summary`` writes values into the active run's summary, bypassing the
    step-monotonicity that drops a post-fit best-checkpoint validation log. Also accepts an int
    (``best_epoch``)."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb.init(project="ivcm-test", dir=str(tmp_path))
    assert wandb.run is not None
    try:
        wandb.run.summary["val/auc"] = 0.1  # stand in for the dropped last-epoch value
        update_summary({"val/auc": 0.9, "val/dice": 0.8, "best_epoch": 27})

        assert wandb.run.summary["val/auc"] == 0.9
        assert wandb.run.summary["val/dice"] == 0.8
        assert wandb.run.summary["best_epoch"] == 27
    finally:
        finish_wandb()


def test_update_summary_noop_without_run() -> None:
    """No active run -> no error, nothing written."""
    finish_wandb()  # ensure no run is active
    update_summary({"val/auc": 0.9})  # must not raise


def test_log_best_weights_artifact_noop_without_run(tmp_path: Path) -> None:
    """No active run -> the artifact helper writes nothing and does not raise."""
    finish_wandb()  # ensure no run is active
    cfg = OmegaConf.create({})
    log_best_weights_artifact(cfg, {}, 3, tmp_path)  # must not raise
    assert not (tmp_path / "best-weights.ckpt").exists()  # gated out before any staging
