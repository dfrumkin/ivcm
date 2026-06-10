"""Unit tests for ``update_summary`` (points the W&B run summary at the best checkpoint)."""

from __future__ import annotations

import wandb

from ivcm.utils import finish_wandb, update_summary


def test_update_summary_overwrites_active_run(tmp_path, monkeypatch):
    """``update_summary`` writes values into the active run's summary, bypassing the
    step-monotonicity that drops a post-fit best-checkpoint validation log."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    wandb.init(project="ivcm-test", dir=str(tmp_path))
    try:
        wandb.run.summary["val/auc"] = 0.1  # stand in for the dropped last-epoch value
        update_summary({"val/auc": 0.9, "val/dice": 0.8})

        assert wandb.run.summary["val/auc"] == 0.9
        assert wandb.run.summary["val/dice"] == 0.8
    finally:
        finish_wandb()


def test_update_summary_noop_without_run():
    """No active run -> no error, nothing written."""
    finish_wandb()  # ensure no run is active
    update_summary({"val/auc": 0.9})  # must not raise
