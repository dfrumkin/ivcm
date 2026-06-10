"""Tests for the `callbacks=default` group: the monitored best-checkpoint callback and the
best-checkpoint validation path it enables in `train.run()`."""

from pathlib import Path

import lightning as L
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint

import ivcm  # noqa: F401 — registers the `len` OmegaConf resolver on import
from ivcm.train import run
from ivcm.utils import finish_wandb

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _checkpoint_callback(callbacks):
    cbs = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
    assert len(cbs) == 1, f"expected exactly one ModelCheckpoint, got {cbs}"
    return cbs[0]


@pytest.mark.parametrize(
    ("task", "monitor"),
    [("classification", "val/auc_macro"), ("segmentation", "val/auprc")],
)
def test_callbacks_default_is_monitored(task, monitor):
    """`callbacks=default` instantiates a ModelCheckpoint monitoring the per-task validation
    metric (classification `val/auc_macro`, segmentation `val/auprc`).

    `monitor is not None` is what run()'s gate keys off to validate the best checkpoint; this
    guards against regressing to an unmonitored checkpoint.
    """
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=[f"task={task}", "callbacks=default"])

    ckpt = _checkpoint_callback(instantiate(cfg.callbacks))
    assert ckpt.monitor == monitor
    assert ckpt.monitor is not None
    assert ckpt.mode == "max"


def test_callbacks_default_saves_and_validates_best(tmp_path, monkeypatch):
    """A real (non-fast_dev_run) run with `callbacks=default` saves a monitored best
    checkpoint, validates it, and points the W&B summary at that best-checkpoint snapshot.

    Uses the offline W&B logger (LearningRateMonitor needs a logger) pointed at tmp_path, keeps
    checkpointing enabled, and finishes the W&B run in teardown so a later WandbLogger doesn't
    reuse it. `update_summary` is spied (not asserted against a live summary) so the wiring is
    locked deterministically — a short run's best epoch is often the last, which would make a
    value comparison pass even if the summary were never updated."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    summary_calls = []
    monkeypatch.setattr("ivcm.train.update_summary", lambda metrics: summary_calls.append(metrics))
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=classification",
                "callbacks=default",
                f"logger.save_dir={tmp_path}",
                "image_size=32",
                "data.train_dataset.num_samples=4",
                "data.val_dataset.num_samples=4",
                "data.batch_size=2",
                "trainer.max_epochs=2",
                "trainer.accelerator=cpu",
                "trainer.enable_progress_bar=false",
            ],
        )

    try:
        trainer = run(cfg, run_dir=str(tmp_path))

        ckpt_cb = trainer.checkpoint_callback
        assert ckpt_cb is not None
        assert ckpt_cb.monitor == "val/auc_macro"
        assert ckpt_cb.best_model_path  # a best checkpoint was selected and saved
        assert Path(ckpt_cb.best_model_path).exists()
        # The headline metric came from validating that checkpoint.
        assert "val/auc_macro" in trainer.callback_metrics
        # run() pushed the best-checkpoint val/* snapshot into the W&B summary: called once,
        # only with val/* keys, including the monitored metric.
        assert len(summary_calls) == 1
        snapshot = summary_calls[0]
        assert snapshot and all(k.startswith("val/") for k in snapshot)
        assert "val/auc_macro" in snapshot
    finally:
        finish_wandb()


def test_callbacks_none_does_not_update_summary(tmp_path, monkeypatch):
    """Without a monitored best checkpoint (default `callbacks=none`), the final model is
    validated and the W&B summary is left as-is: `update_summary` must not be called.

    Also regression-tests that run() routes Lightning's auto-injected (unmonitored) checkpoint
    under the run dir: no `default_root_dir` override is set here, yet the conftest guard would
    fail this test if a stray ./checkpoints appeared in the repo root."""
    summary_calls = []
    monkeypatch.setattr("ivcm.train.update_summary", lambda metrics: summary_calls.append(metrics))
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=classification",
                "logger=none",
                "image_size=32",
                "data.train_dataset.num_samples=4",
                "data.val_dataset.num_samples=4",
                "data.batch_size=2",
                "trainer.max_epochs=1",
                "trainer.accelerator=cpu",
                "trainer.enable_progress_bar=false",
            ],
        )

    trainer = run(cfg, run_dir=str(tmp_path))
    assert summary_calls == []
    # run() pinned default_root_dir to the run dir, so the auto checkpoint lands under tmp_path.
    assert Path(trainer.default_root_dir) == tmp_path


def test_ckpt_path_is_threaded_to_fit(tmp_path, monkeypatch):
    """`ckpt_path` from config reaches `trainer.fit(ckpt_path=...)` so training can resume.

    `fit`/`validate` are stubbed so the test asserts the wiring (config -> fit) directly and
    fast, without a real resume (Lightning owns the actual restore)."""
    captured = {}

    def fake_fit(self, model, datamodule=None, ckpt_path=None):
        captured["fit_ckpt_path"] = ckpt_path

    def fake_validate(self, model=None, datamodule=None, ckpt_path=None):
        return []

    monkeypatch.setattr(L.Trainer, "fit", fake_fit)
    monkeypatch.setattr(L.Trainer, "validate", fake_validate)

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=classification",
                "logger=none",
                "image_size=32",
                "data.train_dataset.num_samples=4",
                "data.val_dataset.num_samples=4",
                "data.batch_size=2",
                "trainer.accelerator=cpu",
                "ckpt_path=/some/checkpoint.ckpt",
            ],
        )

    run(cfg, run_dir=str(tmp_path))
    assert captured["fit_ckpt_path"] == "/some/checkpoint.ckpt"


def test_ckpt_path_defaults_to_none(tmp_path, monkeypatch):
    """With no override, `ckpt_path` is null -> `fit(ckpt_path=None)` (a fresh run)."""
    captured = {}

    def fake_fit(self, model, datamodule=None, ckpt_path=None):
        captured["fit_ckpt_path"] = ckpt_path

    monkeypatch.setattr(L.Trainer, "fit", fake_fit)
    monkeypatch.setattr(L.Trainer, "validate", lambda *a, **k: [])

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=classification",
                "logger=none",
                "image_size=32",
                "data.train_dataset.num_samples=4",
                "data.val_dataset.num_samples=4",
                "data.batch_size=2",
                "trainer.accelerator=cpu",
            ],
        )

    run(cfg, run_dir=str(tmp_path))
    assert captured["fit_ckpt_path"] is None


def test_train_checkpoint_dirpath_stable_across_resume(tmp_path, monkeypatch):
    """A fresh run checkpoints to <run_dir>/checkpoints; a resumed run continues the *original*
    checkpoint dir (the resumed ckpt's folder), not the new run_dir — so Lightning restores the
    best-checkpoint state instead of seeing a changed dirpath. fit/validate are stubbed so no
    real checkpoint or training is needed."""
    monkeypatch.setattr(L.Trainer, "fit", lambda *a, **k: None)
    monkeypatch.setattr(L.Trainer, "validate", lambda *a, **k: [])

    def _dirpath(overrides, run_dir):
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "task=classification",
                    "callbacks=default",
                    "logger=none",
                    "image_size=32",
                    "data.train_dataset.num_samples=4",
                    "data.val_dataset.num_samples=4",
                    "data.batch_size=2",
                    "trainer.accelerator=cpu",
                    *overrides,
                ],
            )
        return Path(run(cfg, run_dir=str(run_dir)).checkpoint_callback.dirpath)

    fresh_run = tmp_path / "fresh"
    assert _dirpath([], fresh_run) == (fresh_run / "checkpoints").resolve()

    # A checkpoint left by a previous run; resuming gets a *new* run dir but must keep its dir.
    old_ckpt = tmp_path / "old" / "checkpoints" / "last.ckpt"
    old_ckpt.parent.mkdir(parents=True)
    old_ckpt.write_text("ckpt")
    resumed = _dirpath([f"ckpt_path={old_ckpt}"], tmp_path / "new")
    assert resumed == old_ckpt.resolve().parent
    assert resumed != (tmp_path / "new" / "checkpoints").resolve()
