"""Tests for the `callbacks=default` group: the monitored best-checkpoint callback and the
best-checkpoint validation path it enables in `train.run()`."""

from pathlib import Path
from typing import cast

import lightning as L
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from ivcm.train import (
    _save_best_weights_artifact,  # pyright: ignore[reportPrivateUsage]
    run,
)
from ivcm.utils import finish_wandb

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _noop(*args: object, **kwargs: object) -> None:
    """Stub for ``Trainer.fit`` (monkeypatched) — accepts any args, does nothing."""
    return None


def _noop_validate(*args: object, **kwargs: object) -> list[object]:
    """Stub for ``Trainer.validate`` (monkeypatched) — returns an empty metrics list."""
    return []


def _checkpoint_callback(callbacks: list[L.Callback]) -> ModelCheckpoint:
    cbs = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
    assert len(cbs) == 1, f"expected exactly one ModelCheckpoint, got {cbs}"
    return cbs[0]


@pytest.mark.parametrize(
    ("task", "monitor"),
    [("classification", "val/auc_macro"), ("segmentation", "val/auprc")],
)
def test_callbacks_default_is_monitored(task: str, monitor: str) -> None:
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


def test_callbacks_default_saves_and_validates_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real (non-fast_dev_run) run with `callbacks=default` saves a monitored best
    checkpoint, validates it, and points the W&B summary at that best-checkpoint snapshot.

    Uses the offline W&B logger (LearningRateMonitor needs a logger) pointed at tmp_path, keeps
    checkpointing enabled, and finishes the W&B run in teardown so a later WandbLogger doesn't
    reuse it. `update_summary` is spied (not asserted against a live summary) so the wiring is
    locked deterministically — a short run's best epoch is often the last, which would make a
    value comparison pass even if the summary were never updated."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    summary_calls: list[dict[str, float | int]] = []
    monkeypatch.setattr("ivcm.train.update_summary", summary_calls.append)
    # Spy the artifact upload so the wiring is asserted without staging/logging a real artifact.
    artifact_calls: list[object] = []

    def _spy_artifact(*args: object, **kwargs: object) -> None:
        artifact_calls.append((args, kwargs))

    monkeypatch.setattr("ivcm.train.log_best_weights_artifact", _spy_artifact)
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
        assert isinstance(ckpt_cb, ModelCheckpoint)
        assert ckpt_cb.monitor == "val/auc_macro"
        # Lightning leaves best_model_path partially untyped (set internally without an
        # annotation); cast to str so the two assertions below stay type-clean.
        best_model_path = cast(str, ckpt_cb.best_model_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        assert best_model_path  # a best checkpoint was selected and saved
        assert Path(best_model_path).exists()
        # The headline metric came from validating that checkpoint.
        assert "val/auc_macro" in trainer.callback_metrics
        # run() pushed two summary writes: the best-checkpoint val/* snapshot (val/* keys only,
        # including the monitored metric) and the best_epoch marker.
        assert len(summary_calls) == 2
        val_snapshot = next(s for s in summary_calls if all(k.startswith("val/") for k in s))
        assert "val/auc_macro" in val_snapshot
        epoch_write = next(s for s in summary_calls if "best_epoch" in s)
        assert set(epoch_write) == {"best_epoch"} and isinstance(epoch_write["best_epoch"], int)
        # The weights-only artifact upload was invoked exactly once (under the active W&B run).
        assert len(artifact_calls) == 1
    finally:
        finish_wandb()


def test_callbacks_none_does_not_update_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a monitored best checkpoint (default `callbacks=none`), the final model is
    validated and the W&B summary is left as-is: `update_summary` must not be called.

    Also regression-tests that run() routes Lightning's auto-injected (unmonitored) checkpoint
    under the run dir: no `default_root_dir` override is set here, yet the conftest guard would
    fail this test if a stray ./checkpoints appeared in the repo root."""
    summary_calls: list[dict[str, float]] = []
    monkeypatch.setattr("ivcm.train.update_summary", summary_calls.append)
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


def test_save_best_weights_artifact_gated_on_active_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an active W&B run, the best-weights step is a full no-op: no checkpoint read, no
    `best_epoch` write, no artifact staging/upload (the step-2 gate)."""
    finish_wandb()  # ensure no run is active
    load_calls: list[object] = []
    log_calls: list[object] = []
    summary_calls: list[object] = []

    def _spy_load(*a: object, **k: object) -> tuple[dict[str, object], int]:
        load_calls.append((a, k))
        return {}, 0

    def _spy_log(*a: object, **k: object) -> None:
        log_calls.append((a, k))

    def _spy_summary(*a: object, **k: object) -> None:
        summary_calls.append((a, k))

    monkeypatch.setattr("ivcm.train.load_best_checkpoint", _spy_load)
    monkeypatch.setattr("ivcm.train.log_best_weights_artifact", _spy_log)
    monkeypatch.setattr("ivcm.train.update_summary", _spy_summary)

    cfg = OmegaConf.create({})
    ckpt_cb = ModelCheckpoint(monitor="val/auc_macro")
    _save_best_weights_artifact(cfg, ckpt_cb)

    assert load_calls == [] and log_calls == [] and summary_calls == []


def test_artifact_upload_failure_does_not_break_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An artifact staging/upload failure only warns — `run()` still completes (so a
    `cross-validate` fold is not lost) and `best_epoch` was already written to the summary."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    summary_calls: list[dict[str, float | int]] = []
    monkeypatch.setattr("ivcm.train.update_summary", summary_calls.append)
    # Spy the module logger directly: run() reconfigures logging, so caplog (root handler) would
    # not see the warning.
    warnings_logged: list[str] = []

    def _spy_warning(msg: object, *a: object, **k: object) -> None:
        warnings_logged.append(str(msg))

    monkeypatch.setattr("ivcm.train.log.warning", _spy_warning)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr("ivcm.train.log_best_weights_artifact", _boom)
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
        assert trainer is not None  # run() did not propagate the artifact failure
        assert any("best_epoch" in s for s in summary_calls)  # best_epoch written before the failure
        assert any("artifact" in w.lower() for w in warnings_logged)
    finally:
        finish_wandb()


def test_ckpt_path_is_threaded_to_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ckpt_path` from config reaches `trainer.fit(ckpt_path=...)` so training can resume.

    `fit`/`validate` are stubbed so the test asserts the wiring (config -> fit) directly and
    fast, without a real resume (Lightning owns the actual restore)."""
    captured: dict[str, str | None] = {}

    def fake_fit(self: L.Trainer, model: object, datamodule: object = None, ckpt_path: str | None = None) -> None:
        captured["fit_ckpt_path"] = ckpt_path

    def fake_validate(
        self: L.Trainer, model: object = None, datamodule: object = None, ckpt_path: str | None = None
    ) -> list[object]:
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


def test_ckpt_path_defaults_to_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override, `ckpt_path` is null -> `fit(ckpt_path=None)` (a fresh run)."""
    captured: dict[str, str | None] = {}

    def fake_fit(self: L.Trainer, model: object, datamodule: object = None, ckpt_path: str | None = None) -> None:
        captured["fit_ckpt_path"] = ckpt_path

    monkeypatch.setattr(L.Trainer, "fit", fake_fit)
    monkeypatch.setattr(L.Trainer, "validate", _noop_validate)

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


def test_train_checkpoint_dirpath_stable_across_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh run checkpoints to <run_dir>/checkpoints; a resumed run continues the *original*
    checkpoint dir (the resumed ckpt's folder), not the new run_dir — so Lightning restores the
    best-checkpoint state instead of seeing a changed dirpath. fit/validate are stubbed so no
    real checkpoint or training is needed."""
    monkeypatch.setattr(L.Trainer, "fit", _noop)
    monkeypatch.setattr(L.Trainer, "validate", _noop_validate)

    def _dirpath(overrides: list[str], run_dir: Path) -> Path:
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
        ckpt_cb = run(cfg, run_dir=str(run_dir)).checkpoint_callback
        assert isinstance(ckpt_cb, ModelCheckpoint) and ckpt_cb.dirpath is not None
        return Path(ckpt_cb.dirpath)

    fresh_run = tmp_path / "fresh"
    assert _dirpath([], fresh_run) == (fresh_run / "checkpoints").resolve()

    # A checkpoint left by a previous run; resuming gets a *new* run dir but must keep its dir.
    old_ckpt = tmp_path / "old" / "checkpoints" / "last.ckpt"
    old_ckpt.parent.mkdir(parents=True)
    old_ckpt.write_text("ckpt")
    resumed = _dirpath([f"ckpt_path={old_ckpt}"], tmp_path / "new")
    assert resumed == old_ckpt.resolve().parent
    assert resumed != (tmp_path / "new" / "checkpoints").resolve()
