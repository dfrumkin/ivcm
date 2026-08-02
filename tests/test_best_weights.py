"""Tests for the weights-only best-checkpoint helpers (load / write) and `best_epoch` semantics.

Guards the contracts the UI depends on: the produced file is weights-only (no optimizer state),
CPU-portable (loads on MPS/CPU), loadable under PyTorch's `weights_only=True` default, and
strict-loadable into a freshly built module; and `best_epoch` agrees across the checkpoint's stored
epoch, the `best-epoch<NN>.ckpt` filename, and the value written to the W&B summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from torch import nn

from ivcm.train import run
from ivcm.utils import finish_wandb
from ivcm.utils.wandb import load_best_checkpoint, log_best_weights_artifact, write_best_weights_file

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _classification_model() -> nn.Module:
    """A real ClassificationModule (small input) built exactly as `train.run()` builds it."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=["task=classification", "image_size=32"])
    return cast(
        nn.Module,
        instantiate(
            cfg.model,
            criterion=instantiate(cfg.loss),
            optimizer=instantiate(cfg.optimizer),
            scheduler=None,
        ),
    )


def test_load_and_write_weights_file_contract(tmp_path: Path) -> None:
    """Load a full Lightning-style checkpoint, then write the weights-only file, and assert:
    weights-only (no optimizer state), CPU tensors, `weights_only=True`-loadable, strict-loadable."""
    model = _classification_model()
    state_dict = model.state_dict()
    # A full Lightning checkpoint carries far more than tensors (optimizer/loop state); the load
    # step must read it with weights_only=False and keep only the model weights + epoch.
    full = {
        "state_dict": state_dict,
        "epoch": 7,
        "global_step": 100,
        "optimizer_states": [{"step": 100}],
        "lr_schedulers": [{"last_epoch": 7}],
    }
    full_path = tmp_path / "best-epoch07.ckpt"
    torch.save(full, full_path)

    loaded_sd, epoch = load_best_checkpoint(full_path)
    assert epoch == 7
    assert set(loaded_sd) == set(state_dict)
    assert all(t.device.type == "cpu" for t in loaded_sd.values())

    weights_path = tmp_path / "best-weights.ckpt"
    write_best_weights_file(loaded_sd, epoch, weights_path)

    # Minimal file: only tensors + an int, so it loads under the newer weights_only=True default.
    minimal = torch.load(weights_path, weights_only=True)
    assert set(minimal) == {"state_dict", "epoch"}
    assert minimal["epoch"] == 7
    assert "optimizer_states" not in minimal
    assert all(t.device.type == "cpu" for t in minimal["state_dict"].values())

    # Drop-in for the UI: strict load into a freshly built module of the same config.
    fresh = _classification_model()
    fresh.load_state_dict(minimal["state_dict"])


def test_log_best_weights_artifact_builds_expected_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The logged artifact has the UI-contract shape: name `weights-<run id>`, type `model`,
    metadata with best_epoch/monitor/score, and entries at `best-weights.ckpt` and
    `.hydra/config.yaml` (the relative layout a downloaded artifact must preserve). Uses a fake
    `wandb.run`/`wandb.Artifact` so no real upload happens."""
    logged: list[object] = []

    class _FakeArtifact:
        def __init__(self, name: str, metadata: dict[str, object], **kwargs: object) -> None:
            self.name = name
            self.metadata = metadata
            self.artifact_type = kwargs.get("type")
            self.files: dict[str, str] = {}

        def add_file(self, local_path: str, name: str) -> None:
            self.files[name] = local_path

    class _FakeRun:
        id = "abc123"

        def log_artifact(self, artifact: object) -> None:
            logged.append(artifact)

    monkeypatch.setattr(wandb, "run", _FakeRun())
    monkeypatch.setattr(wandb, "Artifact", _FakeArtifact)

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=["task=classification", "image_size=32"])
    state_dict = {"w": torch.zeros(2)}
    log_best_weights_artifact(cfg, state_dict, 5, tmp_path, monitor="val/auc_macro", score=0.9)

    assert len(logged) == 1
    art = cast(_FakeArtifact, logged[0])
    assert art.name == "weights-abc123"
    assert art.artifact_type == "model"
    assert art.metadata == {"best_epoch": 5, "monitor": "val/auc_macro", "score": 0.9}
    # The two entries the UI loader depends on, at the exact relative paths.
    assert set(art.files) == {"best-weights.ckpt", ".hydra/config.yaml"}
    assert art.files["best-weights.ckpt"] == str(tmp_path / "best-weights.ckpt")
    assert (tmp_path / "best-weights.ckpt").exists()  # weights staged locally
    assert (tmp_path / "best-weights-config.yaml").exists()  # config staged locally


def test_best_epoch_agrees_across_checkpoint_filename_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`best_epoch` is consistent three ways — the checkpoint's stored epoch, the `best-epoch<NN>`
    filename, and the value handed to both the summary write and the artifact helper — so no
    zero/one-based offset creeps in."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    summary_calls: list[dict[str, float | int]] = []
    monkeypatch.setattr("ivcm.train.update_summary", summary_calls.append)
    artifact_epochs: list[int] = []

    def _spy(
        cfg: DictConfig,
        state_dict: object,
        best_epoch: int,
        dest_dir: object,
        monitor: str | None = None,
        score: float | None = None,
    ) -> None:
        artifact_epochs.append(best_epoch)

    monkeypatch.setattr("ivcm.train.log_best_weights_artifact", _spy)

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
        best_path = cast(str, ckpt_cb.best_model_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        file_epoch = int(Path(best_path).stem.split("epoch")[-1])
        _, ckpt_epoch = load_best_checkpoint(best_path)
        epoch_write = next(s for s in summary_calls if "best_epoch" in s)
        summary_epoch = epoch_write["best_epoch"]

        assert len(artifact_epochs) == 1
        assert file_epoch == ckpt_epoch == summary_epoch == artifact_epochs[0]
    finally:
        finish_wandb()
