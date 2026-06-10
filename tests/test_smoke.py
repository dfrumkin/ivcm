from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

import ivcm
from ivcm.train import run

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# Small, CPU-only overrides shared across the training tests. image_size is a root field;
# sample counts live on the synthetic datasets (data.{train,val}_dataset.num_samples).
TINY = [
    "image_size=32",
    "data.train_dataset.num_samples=4",
    "data.val_dataset.num_samples=4",
    "data.batch_size=2",
    "trainer.accelerator=cpu",
    "trainer.enable_progress_bar=false",
    "+trainer.enable_checkpointing=false",
]


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(config_name="train", overrides=[*TINY, *overrides])


def test_version_is_string() -> None:
    assert isinstance(ivcm.__version__, str)


@pytest.mark.parametrize("task", ["classification", "segmentation"])
def test_fast_dev_run(task: str) -> None:
    """Compose each task's config and run a single fast_dev_run train+val step."""
    cfg = _compose(f"task={task}")

    datamodule = instantiate(cfg.data)
    model = instantiate(
        cfg.model,
        criterion=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=instantiate(cfg.scheduler),
    )

    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)

    loss = trainer.callback_metrics.get("train/loss_step", trainer.callback_metrics.get("train/loss"))
    assert loss is not None
    assert torch.isfinite(loss)


def test_cli_fast_dev_run_does_not_init_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented `+trainer.fast_dev_run=true` path runs end-to-end via the real
    entrypoint and, with the default W&B logger configured, does not initialize wandb."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = _compose(
        "task=classification",
        "+trainer.fast_dev_run=true",
        f"logger.save_dir={tmp_path}",  # if wandb *did* init, it would write here
    )

    trainer = run(cfg, run_dir=str(tmp_path))

    assert trainer.global_step >= 1  # training actually ran
    # fast_dev_run suppresses logging: no wandb run directory should have been created.
    assert not (tmp_path / "wandb").exists()


def test_logger_none_disables_logging(tmp_path: Path) -> None:
    """`logger=none` fully disables Lightning logging (no fallback logger)."""
    cfg = _compose("task=classification", "logger=none", "trainer.max_epochs=1")

    trainer = run(cfg, run_dir=str(tmp_path))

    assert trainer.loggers == []
