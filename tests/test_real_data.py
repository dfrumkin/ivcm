from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str(REPO / "configs")
DATA_DIR = REPO / "data"

# The proprietary data is gitignored and absent in CI; these tests only run where it exists.
pytestmark = pytest.mark.skipif(not DATA_DIR.exists(), reason="proprietary data/ not present")


def _compose(*overrides: str) -> DictConfig:
    # Pin data_dir to an absolute path so the test doesn't depend on the pytest working
    # directory (the config default is the relative ${oc.env:IVCM_DATA_DIR,data}); small
    # image_size (classification) and crop_size (segmentation sliding-window size) keep the smoke
    # runs fast. Segmentation validation runs at the real slices' native resolution regardless.
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(
            config_name="train",
            overrides=[
                f"data_dir={DATA_DIR}",
                "image_size=64",
                "crop_size=32",
                "data.batch_size=2",
                *overrides,
            ],
        )


def _build_model(cfg: DictConfig) -> L.LightningModule:
    return instantiate(
        cfg.model,
        criterion=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=instantiate(cfg.scheduler),
    )


def _trainer() -> L.Trainer:
    return L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def test_classification_real() -> None:
    cfg = _compose("task=classification", "data=classification_real")
    datamodule = instantiate(cfg.data)
    assert datamodule.class_names == ["endothelium", "epithelium", "nerves", "stroma"]

    trainer = _trainer()
    trainer.fit(_build_model(cfg), datamodule=datamodule)
    loss = trainer.callback_metrics.get("train/loss_step", trainer.callback_metrics.get("train/loss"))
    assert loss is not None and torch.isfinite(loss)


def test_segmentation_real_cv_split() -> None:
    cfg = _compose("task=segmentation", "data=segmentation_real")
    datamodule = instantiate(cfg.data)
    datamodule.setup()

    train_idx = set(datamodule.train_indices.tolist())
    val_idx = set(datamodule.val_indices.tolist())
    assert train_idx.isdisjoint(val_idx)
    # Train + val together cover every slice exactly once (0..n-1).
    union = train_idx | val_idx
    assert union == set(range(len(union)))

    trainer = _trainer()
    trainer.fit(_build_model(cfg), datamodule=datamodule)
    loss = trainer.callback_metrics.get("train/loss_step", trainer.callback_metrics.get("train/loss"))
    assert loss is not None and torch.isfinite(loss)
