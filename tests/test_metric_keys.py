from pathlib import Path

import lightning as L
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from ivcm.cv import METRIC_KEYS

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_segmentation_logs_metric_keys():
    """SegmentationModule must log the exact val keys cv.py reads from callback_metrics.

    Uses a normal 1-epoch run with a single train/val batch and no sanity validation (not
    fast_dev_run, which populates callback_metrics unpredictably) so the asserted keys come
    from a real validation epoch.
    """
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=segmentation",
                "image_size=32",
                "data.train_dataset.num_samples=4",
                "data.val_dataset.num_samples=4",
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
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=datamodule)

    for key in METRIC_KEYS:
        assert key in trainer.callback_metrics, key
        assert torch.isfinite(trainer.callback_metrics[key])
