"""Hydra entry point for training (classification or segmentation)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ivcm.utils.logging import configure_logging

log = logging.getLogger(__name__)

# Absolute path so the installed `train` console script finds the repo-root configs.
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def run(cfg: DictConfig, run_dir: str | None = None) -> L.Trainer:
    """Build everything from ``cfg`` and run fit + validate.

    Split out from :func:`main` so the full training path is testable without the
    Hydra CLI machinery. Returns the trainer for inspection.
    """
    configure_logging(run_dir)
    L.seed_everything(cfg.seed, workers=True)

    log.info("Run directory: %s", run_dir)
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    datamodule = instantiate(cfg.data)
    scheduler = instantiate(cfg.scheduler) if cfg.get("scheduler") else None
    model = instantiate(
        cfg.model,
        criterion=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=scheduler,
    )

    # `cfg.logger` absent/empty (e.g. `logger=none` or `~logger`) -> logger=False, which
    # fully disables Lightning logging (rather than falling back to the default logger).
    logger = instantiate(cfg.logger) if cfg.get("logger") else False
    trainer: L.Trainer = instantiate(cfg.trainer, logger=logger)

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    return trainer


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg, run_dir=HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
