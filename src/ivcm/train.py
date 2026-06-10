"""Hydra entry point for training (classification or segmentation)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from ivcm.utils import configure_logging, update_summary

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
    # `cfg.callbacks` absent/null (default `callbacks=none`) -> no explicit callbacks.
    callbacks = instantiate(cfg.callbacks) if cfg.get("callbacks") else None
    # Pin ModelCheckpoint to a stable dir so Lightning restores its best-state on resume (it
    # skips that restore when dirpath changes). When resuming, continue the *original*
    # checkpoint dir (the resumed checkpoint's parent) so a fresh Hydra run_dir doesn't break
    # best-across-history; otherwise use <run_dir>/checkpoints. Both resolved so the stored
    # and current dirpath strings match regardless of relative/absolute paths or symlinks.
    ckpt_path = cfg.get("ckpt_path")
    if callbacks:
        if ckpt_path and Path(ckpt_path).is_file():
            ckpt_dir = str(Path(ckpt_path).resolve().parent)
        elif run_dir is not None:
            ckpt_dir = str((Path(run_dir) / "checkpoints").resolve())
        else:
            ckpt_dir = None
        if ckpt_dir is not None:
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
                    cb.dirpath = ckpt_dir
    # Pin default_root_dir to the run dir so Lightning's fallback output location stays under
    # outputs/ instead of the repo-root cwd (Hydra runs with `chdir: false`). This covers the
    # auto-injected *unmonitored* checkpoint Lightning adds when `callbacks != default` and any
    # `lightning_logs`; the `callbacks=default` ModelCheckpoint sets its own dirpath above and
    # is unaffected.
    trainer_kwargs = {"logger": logger, "callbacks": callbacks}
    if run_dir is not None:
        trainer_kwargs["default_root_dir"] = run_dir
    trainer: L.Trainer = instantiate(cfg.trainer, **trainer_kwargs)

    # `ckpt_path` (null by default) resumes training from a checkpoint when set.
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    # Report the best *monitored* checkpoint when one was saved; otherwise the final model.
    # `monitor is not None` restricts this to our `callbacks=default` ModelCheckpoint:
    # Lightning auto-injects an unmonitored checkpoint (monitor=None) that also populates
    # best_model_path, and we must not validate that instead of the final model.
    ckpt_cb = trainer.checkpoint_callback
    use_best = ckpt_cb is not None and ckpt_cb.monitor is not None and bool(ckpt_cb.best_model_path)
    trainer.validate(model, datamodule=datamodule, ckpt_path="best" if use_best else None)
    if use_best:
        # `validate` on the best checkpoint produced a consistent val/* snapshot in
        # callback_metrics, but its W&B log was dropped (same global_step as the final training
        # step -> non-monotonic), leaving the run summary on the last epoch. Point the summary
        # at the best-checkpoint values so it matches the checkpoint that is actually saved.
        best_val = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
        update_summary(best_val)
    return trainer


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg, run_dir=HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
