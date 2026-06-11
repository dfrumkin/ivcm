"""Hydra entry point for training (classification or segmentation)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf, open_dict

from ivcm.utils import configure_logging, stable_wandb_id, update_summary

log = logging.getLogger(__name__)

# Absolute path so the installed `train` console script finds the repo-root configs.
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _wandb_anchor_dir(cfg: DictConfig, run_dir: str | None) -> Path | None:
    """The checkpoint directory a run's W&B identity is anchored to, or ``None`` when there is
    no stable anchor to tag.

    On resume (``ckpt_path`` is a real file) it is the resumed checkpoint's own directory,
    **regardless of the callbacks preset** — ``ckpt_path`` is a root-level resume option, so a
    default ``callbacks=none`` resume of a ``callbacks=default`` checkpoint must still continue
    that run. On a fresh run it is ``<run_dir>/checkpoints``, but only with ``callbacks`` set
    (where train.run pins a ModelCheckpoint there); under ``callbacks=none`` Lightning's
    auto-injected checkpoint lands at its own unstable location, so there is no anchor to tag.
    """
    ckpt_path = cfg.get("ckpt_path")
    resuming = bool(ckpt_path and Path(ckpt_path).is_file())
    if resuming:
        ckpt_dir = Path(ckpt_path).resolve().parent
    elif run_dir is not None:
        ckpt_dir = (Path(run_dir) / "checkpoints").resolve()
    else:
        ckpt_dir = None
    if ckpt_dir is None:
        return None
    return ckpt_dir if (resuming or cfg.get("callbacks")) else None


def _pin_wandb_id(cfg: DictConfig, wandb_dir: Path | None) -> None:
    """Bind a ``WandbLogger`` config to a stable run id anchored at ``wandb_dir`` so a resume
    from that directory continues the same W&B run.

    No-op when there is no anchor, the logger is absent or not a ``WandbLogger``, or an id is
    already set (cv.py pins its own per-fold id before calling ``run()`` — that always wins).
    """
    if wandb_dir is None:
        return
    logger = cfg.get("logger")
    if not logger or not str(logger.get("_target_", "")).endswith("WandbLogger"):
        return
    if logger.get("id"):
        return
    with open_dict(cfg):
        cfg.logger.id = stable_wandb_id(wandb_dir)
        cfg.logger.resume = "allow"


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

    # Anchor a stable W&B run id to the checkpoint dir *before* building the logger (the id must
    # be in the config the logger reads), so a resume from that dir continues the same W&B run.
    wandb_dir = _wandb_anchor_dir(cfg, run_dir)
    _pin_wandb_id(cfg, wandb_dir)

    # `cfg.logger` absent/empty (e.g. `logger=none` or `~logger`) -> logger=False, which
    # fully disables Lightning logging (rather than falling back to the default logger).
    logger = instantiate(cfg.logger) if cfg.get("logger") else False
    # `cfg.callbacks` absent/null (default `callbacks=none`) -> no explicit callbacks.
    callbacks = instantiate(cfg.callbacks) if cfg.get("callbacks") else None
    # Pin ModelCheckpoint to the same stable dir so Lightning restores its best-state on resume
    # (it skips that restore when dirpath changes). `wandb_dir` is the resumed checkpoint's own
    # parent when resuming (so a fresh Hydra run_dir doesn't break best-across-history), else
    # <run_dir>/checkpoints; resolved so stored/current dirpath strings match regardless of
    # relative/absolute paths or symlinks. With callbacks set it equals that checkpoint dir, and
    # is only None here when there is no run_dir.
    ckpt_path = cfg.get("ckpt_path")
    if callbacks and wandb_dir is not None:
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath is None:
                cb.dirpath = str(wandb_dir)
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
    if isinstance(ckpt_cb, ModelCheckpoint) and ckpt_cb.monitor is not None:
        # Lightning sets ModelCheckpoint.best_model_path internally without a type annotation,
        # so pyright reads its access as Unknown here.
        use_best = bool(ckpt_cb.best_model_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    else:
        use_best = False
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
