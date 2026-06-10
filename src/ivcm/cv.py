"""Hydra entry point for segmentation cross-validation (`cross-validate`).

Runs the binary nerve-segmentation task once per fold (4 training blocks, 1 validation
block), collects each fold's validation Dice/IoU, AUPRC/AUROC, and sensitivity/specificity, and
reports the per-fold values plus their mean. Reuses train.py's ``run()`` unchanged.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from rich.console import Console
from rich.table import Table

from ivcm.train import CONFIG_DIR, run
from ivcm.utils import configure_logging, finish_wandb

log = logging.getLogger(__name__)

# Validation metric keys logged by SegmentationModule; this is the contract cv depends on.
# Dice/IoU (overlap) and AUPRC/AUROC (ranking; AUPRC is the best-checkpoint selection metric)
# lead, then the pixel-level sensitivity/specificity.
METRIC_KEYS = (
    "val/dice",
    "val/iou",
    "val/auprc",
    "val/auc",
    "val/sensitivity",
    "val/specificity",
)


def _fold_metrics(callback_metrics, fold: int) -> dict[str, float]:
    """Pull the required metric keys off a fold's callback_metrics, failing loudly."""
    missing = [k for k in METRIC_KEYS if k not in callback_metrics]
    if missing:
        raise KeyError(
            f"fold {fold}: validation metrics {missing} missing from callback_metrics "
            f"(have {sorted(callback_metrics)}). SegmentationModule must log {list(METRIC_KEYS)}."
        )
    return {k: float(callback_metrics[k]) for k in METRIC_KEYS}


def _metrics_path(fold_dir: Path) -> Path:
    return fold_dir / "metrics.json"


def _completed_metrics(fold_dir: Path) -> dict[str, float] | None:
    """Return a fold's persisted metrics if it finished cleanly, else None.

    A `metrics.json` with all METRIC_KEYS marks a completed fold; anything missing/corrupt is
    treated as not-done so the fold re-runs (resuming from a checkpoint if one exists)."""
    path = _metrics_path(fold_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not all(k in data for k in METRIC_KEYS):
        return None
    return {k: float(data[k]) for k in METRIC_KEYS}


def _save_fold_metrics(fold_dir: Path, metrics: dict[str, float]) -> None:
    """Persist a finished fold's metrics so a later re-run skips it instead of retraining."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    _metrics_path(fold_dir).write_text(json.dumps(metrics, indent=2))


def _find_resume_ckpt(fold_dir: Path) -> str | None:
    """The fold's `last.ckpt` (written by callbacks=default's save_last to the stable
    <fold_dir>/checkpoints set in train.run()), or None. Used to resume a fold interrupted
    before it persisted metrics; the stable path also carries ModelCheckpoint best-state."""
    ckpt = fold_dir / "checkpoints" / "last.ckpt"
    return str(ckpt) if ckpt.exists() else None


def _fold_wandb_id(fold_dir: Path) -> str:
    """Stable W&B run id for a fold, persisted so a resumed fold continues the same run
    instead of starting a new one. Generated once on the first attempt, reloaded thereafter."""
    path = fold_dir / "wandb_run_id.txt"
    if path.exists():
        return path.read_text().strip()
    import wandb

    run_id = wandb.util.generate_id()
    fold_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id)
    return run_id


def _summary(per_fold: list[dict[str, float]]) -> tuple[Table, str]:
    """Build a Rich table and a plain-text rendering of the per-fold + mean metrics."""
    short = [k.split("/")[-1] for k in METRIC_KEYS]  # sensitivity, specificity, auc
    means = {k: sum(m[k] for m in per_fold) / len(per_fold) for k in METRIC_KEYS}

    table = Table(title="Segmentation cross-validation")
    table.add_column("fold")
    for name in short:
        table.add_column(name, justify="right")
    lines = ["Segmentation cross-validation", "fold  " + "  ".join(f"{n:>11}" for n in short)]
    for i, m in enumerate(per_fold):
        table.add_row(str(i), *(f"{m[k]:.4f}" for k in METRIC_KEYS))
        lines.append(f"{i:<5} " + "  ".join(f"{m[k]:>11.4f}" for k in METRIC_KEYS))
    table.add_row("mean", *(f"{means[k]:.4f}" for k in METRIC_KEYS), style="bold")
    lines.append(f"{'mean':<5} " + "  ".join(f"{means[k]:>11.4f}" for k in METRIC_KEYS))
    return table, "\n".join(lines)


def _finalize_loggers(trainer) -> None:
    """Tolerantly finalize whatever loggers the trainer has, then close any live W&B run."""
    for lg in trainer.loggers:
        lg.finalize("success")
    finish_wandb()


def _reject_resume(cfg: DictConfig) -> None:
    """Reject a user-supplied global ckpt_path: it cannot map to the per-fold trainings.

    CV resume is per-fold instead (see the fold loop): re-run pointing Hydra at the prior
    output dir (`hydra.run.dir=...`) and each fold skips/resumes/runs based on its own state."""
    if cfg.get("ckpt_path"):
        raise ValueError(
            f"cross-validate does not accept a global ckpt_path (ckpt_path={cfg.ckpt_path!r}); a "
            "single checkpoint cannot map to the per-fold runs. To resume an interrupted CV, "
            "re-run with hydra.run.dir=<previous output dir> — completed folds are skipped and "
            "an interrupted fold resumes from its last.ckpt."
        )


def _prepare_fold_cfg(cfg: DictConfig, fold: int, fold_dir: Path, group: str) -> DictConfig:
    """Build a fold's config: select the fold, route outputs to ``fold_dir``, resume from the
    fold's ``last.ckpt`` if present, and continue its W&B run (stable per-fold id)."""
    cfg_fold = deepcopy(cfg)
    cfg_fold.data.fold = fold
    with open_dict(cfg_fold):
        cfg_fold.trainer.default_root_dir = str(fold_dir)
        resume_ckpt = _find_resume_ckpt(fold_dir)
        if resume_ckpt is not None:
            # A checkpoint only exists from a callbacks=default run; resuming without it would
            # restore weights but drop monitored best-checkpoint tracking and persist
            # final-model metrics — inconsistent with the already-completed folds. Fail loudly.
            if not cfg_fold.get("callbacks"):
                raise ValueError(
                    f"fold {fold} has a checkpoint to resume ({resume_ckpt}) but `callbacks` is "
                    "unset; re-run with callbacks=default so best-checkpoint tracking continues."
                )
            cfg_fold.ckpt_path = resume_ckpt
            log.info("Resuming fold %d from %s", fold, resume_ckpt)
        if cfg_fold.get("logger"):
            cfg_fold.logger.save_dir = str(fold_dir)
            cfg_fold.logger.name = f"fold_{fold}"
            cfg_fold.logger.group = group
            # Continue the same W&B run on resume (stable per-fold id) instead of a new one.
            if str(cfg_fold.logger.get("_target_", "")).endswith("WandbLogger"):
                cfg_fold.logger.id = _fold_wandb_id(fold_dir)
                cfg_fold.logger.resume = "allow"
    return cfg_fold


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="cv")
def main(cfg: DictConfig) -> None:
    _reject_resume(cfg)

    cv_dir = Path(HydraConfig.get().runtime.output_dir)
    n_folds = cfg.data.n_folds
    group = f"cv-{cv_dir.name}"

    per_fold: list[dict[str, float]] = []
    for fold in range(n_folds):
        fold_dir = cv_dir / f"fold_{fold}"

        # Idempotent resume: a fold that already persisted metrics is reused as-is; one with
        # only a checkpoint (interrupted before finishing) resumes from it; otherwise fresh.
        completed = _completed_metrics(fold_dir)
        if completed is not None:
            log.info("Fold %d already complete; reusing saved metrics.", fold)
            per_fold.append(completed)
            continue

        cfg_fold = _prepare_fold_cfg(cfg, fold, fold_dir, group)
        trainer = run(cfg_fold, run_dir=str(fold_dir))
        metrics = _fold_metrics(trainer.callback_metrics, fold)
        _save_fold_metrics(fold_dir, metrics)
        per_fold.append(metrics)
        _finalize_loggers(trainer)

    # Each fold reset the root log handlers to its own file; point logging back at the CV dir.
    configure_logging(str(cv_dir))
    table, text = _summary(per_fold)
    Console().print(table)
    log.info("Cross-validation summary:\n%s", text)


if __name__ == "__main__":
    main()
