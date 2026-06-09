"""Hydra entry point for segmentation cross-validation (`cross-validate`).

Runs the binary nerve-segmentation task once per fold (4 training blocks, 1 validation
block), collects each fold's validation sensitivity/specificity/AUC, and reports the
per-fold values plus their mean. Reuses train.py's ``run()`` unchanged.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from rich.console import Console
from rich.table import Table

from ivcm.train import CONFIG_DIR, run
from ivcm.utils.logging import configure_logging

log = logging.getLogger(__name__)

# Validation metric keys logged by SegmentationModule; this is the contract cv depends on.
METRIC_KEYS = ("val/sensitivity", "val/specificity", "val/auc")


def _fold_metrics(callback_metrics, fold: int) -> dict[str, float]:
    """Pull the required metric keys off a fold's callback_metrics, failing loudly."""
    missing = [k for k in METRIC_KEYS if k not in callback_metrics]
    if missing:
        raise KeyError(
            f"fold {fold}: validation metrics {missing} missing from callback_metrics "
            f"(have {sorted(callback_metrics)}). SegmentationModule must log {list(METRIC_KEYS)}."
        )
    return {k: float(callback_metrics[k]) for k in METRIC_KEYS}


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
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="cv")
def main(cfg: DictConfig) -> None:
    cv_dir = Path(HydraConfig.get().runtime.output_dir)
    n_folds = cfg.data.n_folds
    group = f"cv-{cv_dir.name}"

    per_fold: list[dict[str, float]] = []
    for fold in range(n_folds):
        fold_dir = cv_dir / f"fold_{fold}"
        cfg_fold = deepcopy(cfg)
        cfg_fold.data.fold = fold
        with open_dict(cfg_fold):
            cfg_fold.trainer.default_root_dir = str(fold_dir)
            if cfg_fold.get("logger"):
                cfg_fold.logger.save_dir = str(fold_dir)
                cfg_fold.logger.name = f"fold_{fold}"
                cfg_fold.logger.group = group

        trainer = run(cfg_fold, run_dir=str(fold_dir))
        per_fold.append(_fold_metrics(trainer.callback_metrics, fold))
        _finalize_loggers(trainer)

    # Each fold reset the root log handlers to its own file; point logging back at the CV dir.
    configure_logging(str(cv_dir))
    table, text = _summary(per_fold)
    Console().print(table)
    log.info("Cross-validation summary:\n%s", text)


if __name__ == "__main__":
    main()
