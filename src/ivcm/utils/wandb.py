"""Weights & Biases helpers, shared by train.py / cv.py and tests."""

from __future__ import annotations

from collections.abc import Mapping


def update_summary(metrics: Mapping[str, float]) -> None:
    """Overwrite W&B run-summary entries with ``metrics``.

    A run's summary holds each metric's *last logged* value. A post-fit validation on the best
    checkpoint logs at the same ``global_step`` as the final training step, so W&B drops it as
    non-monotonic and the summary keeps the last-epoch values. Writing the summary directly
    bypasses the step logic, letting the summary reflect the best checkpoint. No-op when wandb
    is absent or no run is active.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    for key, value in metrics.items():
        wandb.run.summary[key] = value


def finish_wandb() -> None:
    """Close any active W&B run so a later ``WandbLogger`` starts a fresh one.

    ``WandbLogger.finalize`` does not call ``wandb.finish()``, so without this an active run
    is reused by the next logger (making test order matter, and conflating CV folds). No-op
    when wandb is absent or no run is active.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
