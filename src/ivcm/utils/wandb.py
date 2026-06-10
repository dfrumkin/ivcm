"""Weights & Biases teardown helper, shared by cv.py and tests."""

from __future__ import annotations


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
