"""Weights & Biases helpers, shared by train.py / cv.py and tests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def stable_wandb_id(directory: Path) -> str:
    """A W&B run id persisted under ``directory``, generated once and reloaded thereafter.

    Lets a resumed run continue the same W&B run instead of starting a new one: the id is bound
    to a checkpoint directory, so the run whose checkpoints live there keeps its W&B identity
    across resumes. Generated on first use (``wandb.util.generate_id``) and reused on every
    later call for the same directory.
    """
    path = directory / "wandb_run_id.txt"
    if path.exists():
        return path.read_text().strip()
    from wandb.util import generate_id

    run_id = generate_id()
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id)
    return run_id


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
