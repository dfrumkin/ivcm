"""Weights & Biases helpers, shared by train.py / cv.py and tests."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

log = logging.getLogger(__name__)


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


def update_summary(metrics: Mapping[str, float | int]) -> None:
    """Overwrite W&B run-summary entries with ``metrics``.

    A run's summary holds each metric's *last logged* value. A post-fit validation on the best
    checkpoint logs at the same ``global_step`` as the final training step, so W&B drops it as
    non-monotonic and the summary keeps the last-epoch values. Writing the summary directly
    bypasses the step logic, letting the summary reflect the best checkpoint. No-op when wandb
    is absent or no run is active.

    Accepts ``int`` values (e.g. ``best_epoch``) alongside floats.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    for key, value in metrics.items():
        wandb.run.summary[key] = value


def wandb_run_active() -> bool:
    """True when a W&B run is currently active.

    The gate for the best-weights-artifact work in ``train.run()``: `best_epoch` and the artifact
    are produced only when a run is active, so no checkpoint is read and nothing is staged under
    ``logger=none`` (or when wandb is absent). Mirrors the ``wandb.run``-only contract of
    ``update_summary``/``finish_wandb``.
    """
    try:
        import wandb
    except ImportError:
        return False
    return wandb.run is not None


def load_best_checkpoint(ckpt_path: str | Path) -> tuple[dict[str, torch.Tensor], int]:
    """Load a full Lightning checkpoint once, returning its CPU ``state_dict`` and stored epoch.

    ``weights_only=False`` is required: this reads a *full* Lightning checkpoint (pickled
    hyperparameters/loop state, not just tensors), which recent PyTorch's ``weights_only=True``
    default would reject. ``map_location="cpu"`` strips CUDA device references so the derived
    weights file loads on the UI machine's MPS/CPU. Pure (no W&B); it writes nothing, so it cannot
    leave partial files.
    """
    ckpt = cast("dict[str, object]", torch.load(ckpt_path, map_location="cpu", weights_only=False))
    state_dict = cast("dict[str, torch.Tensor]", ckpt["state_dict"])
    epoch = int(cast(int, ckpt["epoch"]))
    return state_dict, epoch


def write_best_weights_file(state_dict: Mapping[str, torch.Tensor], epoch: int, dest: Path) -> None:
    """Write a minimal weights-only file the UI can load directly.

    Contains only the model ``state_dict`` (no optimizer/scheduler/loop state) plus a provenance
    ``epoch`` the UI ignores. Kept minimal (tensors + int) so it stays loadable under PyTorch's
    ``weights_only=True`` default — unlike the full Lightning checkpoint — keeping the UI loader
    working across torch versions. Pure (no W&B run required), so it is exercised directly in tests.
    """
    torch.save({"state_dict": dict(state_dict), "epoch": epoch}, dest)


def log_best_weights_artifact(
    cfg: DictConfig,
    state_dict: Mapping[str, torch.Tensor],
    best_epoch: int,
    dest_dir: str | Path,
    monitor: str | None = None,
    score: float | None = None,
) -> None:
    """Stage and log the weights-only best-checkpoint artifact (one per W&B run).

    Writes ``best-weights.ckpt`` and the run's config into ``dest_dir`` (the stable checkpoint
    directory, not a tempdir — avoids any staging/cleanup race and leaves a local weights copy),
    then logs a ``wandb.Artifact`` named ``weights-<run id>`` (``type="model"``) with the weights at
    ``best-weights.ckpt`` and the config at ``.hydra/config.yaml`` (so a downloaded artifact
    directory is drop-in for the UI's ``run_dir``/``ckpt``). No-op when no W&B run is active.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    dest = Path(dest_dir)
    weights_path = dest / "best-weights.ckpt"
    write_best_weights_file(state_dict, best_epoch, weights_path)
    # Stage the run config locally (any name); it is added to the artifact at the .hydra path the UI
    # reads. resolve=False preserves Hydra-only interpolations exactly as the saved .hydra/config.yaml
    # does (the UI resolves only model/loss/optimizer/size, with class_names at config root).
    config_path = dest / "best-weights-config.yaml"
    OmegaConf.save(cfg, config_path, resolve=False)

    metadata: dict[str, object] = {"best_epoch": best_epoch}
    if monitor is not None:
        metadata["monitor"] = monitor
    if score is not None:
        metadata["score"] = score

    artifact = wandb.Artifact(name=f"weights-{wandb.run.id}", type="model", metadata=metadata)
    artifact.add_file(str(weights_path), name="best-weights.ckpt")
    artifact.add_file(str(config_path), name=".hydra/config.yaml")
    wandb.run.log_artifact(artifact)


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
