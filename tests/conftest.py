"""Shared pytest fixtures and guards."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from ivcm import register_resolvers

# conftest is imported before any test module, so this registers the `${len:...}` OmegaConf
# resolver ahead of every config composition. Doing it here (rather than via a side-effect
# `import ivcm` in each test) keeps the registration explicit and in one place.
register_resolvers()

REPO_ROOT = Path(__file__).resolve().parents[1]

# Output dirs a stray Trainer/W&B run drops into the repo root (the cwd, since Hydra uses
# `chdir: false`) when a test forgets to route them to tmp_path. A real-fit test must point
# the Trainer's default_root_dir / logger.save_dir at tmp_path, or disable checkpointing — a
# stray ./checkpoints means that routing was forgotten, and 100s of MB pile up unnoticed.
_GUARDED = ("checkpoints", "lightning_logs", "wandb")


@pytest.fixture(autouse=True)
def _no_repo_root_artifacts() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # invoked by pytest, not by name
    """Fail any test that writes Lightning/W&B output into the repo root, and clean it up.

    Only paths newly created during the test are flagged (any pre-existing ones are left
    untouched), so the guard is safe to run unconditionally across the whole suite.
    """
    before = {name for name in _GUARDED if (REPO_ROOT / name).exists()}
    yield
    strays = [name for name in _GUARDED if name not in before and (REPO_ROOT / name).exists()]
    for name in strays:
        shutil.rmtree(REPO_ROOT / name, ignore_errors=True)
    assert not strays, (
        f"test wrote {strays} into the repo root; route Trainer output to tmp_path "
        "(default_root_dir / logger.save_dir) or set trainer.enable_checkpointing=false"
    )
