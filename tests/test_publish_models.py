"""Non-network test of the models publisher's pure staging helper (no ``gh``, no torch)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ivcm.app.config import ModelConfig

# ``scripts/`` is not an installed package; make it importable for this test (pyright resolves it
# via ``tool.pyright.extraPaths``).
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import publish_models  # noqa: E402


def _make_run(run_dir: Path, marker: bytes) -> ModelConfig:
    """A fake run directory holding a checkpoint + Hydra config, and its ``ModelConfig``."""
    (run_dir / ".hydra").mkdir(parents=True)
    (run_dir / "best-weights.ckpt").write_bytes(marker)
    (run_dir / ".hydra" / "config.yaml").write_text("class_names: [x]\n", encoding="utf-8")
    return ModelConfig(run_dir=str(run_dir), ckpt=str(run_dir / "best-weights.ckpt"))


def test_stage_models_produces_fixed_layout(tmp_path: Path) -> None:
    models = [
        ("cls", _make_run(tmp_path / "src" / "cls", b"ckpt-cls")),
        ("seg", _make_run(tmp_path / "src" / "seg", b"ckpt-seg")),
    ]
    dest = tmp_path / "staged"

    publish_models.stage_models(models, dest)

    # Exactly the fixed layout the frozen loader and the workflow expect.
    assert (dest / "cls" / "best-weights.ckpt").read_bytes() == b"ckpt-cls"
    assert (dest / "cls" / ".hydra" / "config.yaml").is_file()
    assert (dest / "seg" / "best-weights.ckpt").read_bytes() == b"ckpt-seg"
    assert (dest / "seg" / ".hydra" / "config.yaml").is_file()


def test_stage_models_missing_source_raises(tmp_path: Path) -> None:
    run = tmp_path / "cls"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    # No best-weights.ckpt on disk -> a clear failure before anything is staged/published.
    model = ModelConfig(run_dir=str(run), ckpt=str(run / "best-weights.ckpt"))

    with pytest.raises(FileNotFoundError):
        publish_models.stage_models([("cls", model)], tmp_path / "staged")
