"""Publish the two trained models as the ``ivcm-models.zip`` GitHub release asset.

One command replaces the manual stage/zip/upload. It reads the two model ``(run_dir, ckpt)`` pairs
straight from the UI's own :data:`ivcm.app.config.CONFIG` (single source of truth, no drift), stages
them into the fixed ``cls/`` + ``seg/`` layout the Windows build expects, zips them, and publishes
the zip to a ``models-*`` GitHub release via the ``gh`` CLI.

Usage::

    uv run python scripts/publish_models.py                  # -> tag models-v1
    uv run python scripts/publish_models.py --tag models-v2  # a new immutable model revision
    uv run python scripts/publish_models.py --force          # replace an unconsumed upload only

Requires the GitHub CLI (``gh``) installed and authenticated once (``gh auth login`` or a
``GH_TOKEN`` env var). Model revisions are immutable: by default the script refuses to overwrite an
already-published ``ivcm-models.zip``. A genuine weights change gets a **new** tag; ``--force``
exists only to correct an upload that no Windows build has consumed yet.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ivcm.app.config import CONFIG, ModelConfig

ASSET_NAME = "ivcm-models.zip"


def _staged_files(model: ModelConfig) -> list[tuple[Path, str]]:
    """``(source path, staged relative name)`` pairs for one model: checkpoint + hydra config."""
    return [
        (model.ckpt_path(), "best-weights.ckpt"),
        (model.config_path(), ".hydra/config.yaml"),
    ]


def stage_models(models: Sequence[tuple[str, ModelConfig]], dest: Path) -> None:
    """Copy each model into ``dest/<name>`` with the fixed release filenames.

    Pure and network-free (no torch, no ``gh``) — the unit-tested core of the publisher. ``models``
    pairs a destination subdirectory name (``"cls"`` / ``"seg"``) with its ``ModelConfig``; each
    model's checkpoint is staged as ``<name>/best-weights.ckpt`` and its saved Hydra config as
    ``<name>/.hydra/config.yaml``. Raises :class:`FileNotFoundError` with a clear message if any
    source file is missing, so a bad asset is never staged.
    """
    for name, model in models:
        for src, rel in _staged_files(model):
            if not src.is_file():
                raise FileNotFoundError(f"model source for '{name}' not found: {src}")
            target = dest / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def _validate_checkpoint(path: Path) -> None:
    """Assert ``path`` is a weights-only checkpoint (a dict with a ``state_dict``) that loads.

    Runs at publish time (not part of the pure ``stage_models`` helper) so a broken or wrong-format
    checkpoint is caught before an immutable release is minted. ``torch`` is imported lazily so the
    pure staging helper stays importable without it.
    """
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ValueError(f"{path} is not a weights-only checkpoint containing a 'state_dict'")


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    """Zip the contents of ``src_dir`` (``cls/``, ``seg/``) to ``zip_path`` at the archive root."""
    base = zip_path.with_suffix("")  # shutil.make_archive appends ".zip"
    shutil.make_archive(str(base), "zip", root_dir=str(src_dir))


def _run(cmd: list[str]) -> None:
    """Run a command, raising on a non-zero exit."""
    subprocess.run(cmd, check=True)


def _current_commit() -> str | None:
    """The current ``HEAD`` commit SHA, or ``None`` if it can't be resolved."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _existing_assets(tag: str) -> list[str] | None:
    """Asset names on ``tag``'s release, or ``None`` if the release does not exist yet."""
    result = subprocess.run(
        ["gh", "release", "view", tag, "--json", "assets", "--jq", ".assets[].name"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.split()


def _publish(tag: str, zip_path: Path, *, force: bool) -> None:
    """Create the release if missing and upload the asset, refusing overwrite unless ``force``."""
    assets = _existing_assets(tag)
    if assets is None:
        # Release does not exist yet: create it non-interactively (title + notes avoid any prompt),
        # pinned to the current commit so the tag target is explicit rather than gh's default.
        create = ["gh", "release", "create", tag, "--title", tag, "--notes", f"Model weights bundle ({tag})."]
        commit = _current_commit()
        if commit is not None:
            create += ["--target", commit]
        _run(create)
    elif ASSET_NAME in assets and not force:
        raise SystemExit(
            f"{ASSET_NAME} already exists on release '{tag}'. Model revisions are immutable: publish "
            f"a new tag (--tag models-vN) for changed weights, or pass --force only to replace an "
            f"upload no build has consumed yet."
        )

    upload = ["gh", "release", "upload", tag, str(zip_path)]
    if force:
        upload.append("--clobber")
    _run(upload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish ivcm-models.zip to a GitHub release.")
    parser.add_argument("--tag", default="models-v1", help="release tag to publish to (default: models-v1)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing asset (only for fixing an upload no build has consumed)",
    )
    args = parser.parse_args(argv)

    models: list[tuple[str, ModelConfig]] = [
        ("cls", CONFIG.classification),
        ("seg", CONFIG.segmentation),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "models"
        stage_models(models, staging)
        for name, _ in models:
            _validate_checkpoint(staging / name / "best-weights.ckpt")
        zip_path = Path(tmp) / ASSET_NAME
        _zip_dir(staging, zip_path)
        _publish(args.tag, zip_path, force=args.force)

    print(f"Published {ASSET_NAME} to release '{args.tag}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
