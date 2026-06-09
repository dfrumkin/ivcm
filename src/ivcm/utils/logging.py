"""Logging setup: Rich console output plus a plain file log."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def configure_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger with a Rich console handler and optional file handler.

    Args:
        log_dir: Directory to write ``train.log`` into. When ``None`` (or when running
            under Hydra, the caller typically passes the Hydra run dir), only console
            logging is configured.
        level: Logging level for the root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Reset handlers so repeated calls (e.g. in tests) don't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    root.addHandler(console_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
        root.addHandler(file_handler)
