"""Device selection shared by inference entry points."""

from __future__ import annotations

import torch


def select_device() -> torch.device:
    """Pick the best available device once: CUDA, then MPS, then CPU.

    Argmax-only inference (the only inference this project runs) is MPS-safe.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
