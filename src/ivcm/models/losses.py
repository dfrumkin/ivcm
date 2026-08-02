"""Segmentation loss: soft Dice + cross-entropy for soft (two-annotator) targets."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SegDiceCELoss(nn.Module):
    """Binary segmentation loss ``dice_weight * soft_dice + ce_weight * soft_ce`` on a soft target.

    Consumes ``(logits, p)`` where ``logits`` is ``(B, 2, H, W)`` and ``p`` is a ``(B, H, W)`` float
    foreground probability in ``[0, 1]`` (``0.5`` where two annotators disagree; a single annotator
    yields ``{0, 1}``). Both terms accept the soft target:

    - **soft CE** via a class-probability target ``stack([1 - p, p])`` (``F.cross_entropy`` supports
      probability targets), so ``p ∈ {0, 0.5, 1}`` is handled uniformly;
    - **soft Dice** on the foreground probability ``softmax(logits)[:, 1]`` against ``p``, with the
      sums taken over the **whole batch** (a global soft Dice, not a per-image Dice averaged across
      the batch).

    Inputs are validated up front (shape, finiteness, ``[0, 1]`` range), since this is the
    segmentation criterion contract and a malformed override/fixture should fail loudly.
    """

    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4 or logits.shape[1] != 2:
            raise ValueError(f"logits must be (B, 2, H, W) for binary segmentation; got {tuple(logits.shape)}.")
        if tuple(p.shape) != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(f"target p must be (B, H, W) matching logits {tuple(logits.shape)}; got {tuple(p.shape)}.")
        if not bool(torch.isfinite(p).all()) or float(p.min()) < 0.0 or float(p.max()) > 1.0:
            raise ValueError("target p must contain only finite values in [0, 1].")

        soft_ce = F.cross_entropy(logits, torch.stack([1.0 - p, p], dim=1))

        fg = logits.softmax(dim=1)[:, 1]  # (B, H, W) foreground probability
        intersection = (fg * p).sum()
        cardinality = fg.sum() + p.sum()
        soft_dice = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return self.dice_weight * soft_dice + self.ce_weight * soft_ce
