"""Unit tests for the soft Dice + cross-entropy segmentation criterion (``SegDiceCELoss``)."""

from __future__ import annotations

import pytest
import torch

from ivcm.models import SegDiceCELoss


def _asymmetric_logits() -> torch.Tensor:
    # Asymmetric, non-equal logits so a soft-vs-rounded difference can't cancel by symmetry.
    gen = torch.Generator().manual_seed(0)
    return torch.randn(2, 2, 4, 4, generator=gen) * 2.0


def test_soft_target_preserved_in_each_term() -> None:
    """Each term (CE and Dice), and the combo, must value a soft 0.5 target differently from the
    ``>= 0.5``-rounded target — proving 0.5 is preserved, not truncated."""
    logits = _asymmetric_logits()
    soft = torch.full((2, 4, 4), 0.5)  # all-disagreement
    rounded = (soft >= 0.5).float()  # union rounding -> all ones

    combo = SegDiceCELoss()
    ce_only = SegDiceCELoss(dice_weight=0.0, ce_weight=1.0)
    dice_only = SegDiceCELoss(dice_weight=1.0, ce_weight=0.0)

    for loss in (combo, ce_only, dice_only):
        soft_val = loss(logits, soft)
        assert abs(soft_val.item() - loss(logits, rounded).item()) > 1e-6
        assert bool(torch.isfinite(soft_val))


def test_weights_scale_the_two_terms() -> None:
    """The combo equals the weighted sum of the isolated Dice-only and CE-only terms."""
    logits = _asymmetric_logits()
    p = torch.rand(2, 4, 4, generator=torch.Generator().manual_seed(1))

    ce = SegDiceCELoss(dice_weight=0.0, ce_weight=1.0)(logits, p)
    dice = SegDiceCELoss(dice_weight=1.0, ce_weight=0.0)(logits, p)
    combo = SegDiceCELoss(dice_weight=1.0, ce_weight=1.0)(logits, p)

    assert abs(combo.item() - (ce + dice).item()) < 1e-5


@pytest.mark.parametrize(
    "logits, p",
    [
        (torch.zeros(2, 3, 4, 4), torch.zeros(2, 4, 4)),  # non-2-channel logits
        (torch.zeros(2, 2, 4, 4), torch.full((2, 4, 4), 2.0)),  # p out of [0, 1]
        (torch.zeros(2, 2, 4, 4), torch.zeros(2, 5, 5)),  # spatial mismatch
        (torch.zeros(2, 2, 4, 4), torch.full((2, 4, 4), float("nan"))),  # non-finite p
    ],
)
def test_rejects_malformed_inputs(logits: torch.Tensor, p: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        SegDiceCELoss()(logits, p)
