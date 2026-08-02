"""Classification LightningModule."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassRecall,
    MulticlassSpecificity,
)

from ivcm.models.base import BaseModule, Batch

log = logging.getLogger(__name__)


def format_confusion_matrix(counts: torch.Tensor, class_names: Sequence[str]) -> str:
    """Render a confusion-matrix counts grid (rows = true, cols = predicted) as an aligned text table.

    ``counts`` is a ``(C, C)`` integer tensor as returned by ``MulticlassConfusionMatrix.compute()``.
    Pure (no logging/metric state) so it is unit-testable from a supplied matrix.
    """
    names = list(class_names)
    n = len(names)
    grid: list[list[int]] = [[int(counts[i, j].item()) for j in range(n)] for i in range(n)]
    max_count = max((max(row) for row in grid), default=0)
    cell_w = max(len(str(max_count)), max((len(n) for n in names), default=0)) + 2
    label_w = max((len(n) for n in names), default=0) + 2
    header = " " * label_w + "".join(f"{n:>{cell_w}}" for n in names)
    rows = [
        f"{name:<{label_w}}" + "".join(f"{grid[i][j]:>{cell_w}}" for j in range(len(names)))
        for i, name in enumerate(names)
    ]
    return "\n".join(["Confusion matrix (rows = true, cols = predicted):", header, *rows])


class ClassificationModule(BaseModule):
    """N-way classification reporting per-class sensitivity, specificity, and AUC, plus a confusion
    matrix logged (as text) each validation epoch.

    The class count is derived from ``class_names`` (the single source of truth), which also labels
    the per-class metric keys ``val/<metric>/<class>`` and the confusion-matrix table. Inverse-
    frequency class weights, when supplied by the datamodule (``class_weights``, from real data), are
    injected into the criterion in :meth:`on_fit_start`.
    """

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        class_names: Sequence[str],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__(net=net, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        self.class_names = list(class_names)
        num_classes = len(self.class_names)

        # Per-class (average=None) metrics compute to vectors, which self.log cannot accept;
        # they are computed and logged with explicit keys in on_validation_epoch_end.
        self.val_sensitivity = MulticlassRecall(num_classes=num_classes, average=None)
        self.val_specificity = MulticlassSpecificity(num_classes=num_classes, average=None)
        self.val_auc = MulticlassAUROC(num_classes=num_classes, average=None)
        # Confusion matrix is a (C, C) matrix (not a scalar), so it is computed and emitted to the
        # run log explicitly rather than via self.log.
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    def on_fit_start(self) -> None:
        """Inject the datamodule's inverse-frequency ``class_weights`` into the criterion, if any.

        No-op when the datamodule supplies no weights (synthetic data, or ``weighted_loss=false``).
        Fails loudly if weights exist but the configured criterion cannot carry a ``weight`` — the
        "automatic on real data" contract must not be silently skipped by a loss override.
        """
        weights = getattr(cast(Any, self.trainer).datamodule, "class_weights", None)
        if weights is None:
            return
        if not hasattr(self.criterion, "weight"):
            raise TypeError(
                f"datamodule supplied class_weights but criterion {type(self.criterion).__name__} has "
                "no `weight` attribute to apply them to (use a cross-entropy-style loss, or set "
                "data.weighted_loss=false)."
            )
        self.criterion.weight = cast(torch.Tensor, weights).to(self.device)

    def _step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        return loss, logits, targets

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, logits, targets = self._step(batch)
        self.val_sensitivity.update(logits, targets)
        self.val_specificity.update(logits, targets)
        self.val_auc.update(logits, targets)
        # Skip the confusion matrix during sanity validation so its counts reflect only real epochs;
        # the per-class metrics above intentionally still update (they compute-and-reset every epoch).
        if not self.trainer.sanity_checking:
            self.val_confmat.update(logits, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        for metric_name, metric in (
            ("sensitivity", self.val_sensitivity),
            ("specificity", self.val_specificity),
            ("auc", self.val_auc),
        ):
            per_class = metric.compute()  # length-N tensor
            self.log_dict({f"val/{metric_name}/{name}": per_class[i] for i, name in enumerate(self.class_names)})
            self.log(f"val/{metric_name}_macro", per_class.mean())
            metric.reset()

        # Confusion matrix (guard scoped to the CM only): emit for real validation epochs, not sanity.
        if not self.trainer.sanity_checking:
            log.info("\n%s", format_confusion_matrix(self.val_confmat.compute(), self.class_names))
        self.val_confmat.reset()
