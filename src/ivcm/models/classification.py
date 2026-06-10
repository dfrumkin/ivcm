"""Classification LightningModule."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassRecall,
    MulticlassSpecificity,
)

from ivcm.models.base import BaseModule, Batch


class ClassificationModule(BaseModule):
    """4-way classification reporting per-class sensitivity, specificity, and AUC.

    The class count is derived from ``class_names`` (the single source of truth), which
    also labels the per-class metric keys ``val/<metric>/<class>``.
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
