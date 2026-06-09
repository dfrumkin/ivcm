"""Classification LightningModule."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from ivcm.models.base import BaseModule


class ClassificationModule(BaseModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        num_classes: int,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__(net=net, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def _step(self, batch):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        return loss, logits, targets

    def training_step(self, batch, batch_idx):
        loss, logits, targets = self._step(batch)
        self.train_acc(logits, targets)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._step(batch)
        self.val_acc(logits, targets)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True)
        return loss
