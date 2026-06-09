"""Segmentation LightningModule."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchmetrics.segmentation import DiceScore, MeanIoU

from ivcm.models.base import BaseModule


class SegmentationModule(BaseModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        num_classes: int,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__(net=net, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        # Both metrics exclude the background class (0) from the score, and consume
        # integer (B, H, W) preds/targets directly via input_format="index". Dice is the
        # standard segmentation metric and the companion to DiceLoss.
        metric_kwargs = dict(num_classes=num_classes, include_background=False, input_format="index")
        self.train_iou = MeanIoU(**metric_kwargs)
        self.val_iou = MeanIoU(**metric_kwargs)
        self.train_dice = DiceScore(**metric_kwargs)
        self.val_dice = DiceScore(**metric_kwargs)

    def _step(self, batch):
        images, masks = batch
        masks = masks.long()  # Dice/CE losses and the IoU metric require integer targets.
        logits = self(images)  # (B, C, H, W)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self._step(batch)
        self.train_iou(preds, masks)
        self.train_dice(preds, masks)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/iou", self.train_iou, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/dice", self.train_dice, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self._step(batch)
        self.val_iou(preds, masks)
        self.val_dice(preds, masks)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/iou", self.val_iou, prog_bar=False, on_epoch=True)
        self.log("val/dice", self.val_dice, prog_bar=True, on_epoch=True)
        return loss
