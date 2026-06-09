"""Segmentation LightningModule."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryRecall, BinarySpecificity
from torchmetrics.segmentation import DiceScore, MeanIoU

from ivcm.models.base import BaseModule


class SegmentationModule(BaseModule):
    """Binary (background vs. nerve) segmentation.

    Reports pixel-level sensitivity, specificity, and AUC (the foreground class) alongside
    the standard Dice/IoU. ``num_classes`` (= 2, fed from ``${len:${class_names}}``) is
    needed by the Dice/IoU metrics.
    """

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        num_classes: int,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__(net=net, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        # Dice/IoU exclude background and consume integer (B, H, W) preds/targets directly.
        metric_kwargs = dict(num_classes=num_classes, include_background=False, input_format="index")
        self.train_iou = MeanIoU(**metric_kwargs)
        self.val_iou = MeanIoU(**metric_kwargs)
        self.train_dice = DiceScore(**metric_kwargs)
        self.val_dice = DiceScore(**metric_kwargs)

        # Binary pixel-level metrics on the foreground (nerve) class. AUROC uses the default
        # (non-binned) mode: it just stores the pixel predictions (~tens of MB per fold,
        # acceptable here) and computes via sorting. The binned `thresholds=N` mode would
        # bound memory further but triggers torchmetrics' `_bincount` fallback on MPS, which
        # materializes a huge mesh and OOMs — so non-binned is the portable choice.
        self.val_sensitivity = BinaryRecall()
        self.val_specificity = BinarySpecificity()
        self.val_auc = BinaryAUROC()

    def _step(self, batch):
        images, masks = batch
        masks = masks.long()  # Dice/CE losses and the metrics require integer targets.
        logits = self(images)  # (B, C, H, W)
        loss = self.criterion(logits, masks)
        preds = logits.argmax(dim=1)
        fg_prob = logits.softmax(dim=1)[:, 1]  # foreground (nerve) probability, (B, H, W)
        return loss, preds, fg_prob, masks

    def training_step(self, batch, batch_idx):
        loss, preds, _fg_prob, masks = self._step(batch)
        self.train_iou(preds, masks)
        self.train_dice(preds, masks)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/iou", self.train_iou, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/dice", self.train_dice, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, fg_prob, masks = self._step(batch)
        self.val_iou(preds, masks)
        self.val_dice(preds, masks)
        self.val_sensitivity(fg_prob, masks)
        self.val_specificity(fg_prob, masks)
        self.val_auc(fg_prob, masks)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/iou", self.val_iou, prog_bar=False, on_epoch=True)
        self.log("val/dice", self.val_dice, prog_bar=True, on_epoch=True)
        # Stable keys consumed by cv.py's per-fold aggregation.
        self.log("val/sensitivity", self.val_sensitivity, prog_bar=True, on_epoch=True)
        self.log("val/specificity", self.val_specificity, prog_bar=True, on_epoch=True)
        self.log("val/auc", self.val_auc, prog_bar=True, on_epoch=True)
        return loss
