"""Segmentation LightningModule."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

# sliding_window_inference is re-exported by monai.inferers at runtime but pyright resolves it to
# the private submodule and its signature is only partially typed; suppress both at the boundary.
from monai.inferers import (
    sliding_window_inference,  # pyright: ignore[reportPrivateImportUsage, reportUnknownVariableType]  # noqa: E501
)
from torch import nn
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryRecall,
    BinarySpecificity,
)
from torchmetrics.segmentation import DiceScore, MeanIoU

from ivcm.models.base import BaseModule, Batch


def sliding_window_logits(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    roi_size: int,
    overlap: float,
    sw_batch_size: int,
) -> torch.Tensor:
    """Full-image segmentation logits via MONAI sliding-window inference.

    Tiles ``images`` (``(B, C, H, W)``, native resolution) into ``roi_size`` windows, runs
    ``predictor`` on each (in batches of ``sw_batch_size``), and stitches the per-window logits
    back into a ``(B, num_classes, H, W)`` map, averaging in overlap regions. Windows extending
    past the border are padded by MONAI, so ``images`` smaller than ``roi_size`` also work.

    ``predictor`` is any callable returning ``(N, num_classes, h, w)`` logits (not necessarily a
    ``SegmentationModule``), so this is unit-testable with a stub. The scalar ``roi_size`` is
    broadcast to the 2D ``(roi_size, roi_size)`` window MONAI expects.
    """
    logits = sliding_window_inference(
        inputs=images,
        roi_size=(roi_size, roi_size),
        sw_batch_size=sw_batch_size,
        predictor=predictor,
        overlap=overlap,
    )
    return cast(torch.Tensor, logits)


class SegmentationModule(BaseModule):
    """Binary (background vs. nerve) segmentation.

    Reports pixel-level sensitivity, specificity, AUROC, and AUPRC (the foreground class)
    alongside the standard Dice/IoU. AUPRC (average precision) is the best-checkpoint
    selection metric: it is threshold-free (so it does not bake in the 0.5 operating point)
    and, unlike AUROC, imbalance-aware — it ignores the dominant background true negatives.
    ``num_classes`` (= 2, fed from ``${len:${class_names}}``) is needed by the Dice/IoU metrics.
    """

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        num_classes: int,
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
        roi_size: int = 192,
        sw_overlap: float = 0.25,
        sw_batch_size: int = 4,
    ) -> None:
        super().__init__(net=net, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
        # Sliding-window inference parameters. Training runs a direct forward on ``roi_size``
        # crops; validation/prediction slide a ``roi_size`` window over the full-resolution
        # image (see ``sliding_window_logits``). Defaults let a config that omits these keys
        # still instantiate — but a pre-change (resize-trained) checkpoint is not made correct
        # by them; those runs predate crop training.
        self.roi_size = roi_size
        self.sw_overlap = sw_overlap
        self.sw_batch_size = sw_batch_size
        # Dice/IoU exclude background and consume integer (B, H, W) preds/targets directly.
        # (Kwargs are inlined rather than packed into a dict so each keyword keeps its precise
        # type — a heterogeneous dict would widen them to `int | str` and fail the metric sigs.)
        self.train_iou = MeanIoU(num_classes=num_classes, include_background=False, input_format="index")
        self.val_iou = MeanIoU(num_classes=num_classes, include_background=False, input_format="index")
        self.train_dice = DiceScore(num_classes=num_classes, include_background=False, input_format="index")
        self.val_dice = DiceScore(num_classes=num_classes, include_background=False, input_format="index")

        # Binary pixel-level metrics on the foreground (nerve) class. AUROC/AUPRC use the
        # default (non-binned) mode: they just store the pixel predictions (~tens of MB per
        # fold, acceptable here) and compute via sorting. The binned `thresholds=N` mode would
        # bound memory further but triggers torchmetrics' `_bincount` fallback on MPS, which
        # materializes a huge mesh and OOMs — so non-binned is the portable choice.
        #
        # Their sort-based compute hits a buggy MPS kernel at the real pixel scale (returns
        # out-of-bounds indices -> AcceleratorError), so both are computed on CPU:
        # validation_step updates them with CPU tensors and on_validation_epoch_end computes +
        # logs the scalars. They must NOT be handed to self.log() as Metric objects — Lightning's
        # result collection would relocate their state back to the device and re-trigger the bug.
        # Sensitivity and specificity are threshold counts (no sort), so they stay on device as
        # logged metrics.
        self.val_sensitivity = BinaryRecall()
        self.val_specificity = BinarySpecificity()
        self.val_auc = BinaryAUROC()
        # AUPRC (average precision): the best-checkpoint selection metric (val/auprc).
        self.val_auprc = BinaryAveragePrecision()

    def _step(
        self,
        batch: Batch,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared loss/metric extraction. ``logits_fn`` maps images to ``(B, 2, H, W)`` logits:
        a direct forward in training (inputs are already ``roi_size`` crops), a sliding-window
        forward in validation (inputs are full native-resolution images).

        The mask arrives as a float **foreground probability** ``p`` in ``{0, 0.5, 1}`` (``0.5`` where
        two annotators disagree). The loss consumes ``p`` directly (soft target); the metrics consume
        a hard target ``(p >= 0.5)`` — foreground = a pixel *either* annotator marked (union)."""
        images, masks = batch
        logits = logits_fn(images)  # (B, 2, H, W)
        if logits.shape[1] != 2:
            raise ValueError(f"binary segmentation expects 2 output channels; got {logits.shape[1]}.")
        p = masks.to(device=logits.device, dtype=logits.dtype)  # (B, H, W) foreground prob
        loss = self.criterion(logits, p)
        hard = (p >= 0.5).long()  # metric target (union): 1 where either annotator marked foreground
        preds = logits.argmax(dim=1)
        fg_prob = logits.softmax(dim=1)[:, 1]  # foreground (nerve) probability, (B, H, W)
        return loss, preds, fg_prob, hard

    def _sliding_window_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Full-image logits via sliding-window inference over ``roi_size`` windows."""
        return sliding_window_logits(self.forward, images, self.roi_size, self.sw_overlap, self.sw_batch_size)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, preds, _fg_prob, masks = self._step(batch, self.forward)
        self.train_iou(preds, masks)
        self.train_dice(preds, masks)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/iou", self.train_iou, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/dice", self.train_dice, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        loss, preds, fg_prob, masks = self._step(batch, self._sliding_window_logits)
        self.val_iou(preds, masks)
        self.val_dice(preds, masks)
        self.val_sensitivity(fg_prob, masks)
        self.val_specificity(fg_prob, masks)
        # AUROC/AUPRC accumulated on CPU and computed in on_validation_epoch_end (see __init__).
        fg_prob_cpu, masks_cpu = fg_prob.detach().cpu(), masks.detach().cpu()
        self.val_auc.update(fg_prob_cpu, masks_cpu)
        self.val_auprc.update(fg_prob_cpu, masks_cpu)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/iou", self.val_iou, prog_bar=False, on_epoch=True)
        self.log("val/dice", self.val_dice, prog_bar=True, on_epoch=True)
        # Stable keys consumed by cv.py's per-fold aggregation.
        self.log("val/sensitivity", self.val_sensitivity, prog_bar=True, on_epoch=True)
        self.log("val/specificity", self.val_specificity, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        # Compute AUROC/AUPRC on CPU here (logging the Metric objects in validation_step would
        # let Lightning move their state to the device and hit the MPS sort bug). val/auc and
        # val/auprc are stable keys cv.py reads; val/auprc is the best-checkpoint monitor.
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.log("val/auprc", self.val_auprc.compute(), prog_bar=True)
        self.val_auc.reset()
        self.val_auprc.reset()
