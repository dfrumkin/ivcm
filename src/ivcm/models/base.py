"""Shared LightningModule base: wires a net, criterion, and partial optim/scheduler."""

from __future__ import annotations

from collections.abc import Callable

import lightning as L
import torch
from torch import nn


class BaseModule(L.LightningModule):
    """Common optimizer/scheduler plumbing shared by the task modules.

    ``optimizer`` and ``scheduler`` are Hydra *partials* (``_partial_: true``): callables
    that receive the parameters / optimizer to finish construction in
    :meth:`configure_optimizers`.
    """

    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is None:
            return optimizer
        scheduler = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
