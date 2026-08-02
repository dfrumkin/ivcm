"""Confusion matrix output + 5-class classification composition (data-free)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import lightning as L
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ivcm.models import ClassificationModule
from ivcm.models.classification import format_confusion_matrix

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")
CM_MARKER = "Confusion matrix (rows = true"


def test_format_confusion_matrix() -> None:
    counts = torch.tensor([[2, 1], [0, 3]])
    text = format_confusion_matrix(counts, ["cat", "dog"])
    assert CM_MARKER in text
    assert "cat" in text and "dog" in text
    # Row totals correspond to the true-class support (rows = true).
    assert "2" in text and "3" in text
    # One header line + one row per class.
    assert len(text.splitlines()) == 1 + 1 + 2


def test_classification_composes_five_classes() -> None:
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=["task=classification", "logger=none"])
    assert list(cfg.class_names) == ["endothelium", "epithelium", "nerves", "stroma", "gel"]
    assert cfg.model.net.num_classes == 5


class _StubNet(nn.Module):
    def __init__(self, num_classes: int, features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def _sgd(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=0.01)


def _loader(n: int, num_classes: int, features: int) -> DataLoader[tuple[torch.Tensor, ...]]:
    gen = torch.Generator().manual_seed(0)
    images = torch.rand(n, 1, features, generator=gen)
    labels = torch.randint(0, num_classes, (n,), generator=gen)
    return DataLoader(TensorDataset(images, labels), batch_size=n)


def test_confusion_matrix_emitted_once_per_real_epoch(caplog: pytest.LogCaptureFixture) -> None:
    """With sanity validation enabled, exactly one confusion-matrix table is emitted — from the real
    validation epoch, not the sanity pass."""
    num_classes, features = 3, 8
    model = ClassificationModule(
        net=_StubNet(num_classes, features),
        criterion=nn.CrossEntropyLoss(),
        optimizer=_sgd,
        class_names=["x", "y", "z"],
    )
    trainer = L.Trainer(
        max_epochs=1,
        num_sanity_val_steps=2,  # sanity validation ENABLED, so the guard is exercised
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    with caplog.at_level(logging.INFO, logger="ivcm.models.classification"):
        trainer.fit(model, _loader(6, num_classes, features), _loader(6, num_classes, features))

    emissions = [r for r in caplog.records if CM_MARKER in r.getMessage()]
    assert len(emissions) == 1
