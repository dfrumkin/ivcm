"""Automatic inverse-frequency class weighting for classification."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import lightning as L
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn

from ivcm.data import (
    BaseDataModule,
    ClassificationImageDataset,
    SyntheticClassificationDataset,
    build_eval_transform,
    inverse_frequency_weights,
)
from ivcm.models import ClassificationModule

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")
CLASSES = ["a", "b", "c"]
TRAIN_COUNTS = {"a": 8, "b": 4, "c": 2}  # imbalanced


def _make_cls_fixture(root: Path, size: int = 8) -> None:
    for split, counts in (("train", TRAIN_COUNTS), ("val", {c: 1 for c in CLASSES})):
        for cls, n in counts.items():
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(np.zeros((size, size), np.uint8)).save(d / f"{i}.tif")


class _StubNet(nn.Module):
    def __init__(self, num_classes: int, size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(size * size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def _sgd(params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=0.01)


def _datamodule(root: Path, *, weighted_loss: bool, size: int = 8) -> BaseDataModule:
    train = ClassificationImageDataset(root / "train", CLASSES, transform=build_eval_transform(size))
    val = ClassificationImageDataset(root / "val", CLASSES, transform=build_eval_transform(size))
    return BaseDataModule(train, val, batch_size=2, class_names=CLASSES, weighted_loss=weighted_loss)


def _fit(dm: BaseDataModule, size: int = 8) -> ClassificationModule:
    model = ClassificationModule(
        net=_StubNet(len(CLASSES), size),
        criterion=nn.CrossEntropyLoss(),
        optimizer=_sgd,
        class_names=CLASSES,
    )
    trainer = L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(model, datamodule=dm)
    return model


def test_inverse_frequency_weights_properties() -> None:
    counts = [TRAIN_COUNTS[c] for c in CLASSES]  # [8, 4, 2]
    w = inverse_frequency_weights(counts)
    # Rarer class -> larger weight, in class order.
    assert w[0].item() < w[1].item() < w[2].item()
    # Sample-frequency-weighted mean is exactly 1.
    counts_t = torch.tensor(counts, dtype=torch.float32)
    assert abs(float((counts_t * w).sum() / counts_t.sum()) - 1.0) < 1e-6
    # A zero count raises.
    with pytest.raises(ValueError):
        inverse_frequency_weights([1, 0, 3])


def test_real_datamodule_weights_injected_into_criterion(tmp_path: Path) -> None:
    _make_cls_fixture(tmp_path)
    dm = _datamodule(tmp_path, weighted_loss=True)
    expected = inverse_frequency_weights([TRAIN_COUNTS[c] for c in CLASSES])
    assert dm.class_weights is not None
    assert torch.allclose(dm.class_weights, expected)

    model = _fit(dm)
    weight = cast(nn.CrossEntropyLoss, model.criterion).weight
    assert weight is not None
    assert torch.allclose(weight, expected)


def test_synthetic_datamodule_is_unweighted() -> None:
    train = SyntheticClassificationDataset(8, len(CLASSES), image_size=8, transform=build_eval_transform(8))
    val = SyntheticClassificationDataset(4, len(CLASSES), image_size=8, seed=1, transform=build_eval_transform(8))
    dm = BaseDataModule(train, val, batch_size=2, class_names=CLASSES, weighted_loss=True)
    assert dm.class_weights is None

    model = _fit(dm)
    assert cast(nn.CrossEntropyLoss, model.criterion).weight is None


def test_weighted_loss_false_disables(tmp_path: Path) -> None:
    _make_cls_fixture(tmp_path)
    dm = _datamodule(tmp_path, weighted_loss=False)
    assert dm.class_weights is None

    # The config key is present so `data.weighted_loss=false` composes under Hydra struct mode.
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["task=classification", "data=classification_real", "data.weighted_loss=false"],
        )
    assert cfg.data.weighted_loss is False
