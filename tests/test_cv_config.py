from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import ivcm  # noqa: F401 — registers the `len` OmegaConf resolver on import

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_cv_config_wiring():
    """cv.yaml composes the segmentation stack with the real CV datamodule (no default-task leak)."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv")

    assert cfg.model._target_.endswith("SegmentationModule")
    assert cfg.data._target_.endswith("SegmentationCVDataModule")
    assert cfg.data.n_folds == 5
    assert cfg.data.fold == 0  # default fold, before the cv loop mutates it
    assert cfg.seed is not None  # run() reads cfg.seed unconditionally


def test_cv_config_resolves():
    """Full resolve (logger disabled to dodge ${hydra:...}) catches resolver/root-field gaps."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="cv", overrides=["logger=none"])

    OmegaConf.to_yaml(cfg, resolve=True)  # raises if any interpolation/resolver is broken
    assert cfg.model.net.classes == 2  # ${len:${class_names}} over [background, nerve]
    assert cfg.model.num_classes == 2
    assert cfg.data.fold == 0
    assert "trainer" in cfg  # provided by the shared common.yaml
