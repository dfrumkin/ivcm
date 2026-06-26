"""Model loading and inference for the clinical-review UI (Qt-free).

``InferenceEngine`` loads both trained models in its constructor (not at import) and runs
classification, segmentation, and morphometry. The class-index and mask-resize logic are split
into small pure helpers (taking ready-made logits) so they are unit-testable without a
checkpoint. All forward passes run under ``torch.inference_mode()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn

from ivcm.app.config import AppConfig, ModelConfig
from ivcm.data import build_eval_transform
from ivcm.morphometry import cell_morphometry
from ivcm.utils import select_device


def load_image(path: Path) -> np.ndarray:
    """Load an image as a single-channel ``HWC`` float array in ``[0, 1]``.

    Matches ``ClassificationImageDataset`` / ``NiiSegmentationDataset`` preprocessing: grayscale
    (first frame of a multi-page TIFF), normalised by 255, with a trailing channel axis.
    """
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return image[..., None]


def predicted_class(logits: torch.Tensor) -> int:
    """The argmax class index of single-image classification ``logits`` (shape ``(1, C)``)."""
    return int(logits.argmax(dim=-1).reshape(-1)[0].item())


def mask_from_seg_logits(logits: torch.Tensor, orig_hw: tuple[int, int]) -> np.ndarray:
    """Binary ``uint8`` mask from segmentation ``logits`` (``(1, C, h, w)``), resized to ``orig_hw``.

    Mirrors ``predict.py::_predict_fold``: argmax over the class dim, then nearest-neighbour
    resize back to the original resolution so labels stay discrete and morphometry measures in
    original pixels.
    """
    preds = logits.argmax(dim=1)  # (1, h, w)
    resized = F.interpolate(preds[:, None].float(), size=orig_hw, mode="nearest")
    return resized[0, 0].to(torch.uint8).cpu().numpy()


def _select_node(cfg: DictConfig, key: str) -> Any:
    """``OmegaConf.select`` returning ``None`` for a missing key (kept typed at the boundary)."""
    return OmegaConf.select(cfg, key, default=None)


def _class_names(cfg: DictConfig) -> list[str]:
    names = _select_node(cfg, "class_names")
    if names is None:
        raise ValueError("saved config has no class_names")
    return [str(n) for n in cast(Sequence[object], names)]


def _resolve_image_size(cfg: DictConfig) -> int:
    """Input size: prefer root ``image_size``, fall back to ``data.size`` (both concrete ints)."""
    size = _select_node(cfg, "image_size")
    if size is None:
        size = _select_node(cfg, "data.size")
    if size is None:
        raise ValueError("saved config has neither image_size nor data.size")
    return int(cast(int, size))


def _load_cfg(mc: ModelConfig) -> DictConfig:
    """Load a run's saved Hydra config, after checking the config and checkpoint both exist."""
    hint = "edit ivcm/app/config.py to point at a real run"
    config_path = mc.config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path} ({hint})")
    ckpt_path = mc.ckpt_path()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path} ({hint})")
    return cast(DictConfig, OmegaConf.load(config_path))


def _load_model(cfg: DictConfig, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Instantiate the run's model and load its checkpoint weights for inference.

    The saved config is never fully resolved (it carries Hydra-only interpolations); only the
    ``model``/``loss``/``optimizer`` sub-nodes are read. ``scheduler=None`` avoids resolving the
    cosine scheduler's ``T_max: ${trainer.max_epochs}`` — inference never uses a scheduler.
    """
    model = cast(
        nn.Module,
        instantiate(
            cfg.model,
            criterion=instantiate(cfg.loss),
            optimizer=instantiate(cfg.optimizer),
            scheduler=None,
        ),
    )
    state = cast("dict[str, torch.Tensor]", torch.load(ckpt_path, map_location=device)["state_dict"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


class InferenceEngine:
    """Loads both models once and runs classification / segmentation / morphometry."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = select_device()

        cls_cfg = _load_cfg(config.classification)
        seg_cfg = _load_cfg(config.segmentation)

        cls_names = _class_names(cls_cfg)
        if "nerves" not in cls_names:
            raise ValueError(f"classification class_names {cls_names} does not contain 'nerves'")
        self.nerve_index = cls_names.index("nerves")

        seg_names = _class_names(seg_cfg)
        if len(seg_names) != 2 or seg_names[1] != "nerve":
            raise ValueError(
                f"segmentation class_names {seg_names} must be binary with 'nerve' as the "
                "foreground (index 1); mask class 1 is taken as the nerve foreground"
            )

        self.cls_size = _resolve_image_size(cls_cfg)
        self.seg_size = _resolve_image_size(seg_cfg)

        self._cls_model = _load_model(cls_cfg, config.classification.ckpt_path(), self.device)
        self._seg_model = _load_model(seg_cfg, config.segmentation.ckpt_path(), self.device)

    def _preprocess(self, image: np.ndarray, size: int) -> torch.Tensor:
        """Resize/convert a single ``HWC`` image to a batched ``(1, 1, size, size)`` device tensor."""
        out = cast("dict[str, torch.Tensor]", build_eval_transform(size)(image=image))
        return out["image"].unsqueeze(0).to(self.device)

    def classify(self, image: np.ndarray) -> bool:
        """Whether the image's predicted class is ``nerves``."""
        x = self._preprocess(image, self.cls_size)
        with torch.inference_mode():
            logits = cast(torch.Tensor, self._cls_model(x))
        return predicted_class(logits) == self.nerve_index

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Binary nerve mask at the original image resolution."""
        height, width = int(image.shape[0]), int(image.shape[1])
        x = self._preprocess(image, self.seg_size)
        with torch.inference_mode():
            logits = cast(torch.Tensor, self._seg_model(x))
        return mask_from_seg_logits(logits, (height, width))

    def morphometry(self, mask: np.ndarray) -> dict[str, float]:
        """Density / area / perimeter for a mask, using the configured morphometry constants."""
        return cell_morphometry(
            mask,
            sigma=self.config.sigma,
            constant=self.config.constant,
            min_size=self.config.min_size,
        )
