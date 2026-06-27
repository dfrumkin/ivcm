"""Model loading and inference for the clinical-review UI (Qt-free).

``InferenceEngine`` loads both trained models in its constructor (not at import) and runs
classification, segmentation, and morphometry. The class-index and mask-resize logic are split
into small pure helpers (taking ready-made logits) so they are unit-testable without a
checkpoint. All forward passes run under ``torch.inference_mode()``.

Two flavours of inference coexist. The single-image ``classify`` / ``segment`` methods serve the
interactive checkbox path (and the per-file fallback used by the directory batch). The batched
``classify_batch`` / ``segment_batch`` methods serve the directory batch (``ivcm.app.batch``): they
preprocess a list of images (optionally in parallel via a passed executor) into one stacked tensor
and run the forward in device sub-batches, amortising per-launch overhead. Preprocessing in the
batch path produces CPU tensors only; the per-sub-batch device transfer happens in
``forward_in_chunks`` so worker threads never touch the device.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import Executor
from functools import partial
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


def predicted_classes(logits: torch.Tensor) -> list[int]:
    """Per-row argmax class indices of batched classification ``logits`` (shape ``(N, C)``).

    The batch generalisation of ``predicted_class``: row ``i`` of the result equals
    ``predicted_class(logits[i : i + 1])``.
    """
    indices = logits.argmax(dim=-1).reshape(-1)
    return [int(indices[i].item()) for i in range(indices.shape[0])]


def forward_in_chunks(model: nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Run ``model`` over a CPU input ``x`` in device sub-batches and concatenate outputs on CPU.

    ``x`` is a stacked ``(N, ...)`` CPU tensor. It is split into sub-batches of at most
    ``batch_size``, each moved to ``device`` and run under ``torch.inference_mode()``; the outputs
    are moved back to CPU and concatenated in order. Moving per sub-batch (not the whole stack)
    bounds device memory and keeps the device-transfer contract explicit, so callers can prepare
    inputs off-device (e.g. in worker threads) without issuing concurrent transfers.
    """
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            chunk = x[start : start + batch_size].to(device)
            outputs.append(cast(torch.Tensor, model(chunk)).cpu())
    return torch.cat(outputs, dim=0)


def mask_from_seg_logits(logits: torch.Tensor, orig_hw: tuple[int, int]) -> np.ndarray:
    """Binary ``uint8`` mask from segmentation ``logits`` (``(1, C, h, w)``), resized to ``orig_hw``.

    Mirrors ``predict.py::_predict_fold``: argmax over the class dim, then nearest-neighbour
    resize back to the original resolution so labels stay discrete and morphometry measures in
    original pixels.
    """
    preds = logits.argmax(dim=1)  # (1, h, w)
    resized = F.interpolate(preds[:, None].float(), size=orig_hw, mode="nearest")
    return resized[0, 0].to(torch.uint8).cpu().numpy()


def _transform_image(image: np.ndarray, size: int) -> torch.Tensor:
    """Resize/convert a single ``HWC`` image to a ``(1, size, size)`` **CPU** CHW tensor.

    Builds a fresh ``build_eval_transform`` per call (so no transform instance is shared across
    worker threads) and does not move to any device — the batch path stacks these and transfers
    once per sub-batch in ``forward_in_chunks``.
    """
    out = cast("dict[str, torch.Tensor]", build_eval_transform(size)(image=image))
    return out["image"]


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
        return _transform_image(image, size).unsqueeze(0).to(self.device)

    def _stack(self, images: Sequence[np.ndarray], size: int, executor: Executor | None) -> torch.Tensor:
        """Preprocess ``images`` (optionally in parallel) into one stacked ``(N, 1, size, size)`` CPU tensor."""
        if executor is None:
            tensors: Iterable[torch.Tensor] = (_transform_image(image, size) for image in images)
        else:
            tensors = executor.map(partial(_transform_image, size=size), images)
        return torch.stack(list(tensors), dim=0)

    def classify(self, image: np.ndarray) -> bool:
        """Whether the image's predicted class is ``nerves``."""
        x = self._preprocess(image, self.cls_size)
        with torch.inference_mode():
            logits = cast(torch.Tensor, self._cls_model(x))
        return predicted_class(logits) == self.nerve_index

    def classify_batch(self, images: Sequence[np.ndarray], executor: Executor | None = None) -> list[bool]:
        """Batched ``classify``: whether each image's predicted class is ``nerves``.

        Preprocesses ``images`` (in parallel when ``executor`` is given) and runs the classifier in
        ``config.batch_size`` device sub-batches. ``images`` must be non-empty (the caller guards
        the empty case; stacking an empty list would raise).
        """
        x = self._stack(images, self.cls_size, executor)
        logits = forward_in_chunks(self._cls_model, x, self.config.batch_size, self.device)
        return [cls == self.nerve_index for cls in predicted_classes(logits)]

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Binary nerve mask at the original image resolution."""
        height, width = int(image.shape[0]), int(image.shape[1])
        x = self._preprocess(image, self.seg_size)
        with torch.inference_mode():
            logits = cast(torch.Tensor, self._seg_model(x))
        return mask_from_seg_logits(logits, (height, width))

    def segment_batch(self, images: Sequence[np.ndarray], executor: Executor | None = None) -> list[np.ndarray]:
        """Batched ``segment``: a binary nerve mask per image at its original resolution.

        Preprocesses ``images`` (in parallel when ``executor`` is given) and runs segmentation in
        ``config.batch_size`` device sub-batches; each mask is resized back to its own original size
        (sizes differ, so this stays per image). ``images`` must be non-empty.
        """
        x = self._stack(images, self.seg_size, executor)
        logits = forward_in_chunks(self._seg_model, x, self.config.batch_size, self.device)
        return [
            mask_from_seg_logits(logits[i : i + 1], (int(image.shape[0]), int(image.shape[1])))
            for i, image in enumerate(images)
        ]

    def morphometry(self, mask: np.ndarray) -> dict[str, float]:
        """Density / area / perimeter for a mask, using the configured morphometry constants."""
        return cell_morphometry(
            mask,
            sigma=self.config.sigma,
            constant=self.config.constant,
            min_size=self.config.min_size,
        )
