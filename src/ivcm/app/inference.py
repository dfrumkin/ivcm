"""Model loading and inference for the clinical-review UI (Qt-free).

``InferenceEngine`` loads both trained models in its constructor (not at import) and runs
classification, segmentation, and morphometry. The class-index and mask logic are split into
small pure helpers (taking ready-made logits) so they are unit-testable without a checkpoint.
All forward passes run under ``torch.inference_mode()``.

Two flavours of inference coexist. The single-image ``classify`` / ``segment`` methods serve the
interactive checkbox path (and the per-file fallback used by the directory batch). The batched
``classify_batch`` / ``segment_batch`` methods serve the directory batch (``ivcm.app.batch``).
Classification stacks a list of images (optionally preprocessed in parallel via a passed
executor) into one tensor and runs the forward in device sub-batches, amortising per-launch
overhead. Segmentation runs at each image's native resolution via sliding-window inference, so
its batch path cannot stack differently-sized images — it runs each image individually (windows
are batched internally by ``sw_batch_size``); decode/preprocess is still parallelised via the
executor.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import Executor
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn

from ivcm.app.config import AppConfig, ModelConfig
from ivcm.data import build_eval_transform, build_seg_eval_transform, ensure_square
from ivcm.models.segmentation import SegmentationModule, sliding_window_logits
from ivcm.morphometry import cell_morphometry
from ivcm.utils import select_device


def load_image(path: Path) -> np.ndarray:
    """Load an image as a single-channel ``HWC`` float array in ``[0, 1]``.

    Matches ``ClassificationImageDataset`` / ``NiiSegmentationDataset`` preprocessing: grayscale
    (first frame of a multi-page TIFF), normalised by 255, with a trailing channel axis. Enforces
    the square-input invariant (raising ``NonSquareImageError``) so the engine only ever yields
    square images; the UI validates the whole directory up front (see ``ivcm.app.batch``).
    """
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    ensure_square(image.shape, source=path.name)
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


def mask_from_seg_logits(logits: torch.Tensor) -> np.ndarray:
    """Binary ``uint8`` mask from segmentation ``logits`` (``(1, C, H, W)``).

    Mirrors ``predict.py::_predict_fold``: argmax over the class dim. The logits come from
    sliding-window inference over the full native-resolution image, so they are already at the
    original resolution — no resize back, so the mask side length is the native pixel resolution
    that morphometry uses (with ``physical_image_size``) to convert its measurements to physical units.
    """
    preds = logits.argmax(dim=1)  # (1, H, W)
    return preds[0].to(torch.uint8).cpu().numpy()


def _transform_image(image: np.ndarray, size: int) -> torch.Tensor:
    """Resize/convert a single ``HWC`` image to a ``(1, size, size)`` **CPU** CHW tensor.

    Builds a fresh ``build_eval_transform`` per call (so no transform instance is shared across
    worker threads) and does not move to any device — the batch path stacks these and transfers
    once per sub-batch in ``forward_in_chunks``.
    """
    out = cast("dict[str, torch.Tensor]", build_eval_transform(size)(image=image))
    return out["image"]


def _transform_seg_image(image: np.ndarray) -> torch.Tensor:
    """Convert a single ``HWC`` image to a ``(1, H, W)`` **CPU** CHW tensor at native resolution.

    Segmentation does not resize — the full-resolution image is fed to sliding-window inference.
    Builds a fresh transform per call so no instance is shared across worker threads.
    """
    out = cast("dict[str, torch.Tensor]", build_seg_eval_transform()(image=image))
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


def strip_criterion_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop ``criterion.*`` entries from a checkpoint ``state_dict`` for inference loading.

    The criterion is a submodule of the training LightningModule, so training-time buffers — e.g.
    the weighted cross-entropy's ``criterion.weight``, injected from the data's class frequencies —
    are saved in the checkpoint. Inference never runs the loss and rebuilds it from config without
    those runtime weights, so those keys would be "unexpected" under a strict load. Removing them
    lets the remaining ``net.*`` weights load strictly.
    """
    return {k: v for k, v in state.items() if not k.startswith("criterion.")}


def disable_net_downloads(model_cfg: DictConfig) -> None:
    """Turn off ImageNet-pretrained initialisation before rebuilding the net for inference.

    Both nets are configured to fetch pretrained weights over the network when constructed
    (``timm.create_model``'s ``pretrained=True``; ``segmentation_models_pytorch``'s
    ``encoder_weights="imagenet"``). Inference overwrites every parameter from the trained
    checkpoint immediately afterwards, so that download is pure waste — and it breaks a standalone
    deliverable: it needs the network, and in a frozen windowed build its progress bar writes to a
    ``None`` stream (``AttributeError: 'NoneType' object has no attribute 'write'``). Disabling it
    yields identical loaded weights while making startup offline-safe.
    """
    net = cast("DictConfig | None", OmegaConf.select(model_cfg, "net"))
    if net is None:
        return
    if OmegaConf.select(net, "pretrained") is not None:
        net.pretrained = False
    if OmegaConf.select(net, "encoder_weights") is not None:
        net.encoder_weights = None


def _load_model(cfg: DictConfig, ckpt_path: Path, device: torch.device) -> nn.Module:
    """Instantiate the run's model and load its checkpoint weights for inference.

    The saved config is never fully resolved (it carries Hydra-only interpolations); only the
    ``model``/``loss``/``optimizer`` sub-nodes are read. ``scheduler=None`` avoids resolving the
    cosine scheduler's ``T_max: ${trainer.max_epochs}`` — inference never uses a scheduler.
    """
    disable_net_downloads(cfg.model)
    model = cast(
        nn.Module,
        instantiate(
            cfg.model,
            criterion=instantiate(cfg.loss),
            optimizer=instantiate(cfg.optimizer),
            scheduler=None,
        ),
    )
    full_state = cast("dict[str, torch.Tensor]", torch.load(ckpt_path, map_location=device)["state_dict"])
    model.load_state_dict(strip_criterion_state(full_state))
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

        self._cls_model = _load_model(cls_cfg, config.classification.ckpt_path(), self.device)
        self._seg_model = _load_model(seg_cfg, config.segmentation.ckpt_path(), self.device)

        # Sliding-window inference parameters, read off the loaded SegmentationModule (its
        # constructor supplies defaults, so a config that omits them still loads). Segmentation
        # runs at native resolution — no seg image size to resolve.
        seg_model = cast(SegmentationModule, self._seg_model)
        self.seg_roi = seg_model.roi_size
        self.seg_overlap = seg_model.sw_overlap
        self.seg_sw_batch = seg_model.sw_batch_size

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

    def _segment_tensor(self, x: torch.Tensor) -> np.ndarray:
        """Sliding-window segment one ``(1, 1, H, W)`` device tensor to a native-size uint8 mask."""
        with torch.inference_mode():
            logits = sliding_window_logits(self._seg_model, x, self.seg_roi, self.seg_overlap, self.seg_sw_batch)
        return mask_from_seg_logits(logits)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Binary nerve mask at the original image resolution (via sliding-window inference)."""
        x = _transform_seg_image(image).unsqueeze(0).to(self.device)
        return self._segment_tensor(x)

    def segment_batch(self, images: Sequence[np.ndarray], executor: Executor | None = None) -> list[np.ndarray]:
        """Batched ``segment``: a binary nerve mask per image at its original resolution.

        Native-resolution images may differ in size, so they cannot be stacked into one tensor;
        each is run through sliding-window inference individually (its windows are batched
        internally by ``sw_batch_size``). Decode/preprocess is parallelised via ``executor`` when
        given. ``images`` must be non-empty.
        """
        if executor is None:
            tensors: Iterable[torch.Tensor] = (_transform_seg_image(image) for image in images)
        else:
            tensors = executor.map(_transform_seg_image, images)
        return [self._segment_tensor(t.unsqueeze(0).to(self.device)) for t in tensors]

    def morphometry(self, mask: np.ndarray) -> dict[str, float]:
        """Count / density / average area / average perimeter for a mask, in physical units from
        the configured parameters."""
        return cell_morphometry(
            mask,
            sigma=self.config.sigma,
            physical_image_size=self.config.physical_image_size,
            min_size=self.config.min_size,
        )
