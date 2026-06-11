"""Assemble an out-of-fold prediction volume from a finished segmentation CV run.

Reads a completed ``cross-validate`` output directory and, for each fold's best checkpoint,
runs inference on that fold's validation slices. Because the folds partition every slice,
the per-fold predictions assemble into a single volume covering each slice exactly once,
written as ``<cv_dir>/predictions.nii`` at the ground-truth resolution and affine for
slice-by-slice visual comparison against ``mask_gt.nii``.

This is intentionally a plain (argparse) command, not a Hydra entry point: it loads the
run's own saved ``.hydra/config.yaml`` as the authoritative config (so the model
architecture, paths, ``n_folds`` and ``image_size`` match what was trained) and never
recomposes today's defaults or writes Hydra metadata into the finished run. It also never
resolves the whole saved config -- that config carries Hydra-only ``${hydra:...}``
interpolations (e.g. ``logger.save_dir``) that are unresolvable outside a Hydra job -- and
only touches the ``data``/``model``/``loss``/``optimizer``/``scheduler`` sub-nodes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from nibabel.nifti1 import Nifti1Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ivcm.data import NiiSegmentationDataset, build_eval_transform, contiguous_folds
from ivcm.data.datamodules import Sample
from ivcm.models.segmentation import SegmentationModule
from ivcm.utils import configure_logging

log = logging.getLogger(__name__)


def _load_nii(path: Path) -> Nifti1Image:
    """Load a ``.nii`` file, typed for ``dataobj``/``affine``/``header`` access."""
    return cast(Nifti1Image, nib.load(path))  # pyright: ignore[reportUnknownMemberType]  # nib.load has untyped **kwargs


def _load_run_cfg(cv_dir: Path) -> DictConfig:
    """Load the finished run's saved, fully-composed config (authoritative)."""
    config_yaml = cv_dir / ".hydra" / "config.yaml"
    if not config_yaml.is_file():
        raise FileNotFoundError(f"{config_yaml} not found; {cv_dir} is not a finished cross-validate run directory.")
    return cast(DictConfig, OmegaConf.load(config_yaml))


def _original_cwd(cv_dir: Path) -> Path:
    """The run's launch cwd (Hydra ``chdir: false``), against which relative data paths were
    resolved. Persisted as ``hydra.runtime.cwd`` in the saved ``hydra.yaml``."""
    hydra_yaml = cv_dir / ".hydra" / "hydra.yaml"
    if not hydra_yaml.is_file():
        raise FileNotFoundError(f"{hydra_yaml} not found; {cv_dir} is not a finished cross-validate run directory.")
    cwd = OmegaConf.select(OmegaConf.load(hydra_yaml), "hydra.runtime.cwd")
    if cwd is None:
        raise ValueError(f"{hydra_yaml} has no hydra.runtime.cwd; cannot resolve relative data paths.")
    return Path(str(cwd))


def _resolve_data_path(raw: str, original_cwd: Path) -> Path:
    """Resolve a (possibly relative) data path against the run's launch cwd, so the command
    finds the same files regardless of where it is invoked from."""
    path = Path(raw)
    return path if path.is_absolute() else original_cwd / path


def _select_device() -> torch.device:
    """Pick the best available device once; argmax-only inference is MPS-safe."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_best_ckpt(fold_dir: Path) -> Path:
    """The fold's single monitored best checkpoint (saved by ``callbacks=default``)."""
    ckpt_dir = fold_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("best-epoch*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"no best-epoch*.ckpt under {ckpt_dir}; the CV must be run with callbacks=default "
            "so the best checkpoint is saved."
        )
    if len(ckpts) > 1:
        raise ValueError(f"expected one best checkpoint under {ckpt_dir}, found {len(ckpts)}: {ckpts}")
    return ckpts[0]


def _load_fold_model(cfg: DictConfig, ckpt_path: Path, device: torch.device) -> SegmentationModule:
    """Rebuild the model from the run's config and load the fold checkpoint's weights.

    ``BaseModule`` does not ``save_hyperparameters``, so ``load_from_checkpoint`` cannot
    reconstruct the net; instead we instantiate from ``cfg`` (mirroring ``train.run``) and
    load the Lightning ``state_dict``.
    """
    scheduler = instantiate(cfg.scheduler) if cfg.get("scheduler") else None
    model: SegmentationModule = instantiate(
        cfg.model,
        criterion=instantiate(cfg.loss),
        optimizer=instantiate(cfg.optimizer),
        scheduler=scheduler,
    )
    state = torch.load(ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def _predict_fold(
    model: SegmentationModule,
    images: np.ndarray,
    masks: np.ndarray,
    val_idx: np.ndarray,
    size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a fold's validation slices, resized back to original resolution.

    Returns a ``(len(val_idx), H, W)`` uint8 array in ``val_idx`` order (the ``shuffle=False``
    loader yields slices in exactly that order), so the caller can place it with one
    fancy-index write.
    """
    height, width = images.shape[0], images.shape[1]
    dataset = NiiSegmentationDataset(images, masks, val_idx, transform=build_eval_transform(size))
    loader: DataLoader[Sample] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, _masks in loader:
            imgs = imgs.to(device)  # dataloader yields CPU tensors; match the model's device
            logits: torch.Tensor = model(imgs)  # (B, C, size, size)
            batch_preds = logits.argmax(dim=1)  # (B, size, size)
            # Nearest-neighbour back to the GT resolution keeps mask labels discrete.
            resized = F.interpolate(batch_preds[:, None].float(), size=(height, width), mode="nearest")
            resized = resized[:, 0].long().cpu()
            preds.extend(slice_pred.astype(np.uint8) for slice_pred in resized.numpy())

    assert len(preds) == len(val_idx), f"predicted {len(preds)} slices for {len(val_idx)} val indices"
    return np.stack(preds)


def write_oof_predictions(cfg: DictConfig, cv_dir: Path) -> Path:
    """Assemble and write the combined out-of-fold prediction volume.

    Self-contained: pass only the loaded run config and the run dir. Device and launch cwd
    are derived internally so every fold shares one of each.
    """
    cwd = _original_cwd(cv_dir)
    device = _select_device()
    log.info("Using device: %s", device)

    images_path = _resolve_data_path(str(cfg.data.images_path), cwd)
    masks_path = _resolve_data_path(str(cfg.data.masks_path), cwd)
    log.info("Images: %s", images_path)
    log.info("Ground-truth masks: %s", masks_path)
    for path in (images_path, masks_path):
        if not path.is_file():
            raise FileNotFoundError(f"data file not found: {path} (check IVCM_DATA_DIR matches the CV run)")

    images = np.asarray(_load_nii(images_path).dataobj)
    gt = _load_nii(masks_path)
    masks = np.asarray(gt.dataobj)

    size = int(cfg.data.size)
    batch_size = int(cfg.data.batch_size)
    n_folds = int(cfg.data.n_folds)
    n_slices = images.shape[2]
    log.info("Volume: %s, size=%d, n_folds=%d, model=%s", masks.shape, size, n_folds, cfg.model._target_)

    folds = contiguous_folds(n_slices, n_folds)
    out = np.zeros(masks.shape, dtype=np.uint8)
    filled = np.zeros(n_slices, dtype=bool)

    for fold, val_idx in enumerate(folds):
        ckpt = _find_best_ckpt(cv_dir / f"fold_{fold}")
        log.info("Fold %d: predicting %d slices from %s", fold, len(val_idx), ckpt.name)
        model = _load_fold_model(cfg, ckpt, device)
        preds = _predict_fold(model, images, masks, val_idx, size, batch_size, device)
        out[:, :, val_idx] = np.transpose(preds, (1, 2, 0))
        filled[val_idx] = True

    if not filled.all():
        missing = np.where(~filled)[0].tolist()
        raise RuntimeError(f"slices not covered by any fold (folds must partition all slices): {missing}")

    result = Nifti1Image(out, affine=gt.affine, header=gt.header)
    result.header.set_data_dtype(np.uint8)  # pyright: ignore[reportUnknownMemberType]  # nibabel header untyped
    out_path = cv_dir / "predictions.nii"
    nib.save(result, out_path)  # pyright: ignore[reportUnknownMemberType]  # nib.save has untyped args
    log.info("Wrote out-of-fold predictions: %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``predict-cv <cv_dir>``. ``argv`` defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        description="Write a combined out-of-fold prediction volume from a finished CV run."
    )
    parser.add_argument(
        "cv_dir",
        type=Path,
        help="Finished cross-validate output directory (holding fold_*/ and .hydra/).",
    )
    args = parser.parse_args(argv)
    cv_dir = cast(Path, args.cv_dir)

    configure_logging()
    cfg = _load_run_cfg(cv_dir)
    out_path = write_oof_predictions(cfg, cv_dir)
    print(out_path)


if __name__ == "__main__":
    main()
