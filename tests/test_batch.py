"""Tests for the batched inference helpers and the Qt-free directory-batch orchestration.

All tests are Qt-free and checkpoint-free: ``forward_in_chunks`` / ``classify_batch`` /
``segment_batch`` run against tiny deterministic stub modules, and ``process_directory`` runs against
a stub engine and a stub ``load``. The parity tests assert exact equality, which holds for these
per-row-independent stub models (not relied upon for real convolution models — see the inference
module's numerical note).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ivcm.app.batch import process_directory
from ivcm.app.config import AppConfig, ModelConfig
from ivcm.app.inference import (
    InferenceEngine,
    forward_in_chunks,
    predicted_class,
    predicted_classes,
)
from ivcm.app.records import ImageRecord

# --- pure helpers --------------------------------------------------------


def test_predicted_classes_matches_per_row_predicted_class() -> None:
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.0]])
    assert predicted_classes(logits) == [2, 0, 1]
    # Agrees row-by-row with the single-row helper.
    assert predicted_classes(logits) == [predicted_class(logits[i : i + 1]) for i in range(logits.shape[0])]


class _SumModule(nn.Module):
    """Deterministic, per-row-independent stub: each row maps to its flattened sum."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1).sum(dim=1, keepdim=True)


@pytest.mark.parametrize("n", [4, 5])  # exact multiple of batch_size (2) and not
def test_forward_in_chunks_equals_single_full_batch(n: int) -> None:
    model = _SumModule().eval()
    x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)

    chunked = forward_in_chunks(model, x, batch_size=2, device=torch.device("cpu"))
    with torch.inference_mode():
        whole = model(x)

    assert chunked.shape == whole.shape
    torch.testing.assert_close(chunked, whole)


# --- batch-vs-single parity ----------------------------------------------


class _StubClsModel(nn.Module):
    """(N, 1, H, W) -> (N, 3) logits via per-image reductions (deterministic, batch-invariant)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x.mean(dim=(1, 2, 3)), x.amax(dim=(1, 2, 3)), x.amin(dim=(1, 2, 3))], dim=1)


class _StubSegModel(nn.Module):
    """(N, 1, H, W) -> (N, 2, H, W) logits with foreground (class 1) where the pixel exceeds 0.5."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fg = (x > 0.5).float()
        return torch.cat([1.0 - fg, 2.0 * fg], dim=1)


def _stub_engine(size: int = 8, batch_size: int = 2) -> InferenceEngine:
    """An ``InferenceEngine`` with stub models, bypassing checkpoint loading (test-local)."""
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.device = torch.device("cpu")
    engine.cls_size = size
    engine.nerve_index = 1
    # Segmentation runs via sliding-window inference; the stub carries the window params
    # normally read off the loaded SegmentationModule.
    engine.seg_roi = size
    engine.seg_overlap = 0.25
    engine.seg_sw_batch = batch_size
    engine._cls_model = _StubClsModel().eval()  # pyright: ignore[reportPrivateUsage]
    engine._seg_model = _StubSegModel().eval()  # pyright: ignore[reportPrivateUsage]
    engine.config = SimpleNamespace(batch_size=batch_size)  # type: ignore[assignment]
    return engine


def test_classify_and_segment_batch_match_single_image() -> None:
    engine = _stub_engine(size=8, batch_size=2)
    rng = np.random.default_rng(0)
    images = [rng.random((8, 8, 1), dtype=np.float32) for _ in range(5)]  # spans >1 sub-batch

    expected_flags = [engine.classify(image) for image in images]
    expected_masks = [engine.segment(image) for image in images]

    # Serial preprocessing (executor=None) and the parallel-preprocessing path (executor passed, as
    # in production) must both match the single-image results.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for ex in (None, executor):
            assert engine.classify_batch(images, ex) == expected_flags
            for batched, single in zip(engine.segment_batch(images, ex), expected_masks, strict=True):
                np.testing.assert_array_equal(batched, single)


def test_segment_batch_handles_mixed_native_sizes() -> None:
    """Segmentation runs at native resolution, so ``segment_batch`` must handle differently-sized
    images (no stacking) — each mask comes back at its own H×W and matches per-image ``segment``."""
    engine = _stub_engine(size=8, batch_size=2)
    rng = np.random.default_rng(1)
    # Inputs are square (the guaranteed invariant) but differ in resolution; some below the ROI (8).
    shapes = [(8, 8), (10, 10), (12, 12), (5, 5)]
    images = [rng.random((h, w, 1), dtype=np.float32) for h, w in shapes]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for ex in (None, executor):
            masks = engine.segment_batch(images, ex)
            assert [m.shape for m in masks] == shapes  # each at its own native size
            for batched, image in zip(masks, images, strict=True):
                np.testing.assert_array_equal(batched, engine.segment(image))


# --- orchestration -------------------------------------------------------

# Per-file behaviour is encoded as an integer "code" stored in the loaded array and the produced
# mask, so the stub engine can act per file without seeing paths.
_OK = 1  # nerve, segments and measures fine
_PLAIN = 0  # not a nerve
_SEG_FAIL = 2  # nerve; single-image segment raises (only reachable via segment_batch fallback)
_MORPH_FAIL = 3  # nerve; morphometry raises
_CLS_FAIL = 4  # single-image classify raises (only reachable via classify_batch fallback)

_CODES = {"plain": _PLAIN, "ok": _OK, "segfail": _SEG_FAIL, "morphfail": _MORPH_FAIL, "clsfail": _CLS_FAIL}


def _code(array: np.ndarray) -> int:
    return int(round(float(np.asarray(array).flat[0])))


def _load_factory() -> Callable[[Path], np.ndarray]:
    def load(path: Path) -> np.ndarray:
        kind = path.stem.split("_", 1)[1]
        if kind == "loadfail":
            raise OSError("cannot decode")
        return np.full((2, 2), float(_CODES[kind]), dtype=np.float32)

    return load


class _FakeEngine:
    """Stub engine driven by per-image codes, with switches to force batched-forward failures."""

    def __init__(self, *, fail_classify_batch: bool = False, fail_segment_batch: bool = False) -> None:
        self.fail_classify_batch = fail_classify_batch
        self.fail_segment_batch = fail_segment_batch

    def classify(self, image: np.ndarray) -> bool:
        if _code(image) == _CLS_FAIL:
            raise RuntimeError("classify failed")
        return _code(image) in {_OK, _SEG_FAIL, _MORPH_FAIL}

    def classify_batch(self, images: Sequence[np.ndarray], executor: Executor | None = None) -> list[bool]:
        if self.fail_classify_batch:
            raise RuntimeError("batched classify OOM")
        return [self.classify(image) for image in images]

    def segment(self, image: np.ndarray) -> np.ndarray:
        if _code(image) == _SEG_FAIL:
            raise RuntimeError("segment failed")
        return np.full((1, 1), _code(image), dtype=np.uint8)

    def segment_batch(self, images: Sequence[np.ndarray], executor: Executor | None = None) -> list[np.ndarray]:
        if self.fail_segment_batch:
            raise RuntimeError("batched segment OOM")
        return [self.segment(image) for image in images]

    def morphometry(self, mask: np.ndarray) -> dict[str, float]:
        if _code(mask) == _MORPH_FAIL:
            raise RuntimeError("morphometry failed")
        return {"count": 1.0, "density": 1.0, "area": 2.0, "perimeter": 3.0}


def _run(
    files: list[str], engine: _FakeEngine, *, batch_size: int = 2, max_workers: int = 4
) -> tuple[list[ImageRecord], list[str], list[str], list[int]]:
    progress: list[int] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records, skipped, seg_failed = process_directory(
            [Path(name) for name in files],
            engine=engine,
            load=_load_factory(),
            executor=executor,
            batch_size=batch_size,
            on_progress=progress.append,
        )
    return records, skipped, seg_failed, progress


def test_process_directory_happy_path_classifies_segments_and_measures() -> None:
    files = ["0_ok.tif", "1_plain.tif", "2_ok.tif"]
    records, skipped, seg_failed, progress = _run(files, _FakeEngine())

    assert [r.name for r in records] == files  # order preserved
    assert skipped == [] and seg_failed == []
    nerve = [r for r in records if r.is_nerve]
    assert [r.name for r in nerve] == ["0_ok.tif", "2_ok.tif"]
    assert all(
        r.mask is not None and r.metrics == {"count": 1.0, "density": 1.0, "area": 2.0, "perimeter": 3.0} for r in nerve
    )
    plain = next(r for r in records if r.name == "1_plain.tif")
    assert plain.is_nerve is False and plain.mask is None and plain.metrics is None
    assert progress[-1] == len(files)


def test_process_directory_decode_failure_is_skipped_and_excluded() -> None:
    files = ["0_ok.tif", "1_loadfail.tif", "2_plain.tif"]
    records, skipped, seg_failed, _ = _run(files, _FakeEngine())

    assert skipped == ["1_loadfail.tif"] and seg_failed == []
    assert "1_loadfail.tif" not in [r.name for r in records]
    assert [r.name for r in records] == ["0_ok.tif", "2_plain.tif"]


def test_process_directory_morphometry_failure_reverts_and_reports() -> None:
    files = ["0_morphfail.tif", "1_ok.tif"]
    records, skipped, seg_failed, _ = _run(files, _FakeEngine())

    assert skipped == [] and seg_failed == ["0_morphfail.tif"]
    failed = next(r for r in records if r.name == "0_morphfail.tif")
    assert failed.is_nerve is False and failed.mask is None and failed.metrics is None
    assert next(r for r in records if r.name == "1_ok.tif").metrics is not None


def test_process_directory_classify_batch_fallback_attributes_per_file_skip() -> None:
    files = ["0_ok.tif", "1_clsfail.tif", "2_plain.tif"]
    records, skipped, seg_failed, _ = _run(files, _FakeEngine(fail_classify_batch=True))

    # The whole batched classify raised; the single-image fallback skips only the bad file.
    assert skipped == ["1_clsfail.tif"] and seg_failed == []
    assert [r.name for r in records] == ["0_ok.tif", "2_plain.tif"]
    assert next(r for r in records if r.name == "0_ok.tif").is_nerve is True


def test_process_directory_segment_batch_fallback_isolates_failure() -> None:
    files = ["0_ok.tif", "1_segfail.tif", "2_ok.tif"]
    records, skipped, seg_failed, _ = _run(files, _FakeEngine(fail_segment_batch=True))

    # Batched segment raised; single-image fallback fails only the one file.
    assert skipped == [] and seg_failed == ["1_segfail.tif"]
    failed = next(r for r in records if r.name == "1_segfail.tif")
    assert failed.is_nerve is False and failed.mask is None
    assert all(r.mask is not None for r in records if r.name in {"0_ok.tif", "2_ok.tif"})


def test_process_directory_skipped_follows_input_order_in_mixed_chunk() -> None:
    # A load failure is seen at decode (stage 1) while a classify failure is only seen in the
    # classify fallback (stage 2); `skipped` must still report them in input order, not stage order.
    files = ["0_clsfail.tif", "1_loadfail.tif"]
    _records, skipped, seg_failed, _ = _run(files, _FakeEngine(fail_classify_batch=True), batch_size=2)

    assert skipped == ["0_clsfail.tif", "1_loadfail.tif"] and seg_failed == []


def test_process_directory_seg_failed_follows_input_order_in_mixed_chunk() -> None:
    # A morphometry failure is seen at stage 4 and a segmentation failure at stage 3; `seg_failed`
    # must report them in input order, not stage order.
    files = ["0_morphfail.tif", "1_segfail.tif"]
    _records, skipped, seg_failed, _ = _run(files, _FakeEngine(fail_segment_batch=True), batch_size=2)

    assert skipped == [] and seg_failed == ["0_morphfail.tif", "1_segfail.tif"]


def test_process_directory_no_nerve_chunk_skips_segmentation() -> None:
    # A FakeEngine whose batch methods explode if ever called on an empty list would raise here if
    # the empty-batch guard were missing; segment_batch is simply never reached with no nerves.
    files = ["0_plain.tif", "1_plain.tif"]
    records, skipped, seg_failed, progress = _run(files, _FakeEngine())

    assert skipped == [] and seg_failed == []
    assert all(r.is_nerve is False and r.mask is None for r in records)
    assert progress[-1] == len(files)


def test_process_directory_all_decode_fail_chunk_still_advances_progress() -> None:
    files = ["0_loadfail.tif", "1_loadfail.tif"]  # one full chunk, all failing
    records, skipped, seg_failed, progress = _run(files, _FakeEngine(), batch_size=2)

    assert records == [] and seg_failed == []
    assert skipped == files
    assert progress == [2]  # one chunk, cumulative count reaches len(files)


def test_process_directory_preserves_input_order_under_concurrency() -> None:
    # Many files across several chunks with concurrent workers; order must follow input order.
    files = [f"{i:02d}_ok.tif" for i in range(10)]
    records, skipped, seg_failed, progress = _run(files, _FakeEngine(), batch_size=3, max_workers=4)

    assert [r.name for r in records] == files
    assert skipped == [] and seg_failed == []
    assert progress == [3, 6, 9, 10]  # cumulative per-chunk counts


# --- config validation ---------------------------------------------------


def _model_config() -> ModelConfig:
    return ModelConfig(run_dir="run", ckpt="run/ckpt.ckpt")


def test_app_config_accepts_defaults() -> None:
    config = AppConfig(classification=_model_config(), segmentation=_model_config())
    assert config.batch_size >= 1 and config.num_workers >= 1


@pytest.mark.parametrize(("batch_size", "num_workers"), [(0, 4), (-1, 4), (16, 0), (16, -2)])
def test_app_config_rejects_non_positive_batch_or_workers(batch_size: int, num_workers: int) -> None:
    with pytest.raises(ValueError):
        AppConfig(
            classification=_model_config(),
            segmentation=_model_config(),
            batch_size=batch_size,
            num_workers=num_workers,
        )


@pytest.mark.parametrize("physical_image_size", [0.0, -1.0])
def test_app_config_rejects_non_positive_physical_image_size(physical_image_size: float) -> None:
    with pytest.raises(ValueError):
        AppConfig(
            classification=_model_config(),
            segmentation=_model_config(),
            physical_image_size=physical_image_size,
        )
