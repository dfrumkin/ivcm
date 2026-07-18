"""Qt-free directory-batch orchestration for the clinical-review UI.

``process_directory`` runs the per-directory pipeline that ``MainWindow`` used to inline: it
processes the accepted image files in chunks, decoding/preprocessing in parallel and running
classification and segmentation in batches, then computes morphometry per nerve. It is Qt-free and
takes its model access through an ``Engine`` protocol and its decode through a ``load`` callable, so
the failure-attribution contract is unit-testable with stubs (no ``QApplication``, no checkpoints).

Failure attribution mirrors the old serial ``MainWindow._process_file`` exactly:

- a file that fails to **load/classify** is dropped and named in ``skipped``;
- a file that fails to **segment or measure** is reverted to unchecked (``is_nerve=False``,
  ``mask=None``, ``metrics=None``) and named in ``seg_failed``.

The common success path runs in batches; the single-image fallbacks (used only when a *batched*
forward raises, e.g. on device OOM) preserve per-file attribution. All parallel stages use a "safe"
wrapper that catches per-item exceptions so one bad file never aborts collection of the rest, and
``concurrent.futures.Executor.map`` preserves input order, so ``records`` / ``skipped`` /
``seg_failed`` follow input file order.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ivcm.app.records import ImageRecord
from ivcm.data import ensure_square

log = logging.getLogger(__name__)


def ensure_all_square(files: Sequence[Path]) -> None:
    """Validate that every file is a square image, before the directory batch runs.

    Reads each file's pixel dimensions with a lazy header read (``PIL.Image.open(path).size``, no
    full decode) and raises ``NonSquareImageError`` naming the first **confirmed** non-square file.
    ``PIL``'s ``.size`` is ``(width, height)``; it is normalised to ``(height, width)`` so the error
    message matches the array-shape-based checks elsewhere.

    Only non-square images are fatal here: files that fail to open/decode during the dimension read
    are ignored (left to ``process_directory``, which reports them in ``skipped``), preserving the
    existing per-file failure contract. Running this before the batch is what makes a non-square
    input a hard error instead of being swallowed as a skipped file.
    """
    for path in files:
        try:
            with Image.open(path) as image:  # context manager closes the handle (no fd leak on a big dir)
                width, height = image.size
        except Exception:  # unreadable/corrupt: not fatal here — process_directory will skip it
            continue
        ensure_square((height, width), source=path.name)


class Engine(Protocol):
    """The inference surface ``process_directory`` needs (satisfied by ``InferenceEngine``)."""

    def classify_batch(self, images: Sequence[np.ndarray], executor: Executor | None = ...) -> list[bool]: ...
    def segment_batch(self, images: Sequence[np.ndarray], executor: Executor | None = ...) -> list[np.ndarray]: ...
    def classify(self, image: np.ndarray) -> bool: ...
    def segment(self, image: np.ndarray) -> np.ndarray: ...
    def morphometry(self, mask: np.ndarray) -> dict[str, float]: ...


def _safe_load(path: Path, load: Callable[[Path], np.ndarray]) -> tuple[Path, np.ndarray | None]:
    """Decode one file, returning ``(path, None)`` (and logging) on any failure."""
    try:
        return path, load(path)
    except Exception:
        log.exception("Skipping %s (load/classify failed)", path.name)
        return path, None


def _classify_survivors(
    engine: Engine,
    indices: Sequence[int],
    paths: Sequence[Path],
    images: Sequence[np.ndarray],
    executor: Executor,
    skipped: list[tuple[int, str]],
) -> list[tuple[int, Path, np.ndarray, bool]]:
    """Classify a chunk's decoded images, returning ``(index, path, image, is_nerve)`` per survivor.

    ``indices`` are the survivors' global file indices, carried so any failure can be recorded with
    its index (``skipped`` holds ``(index, name)`` and is sorted back into input order at the end).

    The common path is one batched forward. If it raises (a non-per-image failure, e.g. device
    OOM), fall back to single-image ``classify`` so a per-file failure can be attributed to
    ``skipped`` rather than failing the whole chunk. The fallback runs **after** the ``except``
    block (gated on a flag), not inside it, so the failed batch's exception traceback — which pins
    the stacked-input / device-chunk tensors live in the ``classify_batch`` / ``forward_in_chunks``
    frames — is released before the single-image retry runs (important for OOM recovery).
    """
    flags: list[bool] | None
    try:
        flags = engine.classify_batch(images, executor)
    except Exception:
        log.exception("Batched classification failed; falling back to single-image")
        flags = None
    if flags is not None:
        return list(zip(indices, paths, images, flags, strict=True))

    survivors: list[tuple[int, Path, np.ndarray, bool]] = []
    for index, path, image in zip(indices, paths, images, strict=True):
        try:
            survivors.append((index, path, image, engine.classify(image)))
        except Exception:
            log.exception("Skipping %s (load/classify failed)", path.name)
            skipped.append((index, path.name))
    return survivors


def _segment_masks(
    engine: Engine,
    paths: Sequence[Path],
    images: Sequence[np.ndarray],
    executor: Executor,
) -> list[np.ndarray | None]:
    """Segment the nerve subset, returning one mask per image (``None`` for a per-image failure).

    The common path is one batched forward; on a batched failure, fall back to single-image
    ``segment`` so a single bad image yields ``None`` while its peers still segment. As in
    ``_classify_survivors``, the fallback runs **after** the ``except`` block so the failed batch's
    traceback (and the device tensors its frames pin) is released before the single-image retry.
    """
    try:
        return list(engine.segment_batch(images, executor))
    except Exception:
        log.exception("Batched segmentation failed; falling back to single-image")

    masks: list[np.ndarray | None] = []
    for path, image in zip(paths, images, strict=True):
        try:
            masks.append(engine.segment(image))
        except Exception:
            log.exception("Segmentation failed for %s; leaving unchecked", path.name)
            masks.append(None)
    return masks


def _safe_morphometry(engine: Engine, job: tuple[np.ndarray, str]) -> dict[str, float] | None:
    """Measure one ``(mask, name)`` job, returning ``None`` (and logging) on failure.

    Runs in a worker thread; takes the mask and name as one tuple so it maps cleanly over an
    ``Executor.map`` of jobs.
    """
    mask, name = job
    try:
        return engine.morphometry(mask)
    except Exception:
        log.exception("Segmentation failed for %s; leaving unchecked", name)
        return None


def _revert(record: ImageRecord, index: int, seg_failed: list[tuple[int, str]]) -> None:
    """Revert a record whose segmentation/morphometry failed, matching the old serial path.

    Records ``(index, name)`` so ``seg_failed`` can be sorted back into input order at the end.
    """
    record.is_nerve = False
    record.mask = None
    record.metrics = None
    seg_failed.append((index, record.name))


def process_directory(
    files: Sequence[Path],
    *,
    engine: Engine,
    load: Callable[[Path], np.ndarray],
    executor: Executor,
    batch_size: int,
    on_progress: Callable[[int], None],
) -> tuple[list[ImageRecord], list[str], list[str]]:
    """Classify (and segment/measure if nerve) every file, in parallel-decoded, batched chunks.

    Returns ``(records, skipped, seg_failed)``: the kept records (one per loaded+classified file),
    the names of files that failed to load/classify, and the names of nerve files whose
    segmentation or morphometry failed (left unchecked). All three follow input file order — the
    failure lists are collected with each file's global index (since within a chunk load failures
    are seen before classify failures, and segmentation before morphometry) and sorted by it at the
    end, so the reporting matches the old serial path regardless of stage. ``on_progress`` is called
    once per chunk with the cumulative number of files processed (reaching ``len(files)``), only
    from this (orchestration) thread — never from a worker task.
    """
    records: list[ImageRecord] = []
    skipped: list[tuple[int, str]] = []
    seg_failed: list[tuple[int, str]] = []
    completed = 0

    for start in range(0, len(files), batch_size):
        chunk = files[start : start + batch_size]

        # 1. Parallel decode; load failures are dropped to `skipped` (tagged with their file index).
        indices: list[int] = []
        paths: list[Path] = []
        images: list[np.ndarray] = []
        for offset, (path, image) in enumerate(executor.map(partial(_safe_load, load=load), chunk)):
            if image is None:
                skipped.append((start + offset, path.name))
            else:
                indices.append(start + offset)
                paths.append(path)
                images.append(image)

        # 2. Batched classify (with per-file fallback). Empty chunk -> no survivors.
        survivors = _classify_survivors(engine, indices, paths, images, executor, skipped) if images else []
        chunk_records = [ImageRecord(path=path, is_nerve=is_nerve) for _i, path, _image, is_nerve in survivors]
        image_of = {record: image for record, (_i, _p, image, _n) in zip(chunk_records, survivors, strict=True)}
        index_of = {record: index for record, (index, _p, _img, _n) in zip(chunk_records, survivors, strict=True)}

        # 3. Batched segment the nerve subset (with per-file fallback); failures revert + log.
        nerve_records = [record for record in chunk_records if record.is_nerve]
        if nerve_records:
            nerve_paths = [record.path for record in nerve_records]
            masks = _segment_masks(engine, nerve_paths, [image_of[r] for r in nerve_records], executor)
            for record, mask in zip(nerve_records, masks, strict=True):
                if mask is None:
                    _revert(record, index_of[record], seg_failed)
                else:
                    record.mask = mask

            # 4. Parallel morphometry over the successfully-segmented masks; failures revert.
            segmented = [record for record in nerve_records if record.mask is not None]
            if segmented:
                jobs = [(record.mask, record.name) for record in segmented if record.mask is not None]
                metrics = executor.map(partial(_safe_morphometry, engine), jobs)
                for record, result in zip(segmented, metrics, strict=True):
                    if result is None:
                        _revert(record, index_of[record], seg_failed)
                    else:
                        record.metrics = result

        # 5. Commit the chunk and advance progress (once per chunk, including all-failed chunks).
        records.extend(chunk_records)
        completed += len(chunk)
        on_progress(completed)

    # Sort the failure reports back into input file order (see the docstring).
    skipped_names = [name for _index, name in sorted(skipped)]
    seg_failed_names = [name for _index, name in sorted(seg_failed)]
    return records, skipped_names, seg_failed_names
