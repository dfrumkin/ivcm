"""On-disk persistence of the review results (the table and the masks); Qt-free.

A *label directory* holds exactly a ``summary.csv`` file and a ``masks/`` directory:

- ``summary.csv`` has one row per :class:`~ivcm.app.records.ImageRecord`, columns
  ``Pin,Use,Name,Density,Area,Perimeter`` (booleans as ``True``/``False``; metrics at full float
  precision or empty when the record has no metrics).
- ``masks/`` holds one single-channel PNG per record that has a mask, named ``<image-name>.png``
  (the full image file name with ``.png`` appended, so mixed extensions never collide). Masks are
  the standard binary convention -- ``uint8`` mode-``L``, ``0`` background / ``255`` foreground --
  while in memory they stay ``{0, 1}``.

:func:`write_labels` writes failure-safely (into a temp sibling, then a staged swap).
:func:`load_labels` validates the directory against the image directory and reconstructs the
records, raising :class:`BadLabelFormatError` (structural problem with the label dir itself) or
:class:`LabelMismatchError` (the label dir does not correspond to the images).
:func:`is_overwritable_label_dir` is the lighter shape-only check that gates a destructive overwrite.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ivcm.app.records import MAX_PINNED, ImageRecord

SUMMARY_CSV = "summary.csv"
MASKS_DIR = "masks"
CSV_HEADER = ["Pin", "Use", "Name", "Density", "Area", "Perimeter"]
_METRIC_KEYS = ("density", "area", "perimeter")
_PNG_SUFFIX = ".png"

# System files that may sit alongside our own files without meaning the directory is "not ours":
# macOS Finder metadata, Windows thumbnail caches, and AppleDouble resource forks.
_SYSTEM_FILES = frozenset({".DS_Store", "Thumbs.db"})


class BadLabelFormatError(Exception):
    """The label directory is structurally invalid (missing/malformed ``summary.csv`` or ``masks/``)."""


class LabelMismatchError(Exception):
    """The label directory does not correspond to the image directory (names / resolutions)."""


def _is_system_file(name: str) -> bool:
    """Whether ``name`` is an ignorable system file (Finder/Windows metadata, AppleDouble fork)."""
    return name in _SYSTEM_FILES or name.startswith("._")


def mask_filename(image_name: str) -> str:
    """The mask file name for an image: the full image name with ``.png`` appended (collision-free)."""
    return image_name + _PNG_SUFFIX


def image_name_from_mask(mask_name: str) -> str | None:
    """Invert :func:`mask_filename` by stripping the trailing ``.png``; ``None`` if it has none."""
    if mask_name.endswith(_PNG_SUFFIX):
        return mask_name[: -len(_PNG_SUFFIX)]
    return None


# --- writing ---------------------------------------------------------------


def _check_coupling(records: Sequence[ImageRecord]) -> None:
    """Fail fast if any record has exactly one of ``mask``/``metrics`` (the invariant forbids it)."""
    for record in records:
        if (record.mask is None) != (record.metrics is None):
            raise ValueError(
                f"record {record.name!r} violates the mask/metrics coupling (both must be set or both unset)"
            )


def _summary_row(record: ImageRecord) -> list[object]:
    """One ``summary.csv`` row: Pin, Use, Name, and the three metrics (or empty when unmeasured)."""
    metrics = record.metrics
    cells: list[object] = ["", "", ""] if metrics is None else [metrics[key] for key in _METRIC_KEYS]
    return [record.pinned, record.is_nerve, record.name, *cells]


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a ``{0, 1}`` mask as a single-channel ``0``/``255`` PNG (the binary-mask convention)."""
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(path)


def _populate(target: Path, records: Sequence[ImageRecord]) -> None:
    """Write ``summary.csv`` and ``masks/`` (one PNG per masked record) into ``target``."""
    target.mkdir(exist_ok=True)
    masks_dir = target / MASKS_DIR
    masks_dir.mkdir()
    with (target / SUMMARY_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for record in records:
            writer.writerow(_summary_row(record))
    for record in records:
        if record.mask is not None:
            _write_mask(masks_dir / mask_filename(record.name), record.mask)


def _reserve_sibling(parent: Path, prefix: str) -> Path:
    """A unique, currently-nonexistent sibling path under ``parent`` (to rename a directory onto)."""
    reserved = tempfile.mkdtemp(dir=parent, prefix=prefix)
    os.rmdir(reserved)  # mkdtemp created it; drop it so we can rename another directory onto the name
    return Path(reserved)


def write_labels(directory: Path, records: Sequence[ImageRecord]) -> None:
    """Write ``records`` as a label directory at ``directory``, failure-safely and cleanly.

    The full directory is built in a uniquely named temp sibling and only then swapped into place
    (moving any existing ``directory`` aside first, and rolling it back on error), so a mid-write
    failure never destroys a previously good save and the result never contains stale files. Only the
    two temp directories this call creates are ever removed. Raises ``ValueError`` on a record that
    violates the mask/metrics coupling (an internal invariant that should never occur).
    """
    _check_coupling(records)
    directory = Path(directory)
    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)

    tmp_new = Path(tempfile.mkdtemp(dir=parent, prefix=".ivcm_new_"))
    tmp_old: Path | None = None
    committed = False
    try:
        _populate(tmp_new, records)
        if directory.exists():
            tmp_old = _reserve_sibling(parent, ".ivcm_old_")
            directory.rename(tmp_old)
        tmp_new.rename(directory)
        committed = True
    except Exception:
        # Roll the previous save back into place if it was moved aside and the swap did not complete.
        if not committed and tmp_old is not None and not directory.exists():
            try:
                tmp_old.rename(directory)
                tmp_old = None
            except OSError:
                pass  # cannot roll back: leave the old save recoverable as the tmp_old sibling
        raise
    finally:
        if tmp_new.exists():
            shutil.rmtree(tmp_new, ignore_errors=True)
        if committed and tmp_old is not None:
            shutil.rmtree(tmp_old, ignore_errors=True)  # best-effort: the save already succeeded


# --- overwrite-safety check ------------------------------------------------


def is_overwritable_label_dir(directory: Path) -> bool:
    """Whether ``directory`` has exactly our shape (``summary.csv`` + ``masks/`` of ``*.png``).

    Ignores the system-file allowlist at both levels. This is a shape-only check (it does not
    validate CSV content or mask↔image correspondence): it only confirms the directory is one we
    created, so overwriting it -- which deletes the whole subtree -- is safe.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return False
    entries = {entry.name for entry in directory.iterdir() if not _is_system_file(entry.name)}
    if entries != {SUMMARY_CSV, MASKS_DIR}:
        return False
    if not (directory / SUMMARY_CSV).is_file() or not (directory / MASKS_DIR).is_dir():
        return False
    for entry in (directory / MASKS_DIR).iterdir():
        if _is_system_file(entry.name):
            continue
        if not entry.is_file() or not entry.name.endswith(_PNG_SUFFIX):
            return False
    return True


# --- loading ---------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    """One parsed ``summary.csv`` row (before correspondence checks and normalization)."""

    pin: bool
    use: bool
    name: str
    metrics: dict[str, float] | None


def _parse_bool(value: str, row_number: int, column: str) -> bool:
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise BadLabelFormatError(f"row {row_number}: {column} is not a boolean: {value!r}")


def _is_bare_filename(name: str) -> bool:
    """Whether ``name`` is a plain filename (no path components, no traversal), platform-independently."""
    return name not in {"", ".", ".."} and "/" not in name and "\\" not in name and name == Path(name).name


def _parse_metrics(cells: Sequence[str], row_number: int) -> dict[str, float] | None:
    """The three metric cells as a dict, ``None`` if all empty, or a ``BadLabelFormatError`` if partial."""
    present = [cell.strip() != "" for cell in cells]
    if not any(present):
        return None
    if not all(present):
        raise BadLabelFormatError(f"row {row_number}: metrics are partially filled (all or none required)")
    try:
        values = [float(cell) for cell in cells]
    except ValueError as exc:
        raise BadLabelFormatError(f"row {row_number}: metric is not a number") from exc
    return dict(zip(_METRIC_KEYS, values, strict=True))


def _read_summary(path: Path) -> list[_Row]:
    """Parse and structurally validate ``summary.csv`` into rows (raises ``BadLabelFormatError``)."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
    except OSError as exc:
        raise BadLabelFormatError(f"cannot read {SUMMARY_CSV}") from exc
    if not table:
        raise BadLabelFormatError(f"{SUMMARY_CSV} is empty")
    header, *data = table
    if header != CSV_HEADER:
        raise BadLabelFormatError(f"unexpected {SUMMARY_CSV} header: {header!r}")
    rows: list[_Row] = []
    for offset, cells in enumerate(data):
        row_number = offset + 2  # 1-based, after the header
        if len(cells) != len(CSV_HEADER):
            raise BadLabelFormatError(f"row {row_number} has {len(cells)} columns, expected {len(CSV_HEADER)}")
        pin = _parse_bool(cells[0], row_number, "Pin")
        use = _parse_bool(cells[1], row_number, "Use")
        name = cells[2]
        if not _is_bare_filename(name):
            raise BadLabelFormatError(f"row {row_number}: invalid Name {name!r}")
        metrics = _parse_metrics(cells[3:6], row_number)
        rows.append(_Row(pin=pin, use=use, name=name, metrics=metrics))
    return rows


def _list_mask_files(masks_dir: Path) -> set[str]:
    """Names of the mask files under ``masks/``; a non-system non-``.png`` entry is a bad format."""
    names: set[str] = set()
    for entry in masks_dir.iterdir():
        if _is_system_file(entry.name):
            continue
        if not entry.is_file() or not entry.name.endswith(_PNG_SUFFIX):
            raise BadLabelFormatError(f"{MASKS_DIR}/ contains a non-PNG entry: {entry.name!r}")
        names.add(entry.name)
    return names


def _accepted_images(image_dir: Path, accepted_extensions: Sequence[str]) -> dict[str, Path]:
    """Map of image file name -> path for the accepted images in ``image_dir`` (matches the create path)."""
    extensions = tuple(ext.lower() for ext in accepted_extensions)
    return {path.name: path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions}


def _image_size(path: Path) -> tuple[int, int]:
    """The ``(width, height)`` of an image via a header read; an unreadable image is a mismatch."""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception as exc:  # unreadable/corrupt referenced image -> the image side is at fault
        raise LabelMismatchError(f"cannot open image {path.name}") from exc


def _mask_size(path: Path) -> tuple[int, int]:
    """The ``(width, height)`` of a mask PNG; an unreadable mask is a bad format (our own file)."""
    try:
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        raise BadLabelFormatError(f"cannot open mask {path.name}") from exc


def _load_mask(path: Path) -> np.ndarray:
    """A mask PNG as a 2D ``{0, 1}`` ``uint8`` array (single-channel, so never an unexpected 3D array)."""
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
    except Exception as exc:
        raise BadLabelFormatError(f"cannot decode mask {path.name}") from exc
    return (array > 0).astype(np.uint8)


def _normalize(rows: list[_Row]) -> tuple[list[tuple[_Row, bool, bool]], bool]:
    """Apply the pin/use normalizations, returning ``(row, pinned, is_nerve)`` and a changed flag.

    In order: a pinned row with no metrics is unpinned; ``Pin`` otherwise forces ``Use``; at most
    :data:`~ivcm.app.records.MAX_PINNED` rows stay pinned (extras, in file order, are unpinned).
    """
    result: list[tuple[_Row, bool, bool]] = []
    normalized = False
    pin_count = 0
    for row in rows:
        pinned = row.pin
        is_nerve = row.use
        if pinned and row.metrics is None:
            pinned = False
            normalized = True
        if pinned and not is_nerve:
            is_nerve = True
            normalized = True
        if pinned:
            if pin_count >= MAX_PINNED:
                pinned = False
                normalized = True
            else:
                pin_count += 1
        result.append((row, pinned, is_nerve))
    return result, normalized


def load_labels(image_dir: Path, label_dir: Path, accepted_extensions: Sequence[str]) -> tuple[list[ImageRecord], bool]:
    """Load ``label_dir`` against ``image_dir`` into records, validating structure and correspondence.

    Returns ``(records, normalized)`` where ``normalized`` is whether any normalization changed a row
    (so a byte-faithful file loads clean). Raises :class:`BadLabelFormatError` for a structural
    problem with the label dir, or :class:`LabelMismatchError` when it does not correspond to the
    images (unknown name, dangling mask, wrong or non-square resolution, unreadable image).
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    summary_path = label_dir / SUMMARY_CSV
    masks_dir = label_dir / MASKS_DIR
    if not summary_path.is_file():
        raise BadLabelFormatError(f"missing {SUMMARY_CSV}")
    if not masks_dir.is_dir():
        raise BadLabelFormatError(f"missing {MASKS_DIR}/ directory")

    rows = _read_summary(summary_path)
    mask_files = _list_mask_files(masks_dir)

    # Structural: a row has its mask file iff it has metrics (mask and metrics are coupled).
    for row in rows:
        has_mask_file = mask_filename(row.name) in mask_files
        if has_mask_file != (row.metrics is not None):
            raise BadLabelFormatError(
                f"row for {row.name!r}: a mask file and metrics must both be present or both absent"
            )

    # Correspondence: unique names, each an accepted image; each mask maps to a known row.
    accepted = _accepted_images(image_dir, accepted_extensions)
    seen: set[str] = set()
    for row in rows:
        if row.name in seen:
            raise LabelMismatchError(f"duplicate Name {row.name!r} in {SUMMARY_CSV}")
        seen.add(row.name)
        if row.name not in accepted:
            raise LabelMismatchError(f"{row.name!r} is not an image file in {image_dir}")
    for mask_name in mask_files:
        image_name = image_name_from_mask(mask_name)
        if image_name is None or image_name not in seen:
            raise LabelMismatchError(f"mask {mask_name!r} has no corresponding row")

    # Every referenced image must be square; every mask must match its image's resolution.
    for row in rows:
        width, height = _image_size(accepted[row.name])
        if width != height:
            raise LabelMismatchError(f"{row.name!r} is not square ({width}x{height})")
        if row.metrics is not None:
            if _mask_size(masks_dir / mask_filename(row.name)) != (width, height):
                raise LabelMismatchError(f"mask for {row.name!r} does not match the image resolution")

    normalized_rows, normalized = _normalize(rows)
    records = [
        ImageRecord(
            path=image_dir / row.name,
            is_nerve=is_nerve,
            pinned=pinned,
            mask=(_load_mask(masks_dir / mask_filename(row.name)) if row.metrics is not None else None),
            metrics=row.metrics,
        )
        for row, pinned, is_nerve in normalized_rows
    ]
    return records, normalized
