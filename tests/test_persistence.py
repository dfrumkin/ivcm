"""Tests for the label-directory persistence layer (Qt-free: no QApplication, no checkpoints)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ivcm.app.persistence import (
    MASKS_DIR,
    SUMMARY_CSV,
    BadLabelFormatError,
    LabelMismatchError,
    is_overwritable_label_dir,
    load_labels,
    mask_filename,
    write_labels,
)
from ivcm.app.records import ImageRecord

_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _write_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    """Write a small grayscale image (``size`` is ``(width, height)``)."""
    Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8), mode="L").save(path)


def _mask(pattern: tuple[tuple[int, ...], ...]) -> np.ndarray:
    return np.array(pattern, dtype=np.uint8)


def _record(image_dir: Path, name: str, *, use: bool, pin: bool = False, mask: np.ndarray | None = None) -> ImageRecord:
    metrics = None if mask is None else {"count": 4.0, "density": 1.5, "area": 2.5, "perimeter": 3.5}
    return ImageRecord(path=image_dir / name, is_nerve=use, pinned=pin, mask=mask, metrics=metrics)


def _make_images(image_dir: Path, names: list[str], size: tuple[int, int] = (8, 8)) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _write_image(image_dir / name, size)


# --- round trip ------------------------------------------------------------


def test_round_trip_reconstructs_records_and_masks(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif", "b.tif", "c.tif"])
    mask = _mask(((1, 0), (0, 1)))  # any {0,1} content; resolution need not equal the image here
    # a: pinned + measured; b: unchecked, no mask; c: used + measured.
    records = [
        _record(image_dir, "a.tif", use=True, pin=True, mask=mask),
        _record(image_dir, "b.tif", use=False),
        _record(image_dir, "c.tif", use=True, mask=mask),
    ]

    label_dir = tmp_path / "labels"
    write_labels(label_dir, records)

    # Masks are written only for masked records, and only for those (b has none).
    assert (label_dir / MASKS_DIR / mask_filename("a.tif")).is_file()
    assert not (label_dir / MASKS_DIR / mask_filename("b.tif")).exists()

    # For load to accept, the image resolution must match the mask resolution (both 2x2 here).
    _make_images(image_dir, ["a.tif", "b.tif", "c.tif"], size=(2, 2))
    loaded, normalized = load_labels(image_dir, label_dir, _EXTS)

    assert normalized is False  # an app-written directory needs no normalization
    by_name = {r.name: r for r in loaded}
    assert by_name["a.tif"].pinned and by_name["a.tif"].is_nerve
    assert by_name["a.tif"].metrics == {"count": 4.0, "density": 1.5, "area": 2.5, "perimeter": 3.5}
    np.testing.assert_array_equal(by_name["a.tif"].mask, mask)
    assert by_name["b.tif"].mask is None and by_name["b.tif"].metrics is None
    assert not by_name["b.tif"].is_nerve


def test_mask_encoding_round_trips_zero_one(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(3, 3))
    mask = _mask(((0, 1, 0), (1, 1, 1), (0, 0, 1)))
    write_labels(tmp_path / "labels", [_record(image_dir, "a.tif", use=True, mask=mask)])

    # The stored PNG is the 0/255 binary convention.
    stored = np.asarray(Image.open(tmp_path / "labels" / MASKS_DIR / mask_filename("a.tif")).convert("L"))
    assert set(np.unique(stored).tolist()) == {0, 255}

    loaded, _ = load_labels(image_dir, tmp_path / "labels", _EXTS)
    loaded_mask = loaded[0].mask
    assert loaded_mask is not None
    np.testing.assert_array_equal(loaded_mask, mask)
    assert set(np.unique(loaded_mask).tolist()) <= {0, 1}


def test_mixed_extension_masks_do_not_collide(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["foo.tif", "foo.png"], size=(2, 2))
    tif_mask = _mask(((1, 0), (0, 0)))
    png_mask = _mask(((0, 0), (0, 1)))
    write_labels(
        tmp_path / "labels",
        [
            _record(image_dir, "foo.tif", use=True, mask=tif_mask),
            _record(image_dir, "foo.png", use=True, mask=png_mask),
        ],
    )

    loaded, _ = load_labels(image_dir, tmp_path / "labels", _EXTS)
    by_name = {r.name: r for r in loaded}
    np.testing.assert_array_equal(by_name["foo.tif"].mask, tif_mask)
    np.testing.assert_array_equal(by_name["foo.png"].mask, png_mask)


# --- helpers to build a label dir directly for the failure tests -----------


def _write_summary(label_dir: Path, rows: list[list[object]]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / MASKS_DIR).mkdir(exist_ok=True)
    with (label_dir / SUMMARY_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Pin", "Use", "Name", "Count", "Density", "Area", "Perimeter"])
        writer.writerows(rows)


# --- bad format ------------------------------------------------------------


def test_missing_summary_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    (label_dir / MASKS_DIR).mkdir(parents=True)
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_missing_masks_dir_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / SUMMARY_CSV).write_text("Pin,Use,Name,Count,Density,Area,Perimeter\n")
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_wrong_header_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    (label_dir / MASKS_DIR).mkdir(parents=True)
    (label_dir / SUMMARY_CSV).write_text("Pin,Use,Name\nFalse,True,a.tif\n")
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_partial_metric_triple_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "True", "a.tif", "1.0", "", "", ""]])  # count only
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


@pytest.mark.parametrize("name", ["../a.tif", "sub/a.tif", "a\\b.tif", "..", "."])
def test_path_like_name_is_bad_format(tmp_path: Path, name: str) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", name, "", "", "", ""]])
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_metrics_without_mask_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "True", "a.tif", "4.0", "1.0", "2.0", "3.0"]])  # metrics, no mask file
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_mask_without_metrics_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "True", "a.tif", "", "", "", ""]])  # no metrics
    _write_image(label_dir / MASKS_DIR / mask_filename("a.tif"), size=(2, 2))  # but a mask file exists
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_non_png_in_masks_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "a.tif", "", "", "", ""]])
    (label_dir / MASKS_DIR / "stray.txt").write_text("not a mask")
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


def test_corrupt_mask_is_bad_format(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "True", "a.tif", "4.0", "1.0", "2.0", "3.0"]])
    (label_dir / MASKS_DIR / mask_filename("a.tif")).write_bytes(b"not a real png")
    with pytest.raises(BadLabelFormatError):
        load_labels(image_dir, label_dir, _EXTS)


# --- mismatch --------------------------------------------------------------


def test_unknown_name_is_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"])
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "ghost.tif", "", "", "", ""]])
    with pytest.raises(LabelMismatchError):
        load_labels(image_dir, label_dir, _EXTS)


def test_dangling_mask_is_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "a.tif", "", "", "", ""]])
    _write_image(label_dir / MASKS_DIR / mask_filename("b.tif"), size=(2, 2))  # no row for b
    with pytest.raises(LabelMismatchError):
        load_labels(image_dir, label_dir, _EXTS)


def test_mask_resolution_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(4, 4))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "True", "a.tif", "4.0", "1.0", "2.0", "3.0"]])
    _write_image(label_dir / MASKS_DIR / mask_filename("a.tif"), size=(2, 2))  # wrong size
    with pytest.raises(LabelMismatchError):
        load_labels(image_dir, label_dir, _EXTS)


def test_non_square_referenced_image_is_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(4, 8))  # non-square
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "a.tif", "", "", "", ""]])
    with pytest.raises(LabelMismatchError):
        load_labels(image_dir, label_dir, _EXTS)


def test_unreadable_referenced_image_is_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "a.tif").write_bytes(b"not an image")  # accepted extension, but corrupt
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "a.tif", "", "", "", ""]])
    with pytest.raises(LabelMismatchError):
        load_labels(image_dir, label_dir, _EXTS)


def test_image_absent_from_csv_loads_fine(tmp_path: Path) -> None:
    # A skipped image (present on disk, no CSV row) is allowed and simply not loaded.
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif", "extra.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "a.tif", "", "", "", ""]])
    records, normalized = load_labels(image_dir, label_dir, _EXTS)
    assert [r.name for r in records] == ["a.tif"]
    assert normalized is False


def test_ds_store_inside_masks_is_ignored(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["False", "False", "a.tif", "", "", "", ""]])
    (label_dir / MASKS_DIR / ".DS_Store").write_bytes(b"finder junk")
    records, _ = load_labels(image_dir, label_dir, _EXTS)
    assert [r.name for r in records] == ["a.tif"]


# --- normalizations --------------------------------------------------------


def test_pin_without_metrics_loads_unpinned_and_normalized(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["True", "True", "a.tif", "", "", "", ""]])  # pinned but unmeasured
    records, normalized = load_labels(image_dir, label_dir, _EXTS)
    assert normalized is True
    assert records[0].pinned is False


def test_pin_forces_use_and_normalized(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["True", "False", "a.tif", "4.0", "1.0", "2.0", "3.0"]])
    _write_image(label_dir / MASKS_DIR / mask_filename("a.tif"), size=(2, 2))
    records, normalized = load_labels(image_dir, label_dir, _EXTS)
    assert normalized is True
    assert records[0].pinned and records[0].is_nerve


def test_more_than_three_pins_are_capped(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    names = [f"{i}.tif" for i in range(4)]
    _make_images(image_dir, names, size=(2, 2))
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [["True", "True", name, "4.0", "1.0", "2.0", "3.0"] for name in names])
    for name in names:
        _write_image(label_dir / MASKS_DIR / mask_filename(name), size=(2, 2))
    records, normalized = load_labels(image_dir, label_dir, _EXTS)
    assert normalized is True
    assert sum(1 for r in records if r.pinned) == 3
    # The first three (in file order) stay pinned.
    assert [r.pinned for r in sorted(records, key=lambda r: r.name)] == [True, True, True, False]


# --- overwrite safety ------------------------------------------------------


def test_is_overwritable_accepts_our_shape(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels"
    _make_images(tmp_path / "images", ["a.tif"], size=(2, 2))
    write_labels(label_dir, [_record(tmp_path / "images", "a.tif", use=True, mask=_mask(((1, 0), (0, 1))))])
    assert is_overwritable_label_dir(label_dir) is True
    (label_dir / ".DS_Store").write_bytes(b"junk")  # a system file at the root is tolerated
    assert is_overwritable_label_dir(label_dir) is True


def test_is_overwritable_rejects_extra_root_file(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [])
    (label_dir / "notes.txt").write_text("hello")
    assert is_overwritable_label_dir(label_dir) is False


def test_is_overwritable_rejects_missing_masks(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / SUMMARY_CSV).write_text("Pin,Use,Name,Count,Density,Area,Perimeter\n")
    assert is_overwritable_label_dir(label_dir) is False


def test_is_overwritable_rejects_non_png_in_masks(tmp_path: Path) -> None:
    label_dir = tmp_path / "labels"
    _write_summary(label_dir, [])
    (label_dir / MASKS_DIR / "note.txt").write_text("x")
    assert is_overwritable_label_dir(label_dir) is False


# --- failure-safe write ----------------------------------------------------


def test_write_failure_preserves_existing_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    good = _record(image_dir, "a.tif", use=True, mask=_mask(((1, 1), (1, 1))))
    write_labels(label_dir, [good])  # a first, good save

    # Force the *populating* step to fail on the next write (while writing into the temp dir).
    import ivcm.app.persistence as persistence

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(persistence, "_write_mask", _boom)
    with pytest.raises(OSError):
        write_labels(label_dir, [_record(image_dir, "a.tif", use=True, mask=_mask(((0, 0), (0, 0))))])

    # The previous save is fully intact, and no stray temp sibling survives.
    loaded, _ = load_labels(image_dir, label_dir, _EXTS)
    np.testing.assert_array_equal(loaded[0].mask, _mask(((1, 1), (1, 1))))
    siblings = [p.name for p in tmp_path.iterdir() if p.name not in {"images", "labels"}]
    assert siblings == []


def test_write_failure_during_swap_recovers_previous_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    write_labels(label_dir, [_record(image_dir, "a.tif", use=True, mask=_mask(((1, 1), (1, 1))))])

    # Fail the second rename (the tmp_new -> directory swap), *after* the old save has been moved
    # aside, so the rollback path (renaming the backup back into place) is exercised.
    real_rename = Path.rename
    calls = {"n": 0}

    def flaky_rename(self: Path, target: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("swap failed")
        return real_rename(self, target)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(OSError):
        write_labels(label_dir, [_record(image_dir, "a.tif", use=True, mask=_mask(((0, 0), (0, 0))))])
    monkeypatch.undo()

    # The previous save was rolled back into place, and no stray temp sibling survives.
    loaded, _ = load_labels(image_dir, label_dir, _EXTS)
    np.testing.assert_array_equal(loaded[0].mask, _mask(((1, 1), (1, 1))))
    siblings = [p.name for p in tmp_path.iterdir() if p.name not in {"images", "labels"}]
    assert siblings == []


def test_overwrite_replaces_contents(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif", "b.tif"], size=(2, 2))
    label_dir = tmp_path / "labels"
    write_labels(label_dir, [_record(image_dir, "a.tif", use=True, mask=_mask(((1, 0), (0, 0))))])
    assert (label_dir / MASKS_DIR / mask_filename("a.tif")).is_file()

    # Re-save a different record set: the old mask must be gone (no stale files).
    write_labels(label_dir, [_record(image_dir, "b.tif", use=True, mask=_mask(((0, 1), (0, 0))))])
    assert not (label_dir / MASKS_DIR / mask_filename("a.tif")).exists()
    assert (label_dir / MASKS_DIR / mask_filename("b.tif")).is_file()


def test_write_rejects_mask_metrics_coupling_violation(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _make_images(image_dir, ["a.tif"], size=(2, 2))
    bad = ImageRecord(path=image_dir / "a.tif", is_nerve=True, mask=_mask(((1, 0), (0, 0))), metrics=None)
    with pytest.raises(ValueError):
        write_labels(tmp_path / "labels", [bad])
