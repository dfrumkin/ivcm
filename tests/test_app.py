"""Pure-logic tests for the clinical-review UI (no QApplication, no checkpoints, no Qt)."""

from __future__ import annotations

import io
from importlib import resources
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ivcm.app.inference import mask_from_seg_logits, predicted_class
from ivcm.app.records import ImageRecord, selection_after_toggle, sort_key, sorted_records, top3
from ivcm.app.render import overlay


def _rec(
    name: str,
    is_nerve: bool,
    density: float | None = None,
    area: float | None = None,
    perimeter: float | None = None,
) -> ImageRecord:
    metrics = None
    if density is not None:
        metrics = {"density": density, "area": area or 0.0, "perimeter": perimeter or 0.0}
    return ImageRecord(path=Path(name), is_nerve=is_nerve, metrics=metrics)


def test_sort_key_orders_nerve_first_then_metrics_then_name() -> None:
    a = _rec("a.tif", True, density=5, area=1, perimeter=1)
    b = _rec("b.tif", True, density=10, area=1, perimeter=1)
    e = _rec("e.tif", True, density=10, area=2, perimeter=1)  # higher area breaks the density tie
    c = _rec("c.tif", False)
    d = _rec("d.tif", False)

    order = [r.name for r in sorted_records([a, b, c, d, e])]

    # nerve rows first (by density desc, area desc as tiebreak), then unchecked rows by name.
    assert order == ["e.tif", "b.tif", "a.tif", "c.tif", "d.tif"]


def test_sort_key_unchecked_rows_order_by_name_below_nerve() -> None:
    nerve = _rec("z.tif", True, density=1, area=1, perimeter=1)
    x = _rec("x.tif", False)
    y = _rec("y.tif", False)

    order = [r.name for r in sorted_records([y, x, nerve])]

    assert order == ["z.tif", "x.tif", "y.tif"]


def test_top3_returns_first_three_checked_with_metrics_and_updates() -> None:
    recs = [_rec(f"{i}.tif", True, density=float(i), area=1, perimeter=1) for i in range(5)]
    # density descending => names 4,3,2,1,0; top-3 is 4,3,2.
    assert [r.name for r in top3(recs)] == ["4.tif", "3.tif", "2.tif"]

    recs[4].is_nerve = False  # uncheck the current top -> top-3 shifts down
    assert [r.name for r in top3(recs)] == ["3.tif", "2.tif", "1.tif"]


def test_top3_excludes_pending_and_unchecked() -> None:
    measured = _rec("m.tif", True, density=1, area=1, perimeter=1)
    no_metrics = ImageRecord(path=Path("p.tif"), is_nerve=True, metrics=None)  # checked but unmeasured
    unchecked = _rec("u.tif", False, density=9, area=9, perimeter=9)

    assert [r.name for r in top3([measured, no_metrics, unchecked])] == ["m.tif"]


def test_selection_after_toggle_check_follows_record() -> None:
    # Four unchecked rows by name; check "d" (now a measured nerve) -> it sorts to the top.
    a = _rec("a.tif", False)
    b = _rec("b.tif", False)
    c = _rec("c.tif", False)
    d = _rec("d.tif", True, density=9, area=1, perimeter=1)
    records = sorted([a, b, c, d], key=sort_key)

    # On check, the target follows the toggled record to its new sorted position (row 0 here).
    assert records.index(d) == 0
    assert selection_after_toggle(records, d, now_checked=True, prior_row=3) == 0


def test_selection_after_toggle_uncheck_keeps_row_position() -> None:
    # Three checked nerves; uncheck the top one -> it drops below, a different record takes row 0.
    top = _rec("a.tif", False, density=9, area=1, perimeter=1)  # already reverted to unchecked
    mid = _rec("b.tif", True, density=5, area=1, perimeter=1)
    low = _rec("c.tif", True, density=1, area=1, perimeter=1)
    records = sorted([top, mid, low], key=sort_key)

    # Prior row 0 is kept by position; the record now at row 0 is no longer `top`.
    assert selection_after_toggle(records, top, now_checked=False, prior_row=0) == 0
    assert records[0] is not top


def test_selection_after_toggle_uncheck_last_row_edge() -> None:
    records = [_rec(f"{i}.tif", True, density=float(i), area=1, perimeter=1) for i in range(3)]
    records = sorted(records, key=sort_key)
    last = len(records) - 1
    assert selection_after_toggle(records, records[last], now_checked=False, prior_row=last) == last


def test_selection_after_toggle_uncheck_no_prior_selection_returns_none() -> None:
    records = [_rec("a.tif", False), _rec("b.tif", False)]
    # An invalid prior row (e.g. remove-from-top3 with no table selection) -> no selection.
    assert selection_after_toggle(records, records[0], now_checked=False, prior_row=-1) is None


def test_overlay_blends_mask_region_only() -> None:
    image = np.zeros((2, 2), dtype=np.float32)  # all black
    mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    color = (255, 0, 0)

    out = overlay(image, mask, color, alpha=0.5)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    # Masked pixel blended halfway toward red; background untouched (still black).
    np.testing.assert_array_equal(out[0, 0], [127, 0, 0])
    np.testing.assert_array_equal(out[0, 1], [0, 0, 0])
    np.testing.assert_array_equal(out[1, 1], [0, 0, 0])


def test_predicted_class_is_argmax() -> None:
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.0]])
    assert predicted_class(logits) == 2


def test_is_nerve_mapping_matches_argmax_vs_nerve_index() -> None:
    # The is-nerve decision (as InferenceEngine.classify makes it) is predicted_class == nerve_index.
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.0]])  # argmax == 2
    assert (predicted_class(logits) == 2) is True  # nerves at index 2 -> is_nerve
    assert (predicted_class(logits) == 1) is False  # nerves at a different index -> not nerve


def test_mask_from_seg_logits_resizes_to_original() -> None:
    # 2x2 logits where foreground (class 1) wins only at the top-left cell.
    logits = torch.zeros((1, 2, 2, 2))
    logits[0, 1, 0, 0] = 10.0

    mask = mask_from_seg_logits(logits, (4, 4))

    assert mask.shape == (4, 4)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    # Nearest-neighbour upsample: the top-left quadrant is foreground, the rest background.
    assert mask[0, 0] == 1
    assert mask[3, 3] == 0


def test_app_icon_is_packaged_and_square() -> None:
    # Guards the real regression: the runtime icon must ship as package data (resolvable via
    # importlib.resources whether run from a checkout or an installed wheel) and be a valid square
    # image. Qt-free, so this stays runnable in the base environment.
    data = resources.files("ivcm.app").joinpath("assets", "icons", "icon.png").read_bytes()
    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG"
    assert image.width == image.height
