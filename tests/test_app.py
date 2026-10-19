"""Pure-logic tests for the clinical-review UI (no QApplication, no checkpoints, no Qt)."""

from __future__ import annotations

import io
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from ivcm.app import config as app_config
from ivcm.app.editing import (
    EditHistory,
    brush_cursor_spec,
    map_to_mask,
    mask_changed,
    raise_slider_to_floor,
    stamp_disk,
    stamp_stroke,
)
from ivcm.app.inference import (
    disable_net_downloads,
    mask_from_seg_logits,
    predicted_class,
    strip_criterion_state,
)
from ivcm.app.records import (
    ImageRecord,
    initial_pins,
    pinned_count,
    pinned_display,
    replacement_for,
    selection_after_toggle,
    sort_key,
    sorted_records,
)
from ivcm.app.render import overlay
from ivcm.app.summary import selection_summary


def _rec(
    name: str,
    is_nerve: bool,
    density: float | None = None,
    area: float | None = None,
    perimeter: float | None = None,
    pinned: bool = False,
    count: float | None = None,
) -> ImageRecord:
    metrics = None
    if density is not None:
        metrics = {
            "count": density if count is None else count,
            "density": density,
            "area": area or 0.0,
            "perimeter": perimeter or 0.0,
        }
    return ImageRecord(path=Path(name), is_nerve=is_nerve, pinned=pinned, metrics=metrics)


def test_sort_key_orders_nerve_first_then_metrics_then_name() -> None:
    a = _rec("a.tif", True, density=1, area=1, perimeter=1, count=5)
    b = _rec("b.tif", True, density=1, area=1, perimeter=1, count=10)
    e = _rec("e.tif", True, density=1, area=2, perimeter=1, count=10)  # higher area breaks the count tie
    c = _rec("c.tif", False)
    d = _rec("d.tif", False)

    order = [r.name for r in sorted_records([a, b, c, d, e])]

    # nerve rows first (by count desc, area desc as tiebreak), then unchecked rows by name.
    assert order == ["e.tif", "b.tif", "a.tif", "c.tif", "d.tif"]


def test_sort_key_unchecked_rows_order_by_name_below_nerve() -> None:
    nerve = _rec("z.tif", True, density=1, area=1, perimeter=1)
    x = _rec("x.tif", False)
    y = _rec("y.tif", False)

    order = [r.name for r in sorted_records([y, x, nerve])]

    assert order == ["z.tif", "x.tif", "y.tif"]


def test_sort_key_ignores_pins() -> None:
    # A pinned row with a low count sorts by (use, count, ...) like any other used row.
    strong = _rec("strong.tif", True, density=1, area=1, perimeter=1, count=100)
    pin_low = _rec("plow.tif", True, density=1, area=1, perimeter=1, pinned=True, count=1)
    pin_high = _rec("phigh.tif", True, density=1, area=1, perimeter=1, pinned=True, count=5)
    unchecked = _rec("u.tif", False)

    order = [r.name for r in sorted_records([unchecked, pin_low, strong, pin_high])]

    assert order == ["strong.tif", "phigh.tif", "plow.tif", "u.tif"]


def test_pinned_count_counts_pinned_records() -> None:
    records = [
        _rec("a.tif", True, density=1, area=1, perimeter=1, pinned=True),
        _rec("b.tif", True, density=1, area=1, perimeter=1, pinned=True),
        _rec("c.tif", True, density=1, area=1, perimeter=1),
        _rec("d.tif", False),
    ]
    assert pinned_count(records) == 2
    assert pinned_count([]) == 0


def _counted(counts: list[float]) -> list[ImageRecord]:
    """Used, measured records named ``c<count>.tif`` with the given counts."""
    return [_rec(f"c{c:g}.tif", True, density=1, area=1, perimeter=1, count=c) for c in counts]


def test_pinned_display_shows_only_pinned_by_count_ascending() -> None:
    recs = _counted([5, 1, 9, 3])
    for r in recs[:3]:
        r.pinned = True  # counts 5, 1, 9; 3 stays unpinned

    assert [r.name for r in pinned_display(recs)] == ["c1.tif", "c5.tif", "c9.tif"]


def test_initial_pins_pick_closest_to_mean_minus_sd_mean_plus_sd() -> None:
    # counts 0, 2, 4, 6, 8: mean 4, sample SD ~3.16 -> targets ~0.84, 4, ~7.16.
    recs = _counted([8, 0, 4, 6, 2])
    recs.append(_rec("unused.tif", False, density=1, area=1, perimeter=1, count=1))  # ignored
    recs.append(ImageRecord(path=Path("unmeasured.tif"), is_nerve=True))  # ignored

    assert [r.name for r in initial_pins(recs)] == ["c0.tif", "c4.tif", "c8.tif"]


def test_initial_pins_are_distinct_and_cover_small_sets() -> None:
    assert [r.name for r in initial_pins(_counted([7]))] == ["c7.tif"]
    assert sorted(r.name for r in initial_pins(_counted([1, 2]))) == ["c1.tif", "c2.tif"]
    assert initial_pins([_rec("u.tif", False)]) == []


def test_replacement_uses_the_slot_target() -> None:
    # counts 0..8 step 2 (mean 4, SD ~3.16); nothing pinned or excluded.
    recs = _counted([0, 2, 4, 6, 8])

    picks = [replacement_for(recs, slot, set()) for slot in range(3)]

    assert [r.name if r else None for r in picks] == ["c0.tif", "c4.tif", "c8.tif"]


def test_replacement_skips_pinned_excluded_unused_and_unmeasured() -> None:
    recs = _counted([0, 2, 4, 6, 8])
    by_name = {r.name: r for r in recs}
    by_name["c4.tif"].pinned = True  # currently pinned
    excluded = {by_name["c6.tif"]}  # pinned earlier this session
    recs.append(_rec("unused.tif", False, density=1, area=1, perimeter=1, count=4))
    recs.append(ImageRecord(path=Path("unmeasured.tif"), is_nerve=True))

    pick = replacement_for(recs, 1, excluded)  # center: mean 4

    assert pick is not None
    assert pick.name == "c2.tif"  # 4 is pinned and 6 excluded; 2 is the next closest


def test_replacement_none_when_nothing_qualifies() -> None:
    recs = _counted([1, 2])
    for r in recs:
        r.pinned = True
    assert replacement_for(recs, 0, set()) is None
    assert replacement_for([_rec("u.tif", False)], 0, set()) is None


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
    # An invalid prior row (e.g. remove-from-sample with no table selection) -> no selection.
    assert selection_after_toggle(records, records[0], now_checked=False, prior_row=-1) is None


def test_selection_summary_covers_only_checked_measured_records() -> None:
    checked = [
        _rec("a.tif", True, density=2, area=10, perimeter=100),
        _rec("b.tif", True, density=4, area=20, perimeter=300),
    ]
    unchecked_but_measured = _rec("c.tif", False, density=99, area=99, perimeter=99)
    checked_but_unmeasured = ImageRecord(path=Path("d.tif"), is_nerve=True, metrics=None)

    summary = selection_summary([*checked, unchecked_but_measured, checked_but_unmeasured], bins=4)

    assert summary.count == 2
    # Only the two checked-and-measured records contribute: the 99s never reach the statistics.
    assert summary.metrics["density"].mean == 3.0
    assert summary.metrics["area"].maximum == 20.0
    assert summary.metrics["perimeter"].maximum == 300.0


def test_selection_summary_statistics_across_all_four_metrics() -> None:
    records = [
        _rec("a.tif", True, density=2, area=10, perimeter=100, count=20),
        _rec("b.tif", True, density=4, area=20, perimeter=200, count=40),
        _rec("c.tif", True, density=6, area=60, perimeter=300, count=60),
    ]

    summary = selection_summary(records, bins=4)

    assert summary.count == 3
    count = summary.metrics["count"]
    assert count.mean == 40.0
    assert count.sd == 20.0  # sample SD (ddof=1) of 20, 40, 60
    assert (count.minimum, count.maximum) == (20.0, 60.0)

    density = summary.metrics["density"]
    assert density.mean == 4.0
    assert density.sd == 2.0  # sample SD (ddof=1) of 2, 4, 6
    assert (density.minimum, density.maximum) == (2.0, 6.0)

    area = summary.metrics["area"]
    assert area.mean == 30.0
    assert (area.minimum, area.maximum) == (10.0, 60.0)

    perimeter = summary.metrics["perimeter"]
    assert perimeter.mean == 200.0
    assert perimeter.sd == 100.0  # sample SD of 100, 200, 300
    assert (perimeter.minimum, perimeter.maximum) == (100.0, 300.0)


def test_selection_summary_histogram_accounts_for_every_selected_image() -> None:
    # Values at the axis ends and between: the image at the maximum must land in the final bin
    # rather than fall outside the histogram, so the counts total the selected count.
    records = [_rec(f"{i}.tif", True, density=float(i), area=1, perimeter=1) for i in (0, 1, 3, 4)]

    summary = selection_summary(records, bins=4)
    histogram = summary.metrics["density"].histogram

    assert len(histogram) == 4
    assert sum(histogram) == summary.count == 4
    # Bins over [0, 4]: [0,1) -> 0; [1,2) -> 1; [2,3) -> nothing; [3,4] -> 3 and the max, 4.
    assert histogram == (1, 1, 0, 2)


def test_selection_summary_single_record_has_no_measurable_sd() -> None:
    summary = selection_summary([_rec("a.tif", True, density=5, area=7, perimeter=9)], bins=4)

    assert summary.count == 1
    density = summary.metrics["density"]
    assert (density.mean, density.minimum, density.maximum) == (5.0, 5.0, 5.0)
    assert density.sd is None  # a sample SD needs two images
    assert sum(density.histogram) == 1


def test_selection_summary_empty_selection_has_no_statistics() -> None:
    summary = selection_summary([_rec("a.tif", False), _rec("b.tif", False)], bins=4)

    assert summary.count == 0
    for metric in summary.metrics.values():
        assert (metric.mean, metric.sd, metric.minimum, metric.maximum) == (None, None, None, None)
        assert sum(metric.histogram) == 0


def test_selection_summary_all_zero_metric_collapses_to_first_bin() -> None:
    # A zero maximum leaves the axis no extent; the values must still all be accounted for.
    records = [_rec(f"{i}.tif", True, density=0, area=0, perimeter=0) for i in range(3)]

    density = selection_summary(records, bins=4).metrics["density"]

    assert (density.mean, density.minimum, density.maximum) == (0.0, 0.0, 0.0)
    assert density.histogram == (3, 0, 0, 0)


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


def test_mask_from_seg_logits_argmax_at_native_resolution() -> None:
    # 2x2 logits where foreground (class 1) wins only at the top-left cell. Sliding-window
    # inference already produces native-resolution logits, so the mask is a plain argmax at the
    # logits' own H×W — no resize to a separate original size.
    logits = torch.zeros((1, 2, 2, 2))
    logits[0, 1, 0, 0] = 10.0

    mask = mask_from_seg_logits(logits)

    assert mask.shape == (2, 2)  # same H×W as the logits, not upsampled
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask[0, 0] == 1  # foreground where class 1 wins
    assert mask[1, 1] == 0  # background elsewhere


def test_strip_criterion_state_drops_only_criterion_keys() -> None:
    # A checkpoint from weighted-CE training carries `criterion.weight` (and could carry other
    # criterion buffers). Inference rebuilds the loss without those, so they must be dropped while
    # every `net.*` weight is preserved unchanged.
    net_weight = torch.ones(3)
    state = {
        "net.encoder.conv.weight": net_weight,
        "criterion.weight": torch.tensor([0.5, 2.0]),
    }

    stripped = strip_criterion_state(state)

    assert set(stripped) == {"net.encoder.conv.weight"}
    assert stripped["net.encoder.conv.weight"] is net_weight


def test_strip_criterion_state_is_noop_without_criterion_keys() -> None:
    state = {"net.a": torch.zeros(1), "net.b": torch.ones(1)}
    assert strip_criterion_state(state) == state


def test_map_to_mask_center_margins_and_corners() -> None:
    # No letter-boxing: the displayed image fills the label.
    assert map_to_mask((50.0, 50.0), (100, 100), (100, 100), (100, 100)) == (50, 50)
    # The far bottom-right corner clamps to the last pixel, not one past it.
    assert map_to_mask((99.9, 99.9), (100, 100), (100, 100), (100, 100)) == (99, 99)
    # Horizontal letter-boxing (label wider than the image): the side margins map to nothing.
    assert map_to_mask((5.0, 50.0), (100, 100), (120, 100), (100, 100)) is None
    assert map_to_mask((115.0, 50.0), (100, 100), (120, 100), (100, 100)) is None


def test_map_to_mask_non_square_letterboxes_on_the_other_axis() -> None:
    # A wide image (100x50 displayed) in a square label letter-boxes top/bottom (offset_y = 25).
    assert map_to_mask((50.0, 25.0), (100, 50), (100, 100), (50, 100)) == (0, 50)
    assert map_to_mask((50.0, 10.0), (100, 50), (100, 100), (50, 100)) is None


def test_brush_cursor_spec_style_threshold_and_growth() -> None:
    # The smallest brushes (radius 1 and 2 => ring diameter 2 and 4, both < 6) fall back to a
    # crosshair; from radius 3 (diameter 6) up the ring is used.
    assert brush_cursor_spec(1).style == "crosshair"
    assert brush_cursor_spec(2).style == "crosshair"
    assert brush_cursor_spec(3).style == "ring"
    assert brush_cursor_spec(8).style == "ring"

    # Ring pixmaps grow with the radius; the hotspot is always the pixmap center.
    small = brush_cursor_spec(3)
    large = brush_cursor_spec(8)
    assert large.size > small.size
    for spec in (brush_cursor_spec(1), small, large):
        assert spec.size == 2 * spec.hotspot + 1  # odd side, hotspot at the true center


def test_stamp_disk_paints_erases_and_clips() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    stamp_disk(mask, 5, 5, 2, 1)
    assert mask[5, 5] == 1
    assert mask[5, 7] == 1  # distance 2 <= radius 2
    assert mask[5, 8] == 0  # distance 3 > radius 2

    stamp_disk(mask, 5, 5, 2, 0)
    assert mask.sum() == 0  # fully erased

    stamp_disk(mask, 0, 0, 3, 1)  # centered at the corner: clipped to bounds, no error
    assert mask[0, 0] == 1


def test_stamp_stroke_leaves_no_gap() -> None:
    mask = np.zeros((5, 30), dtype=np.uint8)
    stamp_stroke(mask, (2, 2), (2, 27), 1, 1)
    # The whole center line between the endpoints is painted: a fast drag skips no pixel.
    assert (mask[2, 2:28] == 1).all()


def test_raise_slider_to_floor() -> None:
    assert raise_slider_to_floor(20, 50) == 50  # below the floor is raised
    assert raise_slider_to_floor(70, 50) == 70  # above the floor is left alone
    assert raise_slider_to_floor(50, 50) == 50  # at the floor is unchanged


def test_mask_changed_detects_any_difference() -> None:
    a = np.zeros((3, 3), dtype=np.uint8)
    b = np.zeros((3, 3), dtype=np.uint8)
    assert mask_changed(a, b) is False
    b[1, 1] = 1
    assert mask_changed(a, b) is True


def test_edit_history_undo_redo_and_branch_truncation() -> None:
    history = EditHistory(np.zeros((2, 2), dtype=np.uint8))
    assert not history.can_undo() and not history.can_redo()

    s1 = np.zeros((2, 2), dtype=np.uint8)
    s1[0, 0] = 1
    s2 = np.zeros((2, 2), dtype=np.uint8)
    s2[1, 1] = 1
    history.push(s1)
    history.push(s2)
    assert history.can_undo() and not history.can_redo()

    np.testing.assert_array_equal(history.undo(), s1)
    assert history.can_redo()
    np.testing.assert_array_equal(history.redo(), s2)

    # A push after an undo drops the redo branch — the undone future is not recoverable.
    history.undo()  # back to s1
    s3 = np.zeros((2, 2), dtype=np.uint8)
    s3[0, 1] = 1
    history.push(s3)
    assert not history.can_redo()
    np.testing.assert_array_equal(history.current(), s3)


def test_edit_history_returns_independent_copies() -> None:
    history = EditHistory(np.zeros((2, 2), dtype=np.uint8))
    snapshot = history.current()
    snapshot[0, 0] = 1  # mutating a returned mask must not corrupt the stored state
    np.testing.assert_array_equal(history.current(), np.zeros((2, 2), dtype=np.uint8))


def test_app_icon_is_packaged_and_square() -> None:
    # Guards the real regression: the runtime icon must ship as package data (resolvable via
    # importlib.resources whether run from a checkout or an installed wheel) and be a valid square
    # image. Qt-free, so this stays runnable in the base environment.
    data = resources.files("ivcm.app").joinpath("assets", "icons", "icon.png").read_bytes()
    image = Image.open(io.BytesIO(data))
    assert image.format == "PNG"
    assert image.width == image.height


def test_build_config_frozen_redirects_only_model_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # In a PyInstaller bundle (sys.frozen / sys._MEIPASS), the two model (run_dir, ckpt) pairs must
    # resolve under the bundled models/ directory, while every other operational field is carried
    # over unchanged (dataclasses.replace guards against silently dropping defaults).
    monkeypatch.setattr(app_config.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_config.sys, "_MEIPASS", str(tmp_path), raising=False)

    frozen = app_config.build_config()

    cls = frozen.classification
    assert cls.run_path() == tmp_path / "models" / "cls"
    assert cls.ckpt_path() == tmp_path / "models" / "cls" / "best-weights.ckpt"
    assert cls.config_path() == tmp_path / "models" / "cls" / ".hydra" / "config.yaml"
    seg = frozen.segmentation
    assert seg.run_path() == tmp_path / "models" / "seg"
    assert seg.ckpt_path() == tmp_path / "models" / "seg" / "best-weights.ckpt"

    # Non-path operational defaults preserved verbatim.
    base = app_config.CONFIG
    assert frozen.physical_image_size == base.physical_image_size
    assert frozen.sigma == base.sigma
    assert frozen.min_area == base.min_area
    assert frozen.batch_size == base.batch_size
    assert frozen.num_workers == base.num_workers
    assert frozen.accepted_extensions == base.accepted_extensions
    assert frozen.mask_color == base.mask_color
    assert frozen.mask_alpha == base.mask_alpha


def test_build_config_non_frozen_is_the_local_config() -> None:
    # With no frozen state, the factory returns the machine-specific config unchanged (the module
    # CONFIG is that same object), so local `uv run ivcm-ui` behaviour is untouched.
    assert app_config.build_config() is app_config.CONFIG


def test_disable_net_downloads_turns_off_pretrained_and_encoder_weights() -> None:
    # Inference loads trained weights over the net, so any pretrained/ImageNet download at
    # construction is waste — and it breaks a standalone/offline/windowed bundle. Both the timm
    # (`pretrained`) and SMP (`encoder_weights`) knobs must be disabled before instantiation.
    model_cfg = OmegaConf.create({"net": {"model_name": "resnet18", "pretrained": True, "encoder_weights": "imagenet"}})

    disable_net_downloads(model_cfg)

    assert model_cfg.net.pretrained is False
    assert model_cfg.net.encoder_weights is None


def test_disable_net_downloads_leaves_absent_keys_untouched() -> None:
    # A net without those keys is left as-is (no keys spuriously added, no crash).
    model_cfg = OmegaConf.create({"net": {"model_name": "resnet18"}})

    disable_net_downloads(model_cfg)

    assert "pretrained" not in model_cfg.net
    assert "encoder_weights" not in model_cfg.net
