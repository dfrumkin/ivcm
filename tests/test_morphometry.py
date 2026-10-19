"""Behaviour tests for DC morphometry on synthetic masks with known geometry.

Tolerances are loose enough to absorb sub-pixel boundary smoothing but tight enough to pin the
intended measurement (area ≈ analytic, perimeter ≈ analytic and far below the pixel staircase,
holes filled, components not bridged).
"""

import math

import numpy as np
import pytest

from ivcm.morphometry import cell_area, cell_density, cell_morphometry, cell_perimeter, postprocess_mask


def _close(actual: float, expected: float, rel: float = 0.02) -> bool:
    """True when ``actual`` is within ``rel`` (relative) of ``expected`` -- a typed stand-in for
    ``pytest.approx`` (which is untyped and would trip strict Pyright)."""
    return abs(actual - expected) <= rel * abs(expected)


def _disk(size: int, radius: float, cy: float, cx: float) -> np.ndarray:
    """A filled binary disk of ``radius`` centred at ``(cy, cx)`` in a ``size×size`` image."""
    ys, xs = np.ogrid[:size, :size]
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[(ys - cy) ** 2 + (xs - cx) ** 2 <= radius * radius] = 1
    return mask


def _staircase_perimeter(mask: np.ndarray) -> int:
    """Rook-boundary length: number of 4-adjacent foreground↔background pixel transitions.

    This is the naive pixel-staircase perimeter, which strictly overestimates a smooth boundary.
    """
    binary = mask != 0
    horizontal = int(np.sum(binary[:, :-1] != binary[:, 1:]))
    vertical = int(np.sum(binary[:-1, :] != binary[1:, :]))
    return horizontal + vertical


def test_disk_area_and_perimeter_match_analytic() -> None:
    """A digitised disk measures close to πr² / 2πr, and the smoothed perimeter is far closer to
    2πr than the pixel-staircase baseline (the anti-staircase purpose).

    Passing ``physical_image_size`` equal to the mask side makes the pixel scale exactly 1, so the
    raw pixel geometry is measured in pixel units (isolated from the physical-unit conversion).
    """
    radius = 40.0
    mask = _disk(120, radius, 60, 60)

    area = cell_area(mask, sigma=1.0, physical_image_size=120)
    perimeter = cell_perimeter(mask, sigma=1.0, physical_image_size=120)
    ideal_area = math.pi * radius**2
    ideal_perimeter = 2 * math.pi * radius

    assert _close(area, ideal_area, 0.02)
    assert _close(perimeter, ideal_perimeter, 0.03)

    staircase = _staircase_perimeter(mask)
    assert abs(perimeter - ideal_perimeter) < abs(staircase - ideal_perimeter)


def test_square_area_and_perimeter() -> None:
    """An axis-aligned square measures close to s² / 4s; corner rounding shrinks both slightly.

    ``physical_image_size == mask side`` isolates the raw pixel geometry (scale == 1).
    """
    side = 40
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10 : 10 + side, 10 : 10 + side] = 1

    assert _close(cell_area(mask, sigma=1.0, physical_image_size=60), side * side, 0.02)
    assert _close(cell_perimeter(mask, sigma=1.0, physical_image_size=60), 4 * side, 0.05)


def test_density_is_count_over_physical_area_and_min_area_filters() -> None:
    """Density is ``count / (physical_image_size / 1000)**2`` (cells/mm², µm side length); the
    default ``min_area=9.0`` µm² filters a 1-µm² speck that ``min_area=0`` keeps."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[_disk(100, 8, 25, 25) > 0] = 1
    mask[_disk(100, 8, 25, 75) > 0] = 1
    mask[_disk(100, 8, 75, 50) > 0] = 1
    mask[90, 90] = 1  # a 1-pixel speck: 1 µm² at scale 1 (physical_image_size == mask side)

    # Without filtering the speck counts (4 components); the default min_area=9.0 drops it (3).
    assert cell_density(mask, min_area=0, physical_image_size=100.0) == 4 / (100.0 / 1000) ** 2
    assert cell_density(mask, physical_image_size=100.0) == 3 / (100.0 / 1000) ** 2  # default min_area=9.0


def test_physical_scaling_matches_scale_one_baseline() -> None:
    """Count is unaffected by the physical scale, and the other three measures scale exactly by the
    physical/pixel ratio: density by ``1/S**2`` and area/perimeter (per-cell averages) from raw
    pixels by ``scale**2`` / ``scale`` where ``scale = S / R``."""
    mask = _disk(120, 40.0, 60, 60)
    resolution = mask.shape[0]  # R

    # scale == 1: area/perimeter are the raw pixel measurements; density is count / (R/1000)**2.
    # ``min_area=0``: the tiny 2.5 µm field would otherwise put the whole disk below the threshold.
    baseline = cell_morphometry(mask, sigma=1.0, physical_image_size=resolution, min_area=0)
    size = 2.5
    scaled = cell_morphometry(mask, sigma=1.0, physical_image_size=size, min_area=0)
    scale = size / resolution

    assert scaled["count"] == baseline["count"] == 1.0
    # density * S**2 == count * 1000**2 on both sides, so the two densities encode the same count.
    assert _close(scaled["density"] * size**2, baseline["density"] * resolution**2, 1e-9)
    assert _close(scaled["area"], baseline["area"] * scale**2, 1e-9)
    assert _close(scaled["perimeter"], baseline["perimeter"] * scale, 1e-9)


def test_min_area_applied_consistently_across_metrics() -> None:
    """A speck below min_area is excluded from density *and* contributes no area/perimeter, so all
    four measures stay on the same component set."""
    big = _disk(120, 30, 60, 60)
    mask = big.copy()
    mask[5, 5] = 1  # tiny speck: one pixel, (400 / 120)**2 ≈ 11.1 µm²

    big_only = cell_morphometry(big, sigma=1.0)
    with_speck = cell_morphometry(mask, sigma=1.0, min_area=20.0)

    # The surviving big component is processed identically with or without the (removed) speck, so
    # the measures are exactly equal, not merely close. Density is count / (physical_image_size/1000)**2
    # with the default physical_image_size=400 (µm) -> 1 / 0.4**2 cells/mm².
    assert with_speck["count"] == big_only["count"] == 1.0
    assert with_speck["density"] == big_only["density"] == 1 / (400.0 / 1000) ** 2
    assert with_speck["area"] == big_only["area"]
    assert with_speck["perimeter"] == big_only["perimeter"]


def test_close_components_not_bridged() -> None:
    """Two separated, identically-sized disks count as two, and their average area/perimeter per
    cell match a single disk's — the per-component smoothing does not merge them, and dividing by
    the doubled count cancels the doubled total."""
    radius = 15.0
    single = cell_morphometry(_disk(160, radius, 80, 80), sigma=1.0)

    # Two disks 40px apart (centres at cols 60 and 100): a wide background gap separates them.
    pair = _disk(160, radius, 80, 60)
    pair[_disk(160, radius, 80, 100) > 0] = 1
    # Sanity: genuinely two 8-connected components, not a diagonal touch (density = 2 / 0.4**2).
    assert cell_density(pair) == 2 / (400.0 / 1000) ** 2

    both = cell_morphometry(pair, sigma=1.0)
    assert both["count"] == 2.0
    assert _close(both["area"], single["area"], 0.02)
    assert _close(both["perimeter"], single["perimeter"], 0.02)


def test_hole_is_filled() -> None:
    """An annulus is measured as a solid disk: area ≈ the full outer area and perimeter ≈ the
    single outer circumference (no inner contour)."""
    radius = 40.0
    annulus = _disk(120, radius, 60, 60).copy()
    annulus[_disk(120, 15, 60, 60) > 0] = 0  # punch a hole

    # Filling the hole reconstructs the solid disk exactly, so the measures are identical.
    solid = _disk(120, radius, 60, 60)
    assert cell_density(annulus) == 1 / (400.0 / 1000) ** 2
    assert cell_area(annulus, sigma=1.0) == cell_area(solid, sigma=1.0)
    assert cell_perimeter(annulus, sigma=1.0) == cell_perimeter(solid, sigma=1.0)


def test_empty_mask_is_zero() -> None:
    """A mask with no foreground yields zero for all four measures (no divide-by-zero on count)."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    result = cell_morphometry(empty)
    assert result == {"count": 0.0, "density": 0.0, "area": 0.0, "perimeter": 0.0}


def test_edge_touching_object_is_measured() -> None:
    """A cell overlapping the image border is still measured (padding closes its contour)."""
    mask = _disk(80, 20, 0, 40)  # centre on the top edge: the disk is clipped by the border
    assert cell_density(mask) == 1 / (400.0 / 1000) ** 2
    area = cell_area(mask, sigma=1.0)
    perimeter = cell_perimeter(mask, sigma=1.0)
    assert area > 0 and math.isfinite(area)
    assert perimeter > 0 and math.isfinite(perimeter)


def test_invalid_inputs_raise() -> None:
    """Non-2D / non-square masks and out-of-range parameters fail loudly rather than degrading."""
    with pytest.raises(ValueError):
        cell_area(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        cell_density(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):  # non-square
        cell_area(np.zeros((4, 8), dtype=np.uint8))
    with pytest.raises(ValueError):  # non-square
        cell_density(np.zeros((4, 8), dtype=np.uint8))
    with pytest.raises(ValueError):  # non-positive physical_image_size
        cell_area(np.zeros((4, 4), dtype=np.uint8), physical_image_size=0.0)
    with pytest.raises(ValueError):
        cell_density(np.zeros((4, 4), dtype=np.uint8), physical_image_size=-1.0)
    with pytest.raises(ValueError):
        cell_area(np.zeros((4, 4), dtype=np.uint8), sigma=-1.0)
    with pytest.raises(ValueError):
        cell_density(np.zeros((4, 4), dtype=np.uint8), min_area=-1.0)
    with pytest.raises(ValueError):
        postprocess_mask(np.zeros((4, 4), dtype=np.uint8), min_area=-1.0)


def test_postprocess_mask_removes_cells_below_min_area() -> None:
    """Components strictly below ``min_area`` µm² are removed from the returned mask; one exactly at
    the threshold and a large cell are kept unchanged; the input is not modified; and the metrics
    are those of the cleaned mask (all on the same component set)."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[_disk(60, 12, 30, 40) > 0] = 1  # a large cell
    mask[2:5, 2:5] = 1  # 3x3 = 9 px = 9 µm² at scale 1: exactly the threshold, kept
    mask[50:52, 2:6] = 1  # 2x4 = 8 px = 8 µm²: below the threshold, removed
    original = mask.copy()

    clean, metrics = postprocess_mask(mask, physical_image_size=60.0, min_area=9.0)

    expected = original.copy()
    expected[50:52, 2:6] = 0
    np.testing.assert_array_equal(clean, expected)
    assert clean.dtype == mask.dtype
    np.testing.assert_array_equal(mask, original)  # input untouched
    assert metrics["count"] == 2.0
    assert metrics == cell_morphometry(clean, physical_image_size=60.0, min_area=9.0)


def test_postprocess_mask_threshold_is_physical_not_pixel() -> None:
    """The same 9-pixel component is kept at scale 1 (9 µm²) but removed at scale 0.9 (≈7.3 µm²)."""
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:13, 10:13] = 1

    kept, kept_metrics = postprocess_mask(mask, physical_image_size=60.0, min_area=9.0)
    removed, removed_metrics = postprocess_mask(mask, physical_image_size=54.0, min_area=9.0)

    np.testing.assert_array_equal(kept, mask)
    assert kept_metrics["count"] == 1.0
    assert not removed.any()
    assert removed_metrics == {"count": 0.0, "density": 0.0, "area": 0.0, "perimeter": 0.0}
