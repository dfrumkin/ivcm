"""Behaviour tests for DC morphometry on synthetic masks with known geometry.

Tolerances are loose enough to absorb sub-pixel boundary smoothing but tight enough to pin the
intended measurement (area ≈ analytic, perimeter ≈ analytic and far below the pixel staircase,
holes filled, components not bridged).
"""

import math

import numpy as np
import pytest

from ivcm.morphometry import cell_area, cell_density, cell_morphometry, cell_perimeter


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
    2πr than the pixel-staircase baseline (the anti-staircase purpose)."""
    radius = 40.0
    mask = _disk(120, radius, 60, 60)

    area = cell_area(mask, sigma=1.0)
    perimeter = cell_perimeter(mask, sigma=1.0)
    ideal_area = math.pi * radius**2
    ideal_perimeter = 2 * math.pi * radius

    assert _close(area, ideal_area, 0.02)
    assert _close(perimeter, ideal_perimeter, 0.03)

    staircase = _staircase_perimeter(mask)
    assert abs(perimeter - ideal_perimeter) < abs(staircase - ideal_perimeter)


def test_square_area_and_perimeter() -> None:
    """An axis-aligned square measures close to s² / 4s; corner rounding shrinks both slightly."""
    side = 40
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10 : 10 + side, 10 : 10 + side] = 1

    assert _close(cell_area(mask, sigma=1.0), side * side, 0.02)
    assert _close(cell_perimeter(mask, sigma=1.0), 4 * side, 0.05)


def test_density_counts_components_and_scales() -> None:
    """Density is the component count × constant; a sub-min_size speck is filtered out."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[_disk(100, 8, 25, 25) > 0] = 1
    mask[_disk(100, 8, 25, 75) > 0] = 1
    mask[_disk(100, 8, 75, 50) > 0] = 1
    mask[90, 90] = 1  # a 1-pixel speck

    assert cell_density(mask) == 4.0  # speck included with no filtering
    assert cell_density(mask, min_size=5) == 3.0  # speck filtered out
    assert cell_density(mask, min_size=5, constant=2.5) == 7.5


def test_min_size_applied_consistently_across_metrics() -> None:
    """A speck below min_size is excluded from density *and* contributes no area/perimeter, so the
    three measures stay on the same component set."""
    big = _disk(120, 30, 60, 60)
    mask = big.copy()
    mask[5, 5] = 1  # tiny speck

    big_only = cell_morphometry(big, sigma=1.0)
    with_speck = cell_morphometry(mask, sigma=1.0, min_size=5)

    # The surviving big component is processed identically with or without the (removed) speck,
    # so the measures are exactly equal, not merely close.
    assert with_speck["density"] == big_only["density"] == 1.0
    assert with_speck["area"] == big_only["area"]
    assert with_speck["perimeter"] == big_only["perimeter"]


def test_close_components_not_bridged() -> None:
    """Two separated disks count as two and total area/perimeter ≈ twice a single disk's — the
    per-component smoothing does not merge them."""
    radius = 15.0
    single = cell_morphometry(_disk(160, radius, 80, 80), sigma=1.0)

    # Two disks 40px apart (centres at cols 60 and 100): a wide background gap separates them.
    pair = _disk(160, radius, 80, 60)
    pair[_disk(160, radius, 80, 100) > 0] = 1
    # Sanity: genuinely two 8-connected components, not a diagonal touch.
    assert cell_density(pair) == 2.0

    both = cell_morphometry(pair, sigma=1.0)
    assert _close(both["area"], 2 * single["area"], 0.02)
    assert _close(both["perimeter"], 2 * single["perimeter"], 0.02)


def test_hole_is_filled() -> None:
    """An annulus is measured as a solid disk: area ≈ the full outer area and perimeter ≈ the
    single outer circumference (no inner contour)."""
    radius = 40.0
    annulus = _disk(120, radius, 60, 60).copy()
    annulus[_disk(120, 15, 60, 60) > 0] = 0  # punch a hole

    # Filling the hole reconstructs the solid disk exactly, so the measures are identical.
    solid = _disk(120, radius, 60, 60)
    assert cell_density(annulus) == 1.0
    assert cell_area(annulus, sigma=1.0) == cell_area(solid, sigma=1.0)
    assert cell_perimeter(annulus, sigma=1.0) == cell_perimeter(solid, sigma=1.0)


def test_empty_mask_is_zero() -> None:
    """A mask with no foreground yields zero for all three measures."""
    empty = np.zeros((40, 40), dtype=np.uint8)
    result = cell_morphometry(empty)
    assert result == {"density": 0.0, "area": 0.0, "perimeter": 0.0}


def test_edge_touching_object_is_measured() -> None:
    """A cell overlapping the image border is still measured (padding closes its contour)."""
    mask = _disk(80, 20, 0, 40)  # centre on the top edge: the disk is clipped by the border
    assert cell_density(mask) == 1.0
    area = cell_area(mask, sigma=1.0)
    perimeter = cell_perimeter(mask, sigma=1.0)
    assert area > 0 and math.isfinite(area)
    assert perimeter > 0 and math.isfinite(perimeter)


def test_invalid_inputs_raise() -> None:
    """Non-2D masks and negative parameters fail loudly rather than degrading silently."""
    with pytest.raises(ValueError):
        cell_area(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        cell_density(np.zeros((4, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        cell_area(np.zeros((4, 4), dtype=np.uint8), sigma=-1.0)
    with pytest.raises(ValueError):
        cell_density(np.zeros((4, 4), dtype=np.uint8), min_size=-1)
