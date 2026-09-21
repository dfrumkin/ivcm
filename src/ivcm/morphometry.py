"""Dendritic-cell (DC) morphometry from a binary segmentation mask.

Four measures, reported in the conventional clinical units. ``physical_image_size`` is the
physical side length of the (square) input image **in micrometres** (default ``400`` -- the 400 µm
field of the Heidelberg HRT / Rostock Cornea Module used to acquire these IVCM images). With
``S = physical_image_size`` (µm) and the pixel resolution ``R = mask.shape[0]`` (masks are square,
so ``R == mask.shape[1]``) giving a per-pixel scale ``scale = S / R`` (µm/pixel):

- **count** — DCs per image (surviving 8-connected components, after ``min_size`` filtering);
- **density** — ``count / (S / 1000)**2`` = **DCs per mm²** (the count times a constant, as
  the source protocol defines it; ``S / 1000`` is the side length in mm);
- **area** — average DC surface per image, ``raw_px2 * scale**2 / count`` (**µm²**);
- **perimeter** — average DC outline length per image, ``raw_px * scale / count`` (**µm**).

Density is reported per mm² but area/perimeter per µm² / µm -- the two length units differ because
that is the clinical convention, so density alone carries the µm→mm conversion. Area and perimeter
are per-cell averages (the total divided by ``count``, ``0`` when ``count`` is ``0``) rather than
per-image totals, so they describe a typical cell's size and outline rather than the image's overall
DC burden (density already covers that).

The raw pixel measurements (``raw_px2``/``raw_px``) are unchanged: the 0/1 pixel grid is a biased
substrate for area and (especially) perimeter -- the staircased boundary overestimates perimeter
and quantises area to whole pixels. The common, simple remedy used here -- the Python analogue of
MATLAB ``bwarea``/boundary tracing, not a bit-for-bit reproduction of it -- is to **smooth each
cell and measure on a sub-pixel iso-contour**: per connected component, fill interior holes,
Gaussian-smooth the isolated binary crop, extract the 0.5-level marching-squares contour, and
measure that polygon (Euclidean edge lengths for perimeter, shoelace for area). ``sigma`` governs
the smoothing/bias trade-off: a larger ``sigma`` rounds corners harder and can shrink or erase
small cells and thin dendritic branches; the default is deliberately small. ``scale`` then converts
those raw pixel measurements to physical units.

Functions operate on a single 2D **square** mask; a caller measuring a volume loops over slices.
"""

from __future__ import annotations

from math import ceil
from typing import cast

import numpy as np

# scikit-image and scipy.ndimage ship only partial type info, so their callables come through
# partially-unknown; contain it at the import boundary. The CLI (`uvx pyright`) flags the symbols
# as reportUnknownVariableType; the editor (Pylance, different bundled stubs) additionally flags
# scipy.ndimage as reportMissingTypeStubs on the module line -- both are suppressed below.
from scipy.ndimage import (  # pyright: ignore[reportMissingTypeStubs]
    binary_fill_holes,  # pyright: ignore[reportUnknownVariableType]
    find_objects,  # pyright: ignore[reportUnknownVariableType]
)
from skimage.filters import gaussian  # pyright: ignore[reportUnknownVariableType]
from skimage.measure import find_contours, label  # pyright: ignore[reportUnknownVariableType]

# 8-connectivity (``connectivity=2`` in scikit-image 2D terms), used for labelling so a cell
# and a diagonally-touching speck are one component -- the size filter and the count agree.
_CONNECTIVITY = 2

# ``physical_image_size`` is in µm (so area/perimeter come out in µm² / µm); density is
# conventionally per mm², so its denominator converts the side length from µm to mm.
_UM_PER_MM = 1000.0


def _validate(mask: np.ndarray, *, sigma: float | None, physical_image_size: float, min_size: int) -> None:
    """Reject inputs the measures are not defined for, with a clear message.

    ``sigma`` is only supplied by the smoothing-based measures (area/perimeter); ``None`` skips
    its check for ``cell_density``. The mask must be square: the physical scale (and, for
    area/perimeter, the pixel resolution) is derived from its side length.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W); got shape {mask.shape}. Loop over slices for a volume.")
    if mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square (H == W); got shape {mask.shape}.")
    if physical_image_size <= 0:
        raise ValueError(f"physical_image_size must be > 0; got {physical_image_size}.")
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0; got {min_size}.")
    if sigma is not None and sigma < 0:
        raise ValueError(f"sigma must be >= 0; got {sigma}.")


def _label_components(mask: np.ndarray, min_size: int) -> tuple[np.ndarray, int]:
    """Binarise, 8-connected-label, and drop components smaller than ``min_size`` pixels.

    Returns the (relabelled-by-removal) label image and the surviving component count. Filtering
    on the labelled image keeps density and area/perimeter on exactly the same component set.
    """
    binary = mask != 0
    labels = cast(np.ndarray, label(binary, connectivity=_CONNECTIVITY))
    n = int(labels.max())
    if min_size > 0 and n > 0:
        sizes = np.bincount(labels.ravel())
        too_small = np.flatnonzero(sizes < min_size)
        too_small = too_small[too_small != 0]  # never drop background (label 0)
        if too_small.size:
            labels = labels.copy()
            labels[np.isin(labels, too_small)] = 0
    ids = np.unique(labels)
    ids = ids[ids != 0]
    return labels, int(ids.size)


def _component_contours(component: np.ndarray, sigma: float) -> list[np.ndarray]:
    """Sub-pixel 0.5-level contours of one isolated solid cell.

    Fill the component's interior holes (so it is a solid region), pad with a background border
    wide enough that the Gaussian tail and the contour of an edge-touching cell stay inside the
    array (open contours would break the closed-polygon measures), Gaussian-smooth, and trace.
    """
    filled = cast(np.ndarray, binary_fill_holes(component))
    pad = max(1, ceil(3 * sigma))
    padded = np.pad(filled, pad, mode="constant", constant_values=False)
    smoothed = cast(np.ndarray, gaussian(padded.astype(float), sigma=sigma))
    return cast("list[np.ndarray]", find_contours(smoothed, 0.5))


def _polygon_perimeter(contour: np.ndarray) -> float:
    """Closed-polygon length: sum of Euclidean distances between consecutive vertices.

    ``find_contours`` returns closed contours (first vertex repeated as last), so the vertex
    deltas already include the closing edge.
    """
    deltas = np.diff(contour, axis=0)
    return float(np.hypot(deltas[:, 0], deltas[:, 1]).sum())


def _polygon_area(contour: np.ndarray) -> float:
    """Absolute shoelace area of a closed contour.

    Taking the absolute value makes the result independent of contour orientation; the duplicated
    closing vertex contributes a zero-length term. Computed on the ``(row, col)`` coordinates
    ``find_contours`` returns -- the row/col vs x/y swap does not change the absolute area.
    """
    rows, cols = contour[:, 0], contour[:, 1]
    return 0.5 * float(abs(np.dot(rows, np.roll(cols, -1)) - np.dot(cols, np.roll(rows, -1))))


def _area_perimeter(labels: np.ndarray, sigma: float) -> tuple[float, float]:
    """Total (area, perimeter) over all labelled components, each smoothed in isolation.

    Smoothing per component (on a bounding-box crop containing only that label) prevents a
    neighbouring cell from bridging across a narrow gap, and keeps the area/perimeter basis equal
    to the density count.
    """
    area = 0.0
    perimeter = 0.0
    boxes = cast("list[tuple[slice, ...] | None]", find_objects(labels))
    for i, box in enumerate(boxes, start=1):
        if box is None:  # a label that was filtered out leaves a gap in find_objects
            continue
        component = labels[box] == i
        for contour in _component_contours(component, sigma):
            area += _polygon_area(contour)
            perimeter += _polygon_perimeter(contour)
    return area, perimeter


def cell_density(mask: np.ndarray, *, physical_image_size: float = 400.0, min_size: int = 3) -> float:
    """DC count (8-connected components, after ``min_size`` filtering) per mm².

    ``count / (physical_image_size / 1000)**2`` -- components per mm², using the square image's
    physical side length (``physical_image_size`` in µm; ``/1000`` converts it to mm). Independent
    of the pixel resolution.
    """
    _validate(mask, sigma=None, physical_image_size=physical_image_size, min_size=min_size)
    _, count = _label_components(mask, min_size)
    return count / (physical_image_size / _UM_PER_MM) ** 2


def cell_area(mask: np.ndarray, *, sigma: float = 1.0, physical_image_size: float = 400.0, min_size: int = 3) -> float:
    """Total DC surface area in µm² from the smoothed per-component contours.

    The raw pixel area is scaled by ``(physical_image_size / R)**2`` (``R`` the square mask's side
    length; ``physical_image_size`` in µm). ``sigma`` controls the smoothing/bias trade-off (see the
    module docstring); ``sigma=0`` measures the raw 0.5 contour with no smoothing.
    """
    _validate(mask, sigma=sigma, physical_image_size=physical_image_size, min_size=min_size)
    labels, _ = _label_components(mask, min_size)
    area, _ = _area_perimeter(labels, sigma)
    scale = physical_image_size / mask.shape[0]
    return area * scale**2


def cell_perimeter(
    mask: np.ndarray, *, sigma: float = 1.0, physical_image_size: float = 400.0, min_size: int = 3
) -> float:
    """Total DC outline length in µm from the smoothed per-component contours.

    The raw pixel perimeter is scaled by ``physical_image_size / R`` (``R`` the square mask's side
    length; ``physical_image_size`` in µm). ``sigma`` controls the smoothing/bias trade-off (see the
    module docstring); ``sigma=0`` measures the raw 0.5 contour with no smoothing.
    """
    _validate(mask, sigma=sigma, physical_image_size=physical_image_size, min_size=min_size)
    labels, _ = _label_components(mask, min_size)
    _, perimeter = _area_perimeter(labels, sigma)
    scale = physical_image_size / mask.shape[0]
    return perimeter * scale


def cell_morphometry(
    mask: np.ndarray, *, sigma: float = 1.0, physical_image_size: float = 400.0, min_size: int = 3
) -> dict[str, float]:
    """All four measures at once: ``{"count", "density", "area", "perimeter"}`` (cells, cells/mm²,
    µm², µm; area/perimeter are per-cell averages, not per-image totals).

    The intended single entry point at inference: cleans/labels the mask once and extracts the
    smoothed contours once for area and perimeter, avoiding redundant work. Density is
    ``count / (physical_image_size / 1000)**2``; area/perimeter scale the raw pixel measurements by
    ``scale**2`` / ``scale`` where ``scale = physical_image_size / R`` (see the module docstring),
    then divide by ``count`` (``0.0`` when ``count`` is ``0``, rather than dividing by zero).
    """
    _validate(mask, sigma=sigma, physical_image_size=physical_image_size, min_size=min_size)
    labels, count = _label_components(mask, min_size)
    area, perimeter = _area_perimeter(labels, sigma)
    scale = physical_image_size / mask.shape[0]
    return {
        "count": float(count),
        "density": count / (physical_image_size / _UM_PER_MM) ** 2,
        "area": area * scale**2 / count if count else 0.0,
        "perimeter": perimeter * scale / count if count else 0.0,
    }
