"""Pure, Qt-free helpers for manual mask editing (importable/testable without PySide6).

The interactive editor (``ivcm.app.preview.EditablePreview``) and the main window use these for
the geometry and mask arithmetic: mapping a widget-space point onto a mask pixel, stamping a brush
disk / stroke into a mask, the blender floor applied when a stroke starts, the change check that
gates the Discard confirmation, and the undo/redo snapshot stack. All of it is numpy-only so it
carries no Qt dependency.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, cast

import numpy as np

# Below this ring diameter (display pixels) the ring is too small to read, so the brush cursor falls
# back to a crosshair with a center dot instead.
_CURSOR_MIN_RING_DIAMETER = 6
# Padding (display pixels) added around the drawn shape so the outer contrast ring and anti-aliasing
# have room inside the pixmap.
_CURSOR_MARGIN = 3


class CursorSpec(NamedTuple):
    """The size/style of the brush cursor for a given brush radius, in logical (device-independent) px.

    ``style`` is ``"ring"`` for a hollow circle matching the brush, or ``"crosshair"`` when the ring
    would be too small to read (small brushes, including the 1-px minimum). ``size`` is the square
    pixmap's odd side length (so there is a true center pixel); ``hotspot`` is that center — the point
    that tracks the mouse. The caller rasterizes at ``size`` scaled by the device-pixel-ratio and
    passes the logical ``hotspot`` to ``QCursor``, keeping HiDPI scaling out of this pure helper.
    """

    style: Literal["ring", "crosshair"]
    size: int
    hotspot: int


def brush_cursor_spec(radius: float) -> CursorSpec:
    """Map a brush display ``radius`` to the brush cursor's :class:`CursorSpec` (Qt-free)."""
    radius = max(0.0, radius)
    style: Literal["ring", "crosshair"] = "ring" if 2 * radius >= _CURSOR_MIN_RING_DIAMETER else "crosshair"
    if style == "ring":
        half = int(math.ceil(radius)) + _CURSOR_MARGIN
    else:
        half = _CURSOR_MIN_RING_DIAMETER // 2 + _CURSOR_MARGIN
    size = 2 * half + 1  # odd, so the center pixel is exactly the hotspot
    return CursorSpec(style=style, size=size, hotspot=half)


def map_to_mask(
    point: tuple[float, float],
    displayed_size: tuple[int, int],
    label_size: tuple[int, int],
    mask_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Map a widget-space ``point`` to a ``(row, col)`` mask pixel, or ``None`` if outside the image.

    ``displayed_size`` is the actual ``(width, height)`` of the scaled pixmap the label drew and
    ``label_size`` the ``(width, height)`` of the label; the pixmap is centered, so the letter-boxed
    margins are ``(label - displayed) / 2`` on each axis. Points in those margins (outside the drawn
    image) return ``None``. ``mask_shape`` is ``(rows, cols)``; the returned indices are clamped into
    range so a point on the far edge maps to the last row/column rather than one past it.
    """
    disp_w, disp_h = displayed_size
    label_w, label_h = label_size
    rows, cols = mask_shape
    if disp_w <= 0 or disp_h <= 0:
        return None

    offset_x = (label_w - disp_w) / 2
    offset_y = (label_h - disp_h) / 2
    local_x = point[0] - offset_x
    local_y = point[1] - offset_y
    if not (0.0 <= local_x < disp_w and 0.0 <= local_y < disp_h):
        return None

    col = min(cols - 1, int(local_x / disp_w * cols))
    row = min(rows - 1, int(local_y / disp_h * rows))
    return row, col


def stamp_disk(mask: np.ndarray, row: float, col: float, radius: float, value: int) -> None:
    """Set a filled disk of ``value`` (0 or 1) centered at ``(row, col)`` into ``mask`` in place.

    The disk is clipped to the array bounds, so a center near (or past) an edge simply paints the
    part that lands on the mask. A ``radius`` of 0 still paints the single center pixel.
    """
    height, width = mask.shape
    r0 = max(0, int(math.floor(row - radius)))
    r1 = min(height, int(math.ceil(row + radius)) + 1)
    c0 = max(0, int(math.floor(col - radius)))
    c1 = min(width, int(math.ceil(col + radius)) + 1)
    if r0 >= r1 or c0 >= c1:
        return
    # cast at the numpy boundary: np.arange(start, stop) / broadcast ops type as Unknown here.
    rr = cast("np.ndarray", np.arange(r0, r1)[:, None])
    cc = cast("np.ndarray", np.arange(c0, c1)[None, :])
    disk = cast("np.ndarray", (rr - row) ** 2 + (cc - col) ** 2 <= radius**2)
    mask[r0:r1, c0:c1][disk] = value


def stamp_stroke(
    mask: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    radius: float,
    value: int,
) -> None:
    """Stamp brush disks along the segment ``p0 -> p1`` so a drag paints a continuous band.

    Disks are spaced no further apart than the brush radius, so consecutive stamps overlap and the
    stroke leaves no gap even when the two endpoints are far apart (a fast drag).
    """
    r0, c0 = p0
    r1, c1 = p1
    distance = math.hypot(r1 - r0, c1 - c0)
    steps = max(1, math.ceil(distance / max(radius, 1.0)))
    for i in range(steps + 1):
        t = i / steps
        stamp_disk(mask, r0 + (r1 - r0) * t, c0 + (c1 - c0) * t, radius, value)


def raise_slider_to_floor(value: int, floor: int) -> int:
    """The blender floor applied when a stroke starts: never below ``floor``, otherwise unchanged."""
    return max(value, floor)


def mask_changed(working: np.ndarray, original: np.ndarray) -> bool:
    """Whether ``working`` differs from ``original`` (gates the Discard confirmation)."""
    return not np.array_equal(working, original)


class EditHistory:
    """A small undo/redo stack of mask snapshots for one editing session.

    Seeded with the initial mask. ``push`` records a new state and drops any redo branch (so a fresh
    stroke made after undoing discards the undone future). Snapshots are stored and returned as
    copies, so in-place stamping on a returned mask never corrupts the history.
    """

    def __init__(self, initial: np.ndarray) -> None:
        self._states: list[np.ndarray] = [np.array(initial, copy=True)]
        self._index = 0

    def push(self, mask: np.ndarray) -> None:
        del self._states[self._index + 1 :]
        self._states.append(np.array(mask, copy=True))
        self._index = len(self._states) - 1

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    def undo(self) -> np.ndarray:
        if self.can_undo():
            self._index -= 1
        return self.current()

    def redo(self) -> np.ndarray:
        if self.can_redo():
            self._index += 1
        return self.current()

    def current(self) -> np.ndarray:
        return np.array(self._states[self._index], copy=True)
