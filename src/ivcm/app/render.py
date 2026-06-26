"""Pure, Qt-free overlay compositing.

``overlay`` blends a binary mask (as a colour at a given alpha) onto a grayscale image and
returns a contiguous RGB ``uint8`` array. It imports no Qt, so it is importable and testable in
the base environment; the numpy -> ``QImage``/``QPixmap`` conversion lives in the view modules.
"""

from __future__ import annotations

import numpy as np


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Coerce a grayscale image (``[0, 1]`` float or ``[0, 255]``) to a 2D ``uint8`` array."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]  # single-channel HWC -> HW
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """Grayscale ``image`` as RGB with ``mask`` blended toward ``color`` at ``alpha``.

    The background (where ``mask == 0``) is the original grayscale; the foreground is
    ``(1 - alpha) * gray + alpha * color``. Returns a C-contiguous ``(H, W, 3)`` ``uint8`` array.
    """
    gray = _to_uint8_gray(image)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)

    fg = np.asarray(mask) != 0
    color_arr = np.asarray(color, dtype=np.float32)
    rgb[fg] = (1.0 - alpha) * rgb[fg] + alpha * color_arr

    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


def to_rgb(image: np.ndarray) -> np.ndarray:
    """The grayscale ``image`` as a plain C-contiguous ``(H, W, 3)`` ``uint8`` RGB array."""
    gray = _to_uint8_gray(image)
    return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
