"""numpy -> Qt image conversion for the view modules (imports Qt).

Kept out of :mod:`ivcm.app.render` (which stays Qt-free) so the pure compositing is testable
without PySide6 installed.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap

from ivcm.app.config import AppConfig
from ivcm.app.inference import load_image
from ivcm.app.records import ImageRecord
from ivcm.app.render import overlay, to_rgb


def numpy_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """Convert a contiguous ``(H, W, 3)`` ``uint8`` RGB array to a ``QPixmap``.

    Builds the ``QImage`` from a private ``bytes`` copy and then ``.copy()``s it, so the pixmap
    never aliases a numpy/buffer that could be freed underneath it.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    image = QImage(rgb.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def record_pixmap(record: ImageRecord, config: AppConfig, alpha: float | None = None) -> QPixmap:
    """The record's image as a pixmap: overlaid with its mask if one exists, else the plain image.

    ``alpha`` overrides the mask blend strength in ``[0, 1]`` (0 = no mask, 1 = full mask); when
    ``None`` it falls back to ``config.mask_alpha``.
    """
    image = load_image(record.path)
    if record.mask is not None:
        blend = config.mask_alpha if alpha is None else alpha
        rgb = overlay(image, record.mask, config.mask_color, blend)
    else:
        rgb = to_rgb(image)
    return numpy_to_pixmap(rgb)
