"""Shared mask-strength slider for the preview and top-3 windows (imports Qt).

A horizontal slider over ``0..100`` mapped to a blend alpha in ``[0, 1]`` (0 = no mask,
1 = full mask). Both windows build identical, independent sliders via :func:`make_mask_slider`
and convert positions with :func:`alpha_from_slider`.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider

from ivcm.app.config import AppConfig

_SLIDER_MAX = 100  # integer slider range is 0.._SLIDER_MAX, mapped to alpha 0.0..1.0
_SLIDER_WIDTH = 160  # keep the slider small so focus stays on the images/table


def alpha_from_slider(value: int) -> float:
    """Map an integer slider position in ``0.._SLIDER_MAX`` to a blend alpha in ``[0, 1]``."""
    return value / _SLIDER_MAX


def slider_from_alpha(alpha: float) -> int:
    """Map a blend alpha in ``[0, 1]`` to the nearest integer slider position."""
    return round(alpha * _SLIDER_MAX)


def make_mask_slider(config: AppConfig, on_change: Callable[[int], None]) -> QSlider:
    """A small horizontal mask slider initialised to ``config.mask_alpha``.

    ``on_change`` is connected to ``valueChanged`` and receives the new integer position.
    """
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, _SLIDER_MAX)
    slider.setValue(slider_from_alpha(config.mask_alpha))
    slider.setFixedWidth(_SLIDER_WIDTH)
    slider.setToolTip("Mask strength")
    slider.valueChanged.connect(on_change)
    return slider
