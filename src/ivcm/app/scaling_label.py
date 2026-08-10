"""A ``QLabel`` that scales its image to the space the layout gives it (imports Qt).

Holds the full-resolution source pixmap and rescales it (keeping aspect ratio, centered) whenever
the widget is resized or a new source is set, so the image grows with its window instead of staying
a fixed size. Every rescale is computed from the stored source — not from the previously displayed
(already-scaled) pixmap — so repeated resizes never accumulate scaling blur.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy

_MIN_SIDE = 40  # a small default floor so the widget stays valid; callers set their own real minimum


class ScalingImageLabel(QLabel):
    """A centered image label that rescales its source pixmap to the current widget size.

    Callers push a full-resolution pixmap with :meth:`set_source` and a text placeholder with
    :meth:`show_message`; the two are mutually exclusive, so the label never holds mixed
    text+pixmap state. The per-use minimum size is the caller's responsibility (this class carries
    only a small default floor).
    """

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ignore the pixmap's own size hint so the current (scaled) image can never feed back into
        # layout sizing; the label then fills whatever the layout's stretch gives it, down to the
        # caller-set minimum below (a small default floor keeps the bare widget valid).
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(_MIN_SIDE, _MIN_SIDE)

    def set_source(self, pixmap: QPixmap) -> None:
        """Show ``pixmap`` (full resolution), scaled to fit the current size; clears any placeholder."""
        self._source = pixmap
        self.setText("")  # drop any placeholder text so it never shows behind the pixmap
        self._rescale()

    def show_message(self, text: str) -> None:
        """Show placeholder ``text``; clears the stored source and the displayed pixmap."""
        self._source = None
        self.clear()  # drop the displayed pixmap so no stale image lingers behind the text
        self.setText(text)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        """Rescale the stored source to the current size; a no-op with no source or a degenerate size."""
        source = self._source
        if source is None:
            return
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return
        self.setPixmap(
            source.scaled(
                width,
                height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
