"""The side-preview label, with an in-place mask-editing mode (imports Qt).

In its normal state this is the plain overlay preview (``display``). In edit mode it owns the one
canonical working-mask array, paints it with the mouse (left = add, right = erase), and renders the
overlay directly from a cached base image so no per-stroke disk reads happen. The pure geometry and
mask math live in the Qt-free :mod:`ivcm.app.editing`.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QLabel

from ivcm.app.config import AppConfig
from ivcm.app.editing import map_to_mask, mask_changed, stamp_disk, stamp_stroke
from ivcm.app.qtimage import numpy_to_pixmap, record_pixmap
from ivcm.app.records import ImageRecord
from ivcm.app.render import overlay

PREVIEW_MAX = 500  # max side length of the side preview (shared with the main window's layout budget)


class EditablePreview(QLabel):
    """The overlay preview; in edit mode it paints the working mask under the mouse.

    ``edit_started`` fires on each stroke's first press so the main window can apply the blender
    floor; ``stroke_committed`` fires when a stroke that changed the mask ends so the main window can
    record an undo-history entry.
    """

    edit_started = Signal()
    stroke_committed = Signal()

    def __init__(self, config: AppConfig) -> None:
        super().__init__("Select an image")
        self._config = config
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(PREVIEW_MAX)

        self._editing = False
        self._base_image: np.ndarray | None = None
        self._working_mask: np.ndarray | None = None
        self._alpha = config.mask_alpha
        self._brush_display_radius = 8.0
        self._active_value: int | None = None  # 1 = add, 0 = erase, None = idle
        self._last_point: tuple[int, int] | None = None
        self._stroke_start: np.ndarray | None = None  # mask snapshot to detect a no-op stroke

    # --- normal mode -----------------------------------------------------

    def display(self, record: ImageRecord, config: AppConfig, alpha: float) -> None:
        """Show ``record``'s overlay (or plain image) scaled to fit — the non-editing preview."""
        pixmap = record_pixmap(record, config, alpha)
        self.setPixmap(
            pixmap.scaled(
                PREVIEW_MAX,
                PREVIEW_MAX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # --- edit mode -------------------------------------------------------

    def begin_edit(self, base_image: np.ndarray, working_mask: np.ndarray, alpha: float, brush_radius: float) -> None:
        """Enter edit mode owning ``working_mask``, rendering overlays from ``base_image``."""
        self._editing = True
        self._base_image = base_image
        self._working_mask = working_mask
        self._alpha = alpha
        self._brush_display_radius = brush_radius
        self._active_value = None
        self._last_point = None
        self._stroke_start = None
        # Right mouse is erase; suppress the context menu so a right-drag is never interrupted.
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._render_edit()

    def end_edit(self) -> None:
        """Leave edit mode and drop all in-progress references."""
        self._editing = False
        self._base_image = None
        self._working_mask = None
        self._active_value = None
        self._last_point = None
        self._stroke_start = None
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    def working_mask(self) -> np.ndarray | None:
        """The current canonical working mask (owned here while editing)."""
        return self._working_mask

    def set_working_mask(self, mask: np.ndarray) -> None:
        """Replace the working mask (used by undo/redo) and repaint."""
        self._working_mask = mask
        self._render_edit()

    def set_alpha(self, alpha: float) -> None:
        """Update the blend strength and repaint from the base image + working mask."""
        self._alpha = alpha
        if self._editing:
            self._render_edit()

    def set_brush_radius(self, display_radius: float) -> None:
        """Set the brush size (display pixels) applied by subsequent strokes."""
        self._brush_display_radius = display_radius

    def _render_edit(self) -> None:
        if self._base_image is None or self._working_mask is None:
            return
        rgb = overlay(self._base_image, self._working_mask, self._config.mask_color, self._alpha)
        pixmap = numpy_to_pixmap(rgb)
        self.setPixmap(
            pixmap.scaled(
                PREVIEW_MAX,
                PREVIEW_MAX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # --- painting --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._editing:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._active_value = 1
        elif event.button() == Qt.MouseButton.RightButton:
            self._active_value = 0
        else:
            super().mousePressEvent(event)
            return
        self._last_point = None
        self._stroke_start = None if self._working_mask is None else self._working_mask.copy()
        self.edit_started.emit()  # let the main window raise the blender to the visible floor first
        self._paint_at(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._editing and self._active_value is not None:
            self._paint_at(event.position())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._editing and self._active_value is not None:
            # Only a stroke that actually changed a pixel becomes an undo state — re-painting
            # already-painted (or erasing already-empty) pixels is a no-op, not a history entry.
            dirty = (
                self._working_mask is not None
                and self._stroke_start is not None
                and mask_changed(self._working_mask, self._stroke_start)
            )
            self._active_value = None
            self._last_point = None
            self._stroke_start = None
            if dirty:
                self.stroke_committed.emit()
        else:
            super().mouseReleaseEvent(event)

    def _paint_at(self, position: QPointF) -> None:
        mask = self._working_mask
        pixmap = self.pixmap()
        if mask is None or self._active_value is None or pixmap.isNull():
            return
        displayed = (pixmap.width(), pixmap.height())
        mapped = map_to_mask((position.x(), position.y()), displayed, (self.width(), self.height()), mask.shape)
        if mapped is None:
            return
        radius = self._brush_display_radius * (mask.shape[1] / displayed[0])
        if self._last_point is None:
            stamp_disk(mask, mapped[0], mapped[1], radius, self._active_value)
        else:
            stamp_stroke(mask, self._last_point, mapped, radius, self._active_value)
        self._last_point = mapped
        self._render_edit()
