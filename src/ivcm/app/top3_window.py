"""The synced top-3 window: overlays + metrics for the three top checked images."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ivcm.app.config import AppConfig
from ivcm.app.qtimage import record_pixmap
from ivcm.app.records import ImageRecord

_TILE = 320  # max side length of each tile's image


class Top3Window(QWidget):
    """Shows the current top-3 records; "Remove" asks the main window to uncheck that record.

    The window owns no state: the main window calls :meth:`refresh` with the current top-3 on
    every change, so the two windows stay in sync.
    """

    remove_requested = Signal(object)  # emits the ImageRecord to uncheck

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("Top 3 nerve images")
        self._config = config
        self._row = QHBoxLayout(self)
        self._placeholder = QLabel("No nerve images selected.")
        self._row.addWidget(self._placeholder)

    def refresh(self, records: list[ImageRecord]) -> None:
        """Rebuild the tiles from the given (already top-3) records."""
        self._clear()
        if not records:
            self._placeholder = QLabel("No nerve images selected.")
            self._row.addWidget(self._placeholder)
            return
        for record in records:
            self._row.addWidget(self._build_tile(record))

    def _clear(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_tile(self, record: ImageRecord) -> QWidget:
        tile = QWidget()
        layout = QVBoxLayout(tile)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = record_pixmap(record, self._config)
        image_label.setPixmap(
            pixmap.scaled(
                _TILE,
                _TILE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(image_label)

        metrics = record.metrics or {}
        text = (
            f"{record.name}\n"
            f"density: {metrics.get('density', 0.0):.2f}\n"
            f"area: {metrics.get('area', 0.0):.2f}\n"
            f"perimeter: {metrics.get('perimeter', 0.0):.2f}"
        )
        info_label = QLabel(text)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        remove = QPushButton("Remove")

        def emit_remove() -> None:  # `record` is this call's parameter — no late-binding capture
            self.remove_requested.emit(record)

        remove.clicked.connect(emit_remove)
        layout.addWidget(remove)

        return tile
