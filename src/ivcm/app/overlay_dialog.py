"""Large, blocking overlay view (modal dialog opened on double-click / Enter)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from ivcm.app.config import AppConfig
from ivcm.app.qtimage import record_pixmap
from ivcm.app.records import ImageRecord

_LARGE = 900  # max side length of the large view


class OverlayDialog(QDialog):
    """Modal dialog showing one record large: its overlay if a mask exists, else the image."""

    def __init__(self, record: ImageRecord, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(record.name)
        self.setModal(True)

        label = QLabel(self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = record_pixmap(record, config)
        label.setPixmap(
            pixmap.scaled(
                _LARGE,
                _LARGE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        layout = QVBoxLayout(self)
        layout.addWidget(label)
