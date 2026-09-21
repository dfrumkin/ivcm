"""The synced top-3 window: overlays + metrics for the three top checked images."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ivcm.app.config import AppConfig
from ivcm.app.mask_slider import alpha_from_slider, make_mask_slider
from ivcm.app.qtimage import record_pixmap
from ivcm.app.records import ImageRecord
from ivcm.app.scaling_label import ScalingImageLabel

_TILE = 320  # minimum side length of each tile's image (tiles grow past this with the window)


class Top3Window(QWidget):
    """Shows the current top-3 records with a per-tile pin toggle and a "Don't Use" button.

    The pin toggle reads "Unpin" for a pinned image (asks the main window to unpin it) and "Pin"
    for an unpinned one (asks it to pin it, exactly like checking Pin in the main table — subject to
    the three-pin limit). "Don't Use" asks the main window to uncheck Use (which also unpins). Each
    mirrors the corresponding action in the main table; the button re-renders on the next
    :meth:`refresh`, so pinning flips it to "Unpin".

    The main window calls :meth:`refresh` with the current top-3 on every change, so the two
    windows stay in sync. The window remembers the last records it was given and its own mask
    slider value so a slider change can re-render the tiles locally (independent of the main
    window's slider).
    """

    pin_requested = Signal(object)  # emits the ImageRecord to pin
    unpin_requested = Signal(object)  # emits the ImageRecord to unpin
    dont_use_requested = Signal(object)  # emits the ImageRecord to uncheck (Use, and Pin if set)
    save_requested = Signal()  # the save shortcut (Cmd+S) was pressed while this window had focus

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("Top 3 Nerve Images")
        self._config = config
        self._records: list[ImageRecord] = []

        # The save shortcut only reaches the window that has focus, so mirror it here and let the
        # main window handle it (it gates on whether saving is currently allowed).
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self.save_requested)

        layout = QVBoxLayout(self)
        self._row = QHBoxLayout()
        layout.addLayout(self._row, 1)  # the tile row takes the vertical slack; the slider stays fixed
        self._slider = make_mask_slider(config, self._on_alpha_changed)
        layout.addWidget(self._slider, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._placeholder = QLabel("No nerve images selected.")
        self._row.addWidget(self._placeholder)

    def refresh(self, records: list[ImageRecord]) -> None:
        """Rebuild the tiles from the given (already top-3) records."""
        self._records = list(records)
        self._rebuild_tiles()

    def _on_alpha_changed(self, _value: int) -> None:
        self._rebuild_tiles()

    def _rebuild_tiles(self) -> None:
        self._clear()
        if not self._records:
            self._placeholder = QLabel("No nerve images selected.")
            self._row.addWidget(self._placeholder)
        else:
            alpha = alpha_from_slider(self._slider.value())
            for record in self._records:
                self._row.addWidget(self._build_tile(record, alpha), 1)  # tiles share width evenly
        self._apply_minimum_size()

    def _apply_minimum_size(self) -> None:
        """Set the window's minimum to its layout's own minimum, so the current tiles always fit.

        The layout minimum is derived from the real widgets (each tile image's ``_TILE`` minimum, the
        metrics text, the buttons, and the slider) plus spacing and margins, so it stays honest as the
        top-3 set changes rather than being a stale constant.
        """
        layout = self.layout()
        if layout is not None:
            layout.activate()
            self.setMinimumSize(layout.minimumSize())

    def _clear(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_tile(self, record: ImageRecord, alpha: float) -> QWidget:
        tile = QWidget()
        layout = QVBoxLayout(tile)

        image_label = ScalingImageLabel()
        image_label.setMinimumSize(_TILE, _TILE)  # grows past this with the window, never below
        image_label.set_source(record_pixmap(record, self._config, alpha))
        layout.addWidget(image_label, 1)  # the image takes the height not used by text/buttons

        metrics = record.metrics or {}
        text = (
            f"{record.name}\n"
            f"count: {metrics.get('count', 0.0):.0f}\n"
            f"density: {metrics.get('density', 0.0):.2f}\n"
            f"area: {metrics.get('area', 0.0):.2f}\n"
            f"perimeter: {metrics.get('perimeter', 0.0):.2f}"
        )
        info_label = QLabel(text)
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)

        buttons = QHBoxLayout()
        # A pinned image shows "Unpin"; an unpinned one shows "Pin" (checking Pin in the main table).
        pin_toggle = QPushButton("Unpin" if record.pinned else "Pin")
        dont_use = QPushButton("Don't Use")

        def emit_pin_toggle() -> None:  # `record` is this call's parameter — no late-binding capture
            (self.unpin_requested if record.pinned else self.pin_requested).emit(record)

        def emit_dont_use() -> None:
            self.dont_use_requested.emit(record)

        pin_toggle.clicked.connect(emit_pin_toggle)
        dont_use.clicked.connect(emit_dont_use)
        buttons.addWidget(pin_toggle)
        buttons.addWidget(dont_use)
        layout.addLayout(buttons)

        return tile
