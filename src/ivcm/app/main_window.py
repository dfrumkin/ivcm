"""Main window: directory batch, results table, side preview, and the synced top-3 window."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QTableView,
    QToolBar,
    QWidget,
)

from ivcm.app.inference import InferenceEngine, load_image
from ivcm.app.overlay_dialog import OverlayDialog
from ivcm.app.qtimage import record_pixmap
from ivcm.app.records import ImageRecord, sort_key, top3
from ivcm.app.table_model import ResultsTableModel
from ivcm.app.top3_window import Top3Window

log = logging.getLogger(__name__)

_PREVIEW = 500  # max side length of the side preview


class MainWindow(QMainWindow):
    """Owns the record list, the inference engine, and both windows."""

    def __init__(self, engine: InferenceEngine) -> None:
        super().__init__()
        self.setWindowTitle("ivcm — clinical review")
        self.engine = engine
        self.records: list[ImageRecord] = []

        self.model = ResultsTableModel(self.records, self._on_toggle)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.activated.connect(self._open_large)
        self.table.selectionModel().currentRowChanged.connect(self._on_current_changed)

        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumWidth(_PREVIEW)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.table, 2)
        layout.addWidget(self.preview, 1)
        self.setCentralWidget(central)

        self.top3_window = Top3Window(engine.config)
        self.top3_window.remove_requested.connect(self._on_remove_from_top3)

        self._build_toolbar()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_action = QAction("Choose / switch directory", self)
        open_action.triggered.connect(self._open_directory)
        toolbar.addAction(open_action)

        top3_action = QAction("Top-3 window", self)
        top3_action.triggered.connect(self._show_top3)
        toolbar.addAction(top3_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.quit)
        toolbar.addAction(quit_action)

    def closeEvent(self, event: QCloseEvent) -> None:
        # The top-3 window is a separate top-level window; close it too so closing the main
        # window (X or Quit) actually exits the app rather than leaving it running headless.
        self.top3_window.close()
        super().closeEvent(event)

    # --- directory batch -------------------------------------------------

    def _open_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose image directory")
        if directory:
            self._load_directory(Path(directory))

    def _load_directory(self, directory: Path) -> None:
        extensions = self.engine.config.accepted_extensions
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)

        # Clear current state before the batch so no selection/row from the previous directory
        # carries over by position into the new one (matches the plan's "clears the records").
        self.records.clear()
        self.table.setCurrentIndex(QModelIndex())
        self.model.refresh()
        self.top3_window.refresh([])

        records: list[ImageRecord] = []
        skipped: list[str] = []
        seg_failed: list[str] = []
        progress = QProgressDialog("Processing images…", "", 0, len(files), self)
        progress.setCancelButton(None)  # uncancellable: the batch always runs to completion
        # Application-modal (not window-modal): setValue pumps events during the batch, and the
        # top-3 window has no parent — only application modality blocks its "Remove" clicks from
        # mutating self.records mid-build.
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)

        for i, path in enumerate(files):
            progress.setValue(i)  # pumps events for the modal dialog so it repaints
            record = self._process_file(path, skipped, seg_failed)
            if record is not None:
                records.append(record)
        progress.setValue(len(files))

        self.records[:] = records
        self._resort_and_refresh()
        self._show_top3()

        message = f"Loaded {len(records)} image(s) from {directory}."
        if skipped:
            message += f" Skipped {len(skipped)} (load/classify failed): {', '.join(skipped)}."
        if seg_failed:
            message += f" Segmentation failed (left unchecked) for {len(seg_failed)}: {', '.join(seg_failed)}."
        self.statusBar().showMessage(message)

    def _process_file(self, path: Path, skipped: list[str], seg_failed: list[str]) -> ImageRecord | None:
        """Classify (and segment if nerve) one file; ``None`` if it can't be loaded/classified."""
        try:
            image = load_image(path)
            is_nerve = self.engine.classify(image)
        except Exception:
            log.exception("Skipping %s (load/classify failed)", path.name)
            skipped.append(path.name)
            return None

        record = ImageRecord(path=path, is_nerve=is_nerve)
        if is_nerve:
            try:
                record.mask = self.engine.segment(image)
                record.metrics = self.engine.morphometry(record.mask)
            except Exception:
                log.exception("Segmentation failed for %s; leaving unchecked", path.name)
                record.is_nerve = False
                record.mask = None
                record.metrics = None
                seg_failed.append(path.name)
        return record

    # --- checkbox / top-3 ------------------------------------------------

    def _on_toggle(self, record: ImageRecord) -> None:
        """Checkbox changed: segment if newly checked and unmeasured, then re-sort and refresh."""
        if record.is_nerve and record.mask is None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                image = load_image(record.path)
                record.mask = self.engine.segment(image)
                record.metrics = self.engine.morphometry(record.mask)
            except Exception:
                log.exception("Segmentation failed for %s", record.name)
                record.is_nerve = False
                record.mask = None
                record.metrics = None
                self.statusBar().showMessage(f"Segmentation failed for {record.name}.")
            finally:
                QApplication.restoreOverrideCursor()
        self._resort_and_refresh()

    def _on_remove_from_top3(self, record: ImageRecord) -> None:
        """ "Remove" in the top-3 window is exactly an uncheck in the table."""
        record.is_nerve = False
        self._resort_and_refresh()

    def _show_top3(self) -> None:
        self.top3_window.refresh(top3(self.records))
        self.top3_window.show()
        self.top3_window.raise_()

    # --- refresh / preview ----------------------------------------------

    def _resort_and_refresh(self) -> None:
        selected = self._current_record()
        self.records.sort(key=sort_key)
        self.model.refresh()
        self.top3_window.refresh(top3(self.records))
        self._restore_selection(selected)
        self._update_preview()

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        self._update_preview()

    def _current_record(self) -> ImageRecord | None:
        index = self.table.currentIndex()
        if index.isValid() and 0 <= index.row() < len(self.records):
            return self.records[index.row()]
        return None

    def _restore_selection(self, record: ImageRecord | None) -> None:
        if record is None:
            return
        try:
            row = self.records.index(record)
        except ValueError:
            return
        self.table.setCurrentIndex(self.model.index(row, 1))

    def _update_preview(self) -> None:
        record = self._current_record()
        if record is None:
            self.preview.setText("Select an image")
            return
        pixmap = record_pixmap(record, self.engine.config)
        self.preview.setPixmap(
            pixmap.scaled(
                _PREVIEW,
                _PREVIEW,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _open_large(self, index: QModelIndex) -> None:
        if not index.isValid() or not (0 <= index.row() < len(self.records)):
            return
        OverlayDialog(self.records[index.row()], self.engine.config, self).exec()
