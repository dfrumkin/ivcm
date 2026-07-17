"""Main window: directory batch, results table, side preview, and the synced top-3 window."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QModelIndex, QSize, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ivcm.app.batch import process_directory
from ivcm.app.inference import InferenceEngine, load_image
from ivcm.app.mask_slider import alpha_from_slider, make_mask_slider
from ivcm.app.overlay_dialog import OverlayDialog
from ivcm.app.qtimage import record_pixmap
from ivcm.app.records import ImageRecord, selection_after_toggle, sort_key, top3
from ivcm.app.summary_panel import SummaryPanel
from ivcm.app.table_model import ResultsTableModel
from ivcm.app.top3_window import Top3Window

log = logging.getLogger(__name__)

_PREVIEW = 500  # max side length of the side preview

# Fixed launch widths (px) for the left table's columns: a narrow checkbox column, a name column
# sized for typical filenames (long names are ellided, not widened), and content-width metrics.
_USE_WIDTH = 44
_NAME_WIDTH = 200
_METRIC_WIDTH = 90


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
        self._configure_columns()

        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumWidth(_PREVIEW)

        # Mask slider beneath the preview; controls only this window's preview (independent of
        # the top-3 window's slider).
        self.mask_slider = make_mask_slider(engine.config, self._on_mask_alpha_changed)

        # Summary of the checked images, above the preview; its height is fixed by its own rows,
        # so the preview keeps the rest of the pane.
        self.summary_panel = SummaryPanel()

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.addWidget(self.summary_panel)
        right_layout.addWidget(self.preview)
        right_layout.addWidget(self.mask_slider, alignment=Qt.AlignmentFlag.AlignHCenter)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.table, 2)
        layout.addWidget(right_pane, 1)
        self.setCentralWidget(central)

        self.top3_window = Top3Window(engine.config)
        self.top3_window.remove_requested.connect(self._on_remove_from_top3)

        self._build_toolbar()
        self.resize(self._initial_size())

    def _configure_columns(self) -> None:
        """Fixed launch widths: narrow "Use", a name column for typical names, content metrics."""
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        self.table.setColumnWidth(0, _USE_WIDTH)
        self.table.setColumnWidth(1, _NAME_WIDTH)
        for column in (2, 3, 4):
            self.table.setColumnWidth(column, _METRIC_WIDTH)
        # Names longer than the column are ellided rather than widening the table.
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

    def _initial_size(self) -> QSize:
        """Window size that shows all five table columns plus the preview, with no h-scrollbar.

        Budgets the sum of column widths, the vertical header, the table frame, and a
        vertical-scrollbar-width allowance (so columns still fit once enough rows show a vertical
        scrollbar), then adds the preview pane and layout margins. The height adds the summary
        panel's own height on top of the previous budget, so the panel costs the preview nothing.
        """
        columns = _USE_WIDTH + _NAME_WIDTH + 3 * _METRIC_WIDTH
        frame = 2 * self.table.frameWidth()
        vheader = self.table.verticalHeader().sizeHint().width()
        scrollbar = self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        table_width = columns + frame + vheader + scrollbar
        width = table_width + _PREVIEW + 60  # 60: central-layout margins/spacing slack
        # The panel has a fixed height, so its minimum is its real height (its size *hint* shrinks
        # to the placeholder while nothing is selected, which is not what the layout will give it).
        return QSize(width, 700 + self.summary_panel.minimumHeight())

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
        # The batch below pumps events, so the summary is cleared here too: otherwise the panel
        # would show the previous directory's numbers for the whole duration of the batch.
        self.records.clear()
        self.table.setCurrentIndex(QModelIndex())
        self.model.refresh()
        self.top3_window.refresh([])
        self.summary_panel.refresh([])

        progress = QProgressDialog("Processing images…", "", 0, len(files), self)
        progress.setCancelButton(None)  # uncancellable: the batch always runs to completion
        # Application-modal (not window-modal): setValue pumps events during the batch, and the
        # top-3 window has no parent — only application modality blocks its "Remove" clicks from
        # mutating self.records mid-build.
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)

        # The batch stays synchronous and modal (as before); process_directory runs the chunked,
        # batched, parallel-decoded pipeline and reports cumulative progress per chunk.
        # progress.setValue pumps events so the modal dialog repaints.
        with ThreadPoolExecutor(max_workers=self.engine.config.num_workers) as executor:
            records, skipped, seg_failed = process_directory(
                files,
                engine=self.engine,
                load=load_image,
                executor=executor,
                batch_size=self.engine.config.batch_size,
                on_progress=progress.setValue,
            )
        progress.setValue(len(files))  # also covers the empty-directory case (no chunks)

        self.records[:] = records
        self._resort_and_refresh(lambda _recs: None)  # fresh batch: leave no selection
        self._show_top3()

        message = f"Loaded {len(records)} image(s) from {directory}."
        if skipped:
            message += f" Skipped {len(skipped)} (load/classify failed): {', '.join(skipped)}."
        if seg_failed:
            message += f" Segmentation failed (left unchecked) for {len(seg_failed)}: {', '.join(seg_failed)}."
        self.statusBar().showMessage(message)

    # --- checkbox / top-3 ------------------------------------------------

    def _on_toggle(self, record: ImageRecord) -> None:
        """Checkbox changed: segment if newly checked and unmeasured, then re-sort and refresh.

        The prior current row is captured first (before segmentation/sort can move it). Checking
        follows the toggled record to its new position; unchecking keeps the prior row position
        (a failed check reverts to unchecked and so takes the keep-position branch).
        """
        prior_row = self.table.currentIndex().row()
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
        self._resort_and_refresh(lambda recs: selection_after_toggle(recs, record, record.is_nerve, prior_row))

    def _on_remove_from_top3(self, record: ImageRecord) -> None:
        """ "Remove" in the top-3 window is exactly an uncheck in the table (keeps row position)."""
        prior_row = self.table.currentIndex().row()
        record.is_nerve = False
        self._resort_and_refresh(lambda recs: selection_after_toggle(recs, record, False, prior_row))

    def _show_top3(self) -> None:
        self.top3_window.refresh(top3(self.records))
        self.top3_window.show()
        self.top3_window.raise_()

    # --- refresh / preview ----------------------------------------------

    def _resort_and_refresh(self, select: Callable[[list[ImageRecord]], int | None]) -> None:
        """Re-sort the records, rebuild the views, and select the row ``select`` returns.

        ``select`` is called with the freshly sorted record list and returns the row to select,
        or ``None`` for no selection.
        """
        self.records.sort(key=sort_key)
        target = select(self.records)
        self.model.refresh()
        self.top3_window.refresh(top3(self.records))
        self.summary_panel.refresh(self.records)
        self._apply_selection(target)
        self._update_preview()

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        self._update_preview()

    def _current_record(self) -> ImageRecord | None:
        index = self.table.currentIndex()
        if index.isValid() and 0 <= index.row() < len(self.records):
            return self.records[index.row()]
        return None

    def _apply_selection(self, target: int | None) -> None:
        if target is None:
            self.table.setCurrentIndex(QModelIndex())
            return
        self.table.setCurrentIndex(self.model.index(target, 1))

    def _on_mask_alpha_changed(self, _value: int) -> None:
        self._update_preview()

    def _update_preview(self) -> None:
        record = self._current_record()
        if record is None:
            self.preview.setText("Select an image")
            return
        alpha = alpha_from_slider(self.mask_slider.value())
        pixmap = record_pixmap(record, self.engine.config, alpha)
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
