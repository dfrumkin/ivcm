"""Main window: directory batch, results table, side preview, and the synced top-3 window."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from PySide6.QtCore import QModelIndex, QSize, Qt
from PySide6.QtGui import QAction, QCloseEvent, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QStackedWidget,
    QStyle,
    QTableView,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ivcm.app import persistence
from ivcm.app.batch import ensure_all_square, process_directory
from ivcm.app.editing import EditHistory, mask_changed, raise_slider_to_floor
from ivcm.app.inference import InferenceEngine, load_image
from ivcm.app.mask_slider import alpha_from_slider, make_mask_slider, slider_from_alpha
from ivcm.app.overlay_dialog import OverlayDialog
from ivcm.app.preview import PREVIEW_MAX, EditablePreview
from ivcm.app.records import MAX_PINNED, ImageRecord, pinned_count, selection_after_toggle, sort_key, top3
from ivcm.app.save_dialog import SaveDirectoryDialog
from ivcm.app.summary_panel import SummaryPanel
from ivcm.app.table_model import ResultsTableModel
from ivcm.app.top3_window import Top3Window
from ivcm.data import NonSquareImageError

log = logging.getLogger(__name__)

_PREVIEW = PREVIEW_MAX  # minimum side length of the side preview (also its launch-size budget)

# Brush-size slider: radius in display pixels. The minimum paints roughly a single pixel; the
# maximum is deliberately small so the brush stays a detail tool.
_BRUSH_MIN = 1
_BRUSH_MAX = 8
_BRUSH_DEFAULT = 2

# Fixed launch widths (px) for the left table's columns: two narrow checkbox columns, a name column
# sized for typical filenames (long names are ellided, not widened), and content-width metrics.
_PIN_WIDTH = 44
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

        # Persistence session state (see the Load/Save methods). ``_image_dir`` is the loaded image
        # directory; ``_loaded_label_dir`` is the label dir a Load came from (drives the Save As
        # pre-fill); ``_save_target`` is the confirmed directory saved to this session (drives silent
        # re-saves); ``_dirty`` tracks unsaved changes for the close guard.
        self._image_dir: Path | None = None
        self._loaded_label_dir: Path | None = None
        self._save_target: Path | None = None
        self._dirty = False

        # Mask-editing state (see the "editing mode" methods). The preview widget owns the working
        # mask while editing; the main window holds only the surrounding session state.
        self._editing = False
        self._edit_record: ImageRecord | None = None
        self._original_mask: np.ndarray | None = None
        self._history: EditHistory | None = None

        self.model = ResultsTableModel(
            self.records, self._on_toggle, self._on_pin, self._on_pin_rejected, self._mark_dirty
        )
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.activated.connect(self._open_large)
        self.table.selectionModel().currentRowChanged.connect(self._on_current_changed)
        self._configure_columns()
        # Fix the table to its column budget so window-resize slack flows to the preview (not the
        # table), and so it is never squeezed below its columns (which would show an h-scrollbar).
        self.table.setFixedWidth(self._table_width_budget())

        self.preview = EditablePreview(engine.config)
        self.preview.edit_started.connect(self._on_edit_started)
        self.preview.stroke_committed.connect(self._on_stroke_committed)

        # Mask slider beneath the preview; controls only this window's preview (independent of
        # the top-3 window's slider). It doubles as the "blender" during an edit.
        self.mask_slider = make_mask_slider(engine.config, self._on_mask_alpha_changed)

        # Summary of the checked images, above the preview; its height is fixed by its own rows,
        # so the preview keeps the rest of the pane.
        self.summary_panel = SummaryPanel()

        self._controls = self._build_edit_controls()

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.addWidget(self.summary_panel)
        right_layout.addWidget(self.preview, 1)  # the preview takes the vertical slack in the pane
        right_layout.addWidget(self.mask_slider, alignment=Qt.AlignmentFlag.AlignHCenter)
        right_layout.addWidget(self._controls, alignment=Qt.AlignmentFlag.AlignHCenter)

        central = QWidget()
        layout = QHBoxLayout(central)
        self._central_layout = layout  # kept so the window minimum can read the layout's own minimum
        layout.addWidget(self.table)
        layout.addWidget(right_pane, 1)  # horizontal slack flows to the right pane, enlarging the preview
        self.setCentralWidget(central)

        self.top3_window = Top3Window(engine.config)
        self.top3_window.pin_requested.connect(self._on_pin_from_top3)
        self.top3_window.unpin_requested.connect(self._on_unpin_from_top3)
        self.top3_window.dont_use_requested.connect(self._on_dont_use_from_top3)
        self.top3_window.save_requested.connect(self._on_save_shortcut)

        self._build_toolbar()
        self._update_preview()  # nothing selected yet: placeholder, Edit + blender disabled
        # Set the floor before the launch size so the window can never be shrunk below "everything
        # fits" (computed once the whole UI, including the toolbar and status bar, is built).
        self.setMinimumSize(self._minimum_size())
        self.resize(self._initial_size())

    def _configure_columns(self) -> None:
        """Fixed launch widths: narrow "Pin"/"Use", a name column for typical names, content metrics."""
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        self.table.setColumnWidth(0, _PIN_WIDTH)
        self.table.setColumnWidth(1, _USE_WIDTH)
        self.table.setColumnWidth(2, _NAME_WIDTH)
        for column in (3, 4, 5, 6):
            self.table.setColumnWidth(column, _METRIC_WIDTH)
        # Names longer than the column are ellided rather than widening the table.
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)

    def _table_width_budget(self) -> int:
        """Width that shows all seven columns with no h-scrollbar: columns plus the table's chrome.

        Budgets the sum of column widths, the table frame, the vertical header, and a
        vertical-scrollbar-width allowance (so columns still fit once enough rows show a vertical
        scrollbar). The table is fixed to this width so it neither grows into (nor is squeezed
        below) its column budget as the window resizes.
        """
        columns = _PIN_WIDTH + _USE_WIDTH + _NAME_WIDTH + 4 * _METRIC_WIDTH
        frame = 2 * self.table.frameWidth()
        vheader = self.table.verticalHeader().sizeHint().width()
        scrollbar = self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        return columns + frame + vheader + scrollbar

    def _initial_size(self) -> QSize:
        """Launch size showing the full table plus a comfortably-sized preview.

        The preview grows with the window past this, but this is where it opens. The height adds the
        summary panel's own height on top of the preview budget, so the panel costs the preview
        nothing.
        """
        width = self._table_width_budget() + _PREVIEW + 60  # 60: central-layout margins/spacing slack
        # The panel has a fixed height, so its minimum is its real height (its size *hint* shrinks
        # to the placeholder while nothing is selected, which is not what the layout will give it).
        return QSize(width, 700 + self.summary_panel.minimumHeight())

    def _minimum_size(self) -> QSize:
        """Smallest size that still fits everything: the central layout's minimum plus window chrome.

        The central layout's own minimum already accounts for the fixed-width table, the preview's
        minimum side, the summary panel, the mask slider, and the edit controls, plus spacing and
        margins. Add the toolbar and status-bar heights so the "everything fits" floor holds for the
        whole window, not just the central area.
        """
        central = self._central_layout.minimumSize()
        chrome = self.toolbar.sizeHint().height() + self.statusBar().sizeHint().height()
        return QSize(central.width(), central.height() + chrome)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        self.toolbar = toolbar  # kept so editing mode can grey out the "top menu"
        self.addToolBar(toolbar)

        # "Load" starts the load flow; it is highlighted (bold) until a dataset is loaded/created so
        # the doctor sees where to begin.
        self._load_action = QAction("Load", self)
        self._load_action.triggered.connect(self._on_load)
        toolbar.addAction(self._load_action)

        # "Save" (Cmd+S) writes the current results; disabled until there is something to save.
        self._save_action = QAction("Save", self)
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.setEnabled(False)
        self._save_action.triggered.connect(self._on_save)
        toolbar.addAction(self._save_action)

        # "Top 3" opens the synced window; disabled until there is data to show.
        self._top3_action = QAction("Top 3", self)
        self._top3_action.setEnabled(False)
        self._top3_action.triggered.connect(self._show_top3)
        toolbar.addAction(self._top3_action)

        # Route Quit through the window's close (and thus the unsaved-changes guard); a direct
        # QApplication.quit() exits the event loop without delivering a close event.
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        toolbar.addAction(quit_action)

        self._set_load_highlight(True)

    def _build_edit_controls(self) -> QStackedWidget:
        """The Edit-button ⇄ edit-toolbar container shown under the preview.

        Page 0 is the plain "Edit" button (non-editing); page 1 is the compact icon toolbar shown
        while editing, with three groups: Keep / Discard, Undo / Redo, and the brush-size control
        (a brush icon plus a slider).
        """
        stack = QStackedWidget()

        # Page 0: the Edit button.
        normal = QWidget()
        normal_layout = QHBoxLayout(normal)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        self._edit_button = QPushButton("Edit")
        self._edit_button.setEnabled(False)  # enabled once a row is selected
        self._edit_button.clicked.connect(self._enter_edit)
        normal_layout.addWidget(self._edit_button)

        # Page 1: the editing toolbar.
        editing = QWidget()
        edit_layout = QHBoxLayout(editing)
        edit_layout.setContentsMargins(0, 0, 0, 0)

        # "Keep" commits the mask in memory (not to disk), so it uses an apply/checkmark icon.
        self._keep_button = self._icon_button(
            "dialog-ok-apply", QStyle.StandardPixmap.SP_DialogApplyButton, "Keep", self._keep_edit
        )
        self._discard_button = self._icon_button(
            "edit-delete", QStyle.StandardPixmap.SP_DialogDiscardButton, "Discard", self._discard_edit
        )
        self._undo_button = self._icon_button("edit-undo", QStyle.StandardPixmap.SP_ArrowBack, "Undo", self._undo_edit)
        self._redo_button = self._icon_button(
            "edit-redo", QStyle.StandardPixmap.SP_ArrowForward, "Redo", self._redo_edit
        )

        self._brush_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_slider.setRange(_BRUSH_MIN, _BRUSH_MAX)
        self._brush_slider.setValue(_BRUSH_DEFAULT)
        self._brush_slider.setFixedWidth(100)
        self._brush_slider.setToolTip("Brush size")
        self._brush_slider.valueChanged.connect(self._on_brush_changed)

        edit_layout.addWidget(self._keep_button)
        edit_layout.addWidget(self._discard_button)
        edit_layout.addWidget(self._separator())
        edit_layout.addWidget(self._undo_button)
        edit_layout.addWidget(self._redo_button)
        edit_layout.addWidget(self._separator())
        edit_layout.addWidget(self._brush_icon_label())
        edit_layout.addWidget(self._brush_slider)

        stack.addWidget(normal)  # index 0
        stack.addWidget(editing)  # index 1
        return stack

    def _icon_button(
        self, theme: str, fallback: QStyle.StandardPixmap, tip: str, on_click: Callable[[], object]
    ) -> QToolButton:
        """A small icon tool button, preferring a themed icon and falling back to a standard pixmap."""
        icon = QIcon.fromTheme(theme)
        if icon.isNull():
            icon = self.style().standardIcon(fallback)
        button = QToolButton()
        button.setIcon(icon)
        button.setToolTip(tip)
        button.clicked.connect(on_click)
        return button

    def _brush_icon_label(self) -> QLabel:
        """A small brush icon beside the brush slider so the slider's purpose is obvious.

        Prefers a themed brush icon; falls back to a drawn filled-circle "dot" so it still shows on
        platforms without an icon theme (e.g. macOS).
        """
        size = 16
        icon = QIcon.fromTheme("draw-brush")
        if icon.isNull():
            icon = QIcon.fromTheme("tool-brush")
        label = QLabel()
        label.setPixmap(icon.pixmap(size, size) if not icon.isNull() else self._brush_dot_pixmap(size))
        label.setToolTip("Brush size")
        return label

    @staticmethod
    def _brush_dot_pixmap(size: int) -> QPixmap:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.GlobalColor.gray)
        margin = size // 4
        painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)
        painter.end()
        return pixmap

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def closeEvent(self, event: QCloseEvent) -> None:
        # Guard against losing unsaved changes on close (window-X or Quit). The top-3 window is
        # closed only on a path that actually exits, so a Cancel leaves the app exactly as it was.
        if self._resolve_unsaved_changes():
            self.top3_window.close()
            super().closeEvent(event)
        else:
            event.ignore()

    def _resolve_unsaved_changes(self) -> bool:
        """Guard unsaved changes before a close or a new Load; ``True`` to proceed, ``False`` to abort.

        With no unsaved changes, proceed immediately. Otherwise present the Save/Discard/Cancel
        prompt: **Save** finishes any in-progress edit and saves (proceeding only if that succeeds),
        **Discard** drops any in-progress edit and proceeds, **Cancel** aborts (stay exactly as we
        were). Shared by ``closeEvent`` and ``_on_load`` so a new Load is treated like a close.
        """
        if not self._has_unsaved_changes():
            return True
        choice = self._ask_save_changes()
        if choice == "cancel":
            return False
        if choice == "save":
            return self._finish_edit_and_save()  # False if the edit-commit or the save was aborted
        # Discard: drop an in-progress edit; any committed changes are simply left to be replaced.
        if self._editing:
            self._exit_edit()
            self._update_preview()
        return True

    def _has_unsaved_changes(self) -> bool:
        """Whether there are unsaved changes: committed edits (``_dirty``) or a modified live edit."""
        if self._dirty:
            return True
        if self._editing:
            working = self.preview.working_mask()
            if working is not None and self._original_mask is not None and mask_changed(working, self._original_mask):
                return True
        return False

    def _ask_save_changes(self) -> str:
        """The "Save changes?" prompt; returns ``"save"`` / ``"discard"`` / ``"cancel"``."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("ivcm")
        box.setText("Save changes?")
        save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        return "cancel"

    def _finish_edit_and_save(self) -> bool:
        """Finish any in-progress edit, then save; ``True`` only if both succeed (else abort)."""
        if self._editing:
            working = self.preview.working_mask()
            changed = (
                working is not None and self._original_mask is not None and mask_changed(working, self._original_mask)
            )
            if changed:
                if not self._keep_edit():  # morphometry failed: stay in edit, abort
                    return False
            else:
                # An untouched edit: leaving it via Keep would set an all-zero mask and force Use, an
                # unwanted mutation — just leave edit mode and save the already-committed state.
                self._exit_edit()
                self._update_preview()
        return self._on_save()

    def _mark_dirty(self) -> None:
        """Mark the in-memory state as having unsaved changes (drives the close guard)."""
        self._dirty = True

    def _set_load_highlight(self, on: bool) -> None:
        """Bold (or un-bold) the Load toolbar button to signal it is the starting point."""
        # widgetForAction is typed non-optional in the stubs but can be None before the button
        # widget is realized; guard defensively.
        button = cast("QWidget | None", self.toolbar.widgetForAction(self._load_action))
        if button is not None:
            font = button.font()
            font.setBold(on)
            button.setFont(font)

    # --- load / create ---------------------------------------------------

    def _on_load(self) -> None:
        """Load flow: choose an image directory, then load existing labels or create fresh ones."""
        # A new Load replaces the current dataset, so guard unsaved changes just like a close: the
        # user may Save (updating the last-save strip), Discard, or Cancel (stay where we were).
        if not self._resolve_unsaved_changes():
            return
        start_dir = str(self._image_dir) if self._image_dir is not None else ""
        while True:
            directory = QFileDialog.getExistingDirectory(self, "Choose image directory", start_dir)
            if not directory:
                return  # abort: leave the current view unchanged
            image_dir = Path(directory)
            start_dir = str(image_dir)  # reopen here if we come back from the labels dialog
            if self._labels_choice_loop(image_dir):
                return

    def _labels_choice_loop(self, image_dir: Path) -> bool:
        """The "Load or create labels?" step; ``True`` when done, ``False`` to reopen the image chooser."""
        while True:
            choice = self._ask_load_or_create()
            if choice == "cancel":
                return False  # back to the image chooser
            if choice == "create":
                self._create_labels(image_dir)
                return True
            if self._label_dir_loop(image_dir):
                return True  # loaded successfully
            # the label-directory chooser was cancelled: back to "Load or create labels?"

    def _label_dir_loop(self, image_dir: Path) -> bool:
        """Choose and validate a label directory; ``True`` on success, ``False`` if cancelled."""
        # Start alongside the image directory (its parent), where a sibling label directory lives,
        # rather than inside the image directory itself.
        start_dir = str(image_dir.parent)
        while True:
            directory = QFileDialog.getExistingDirectory(self, "Choose label directory", start_dir)
            if not directory:
                return False  # back to "Load or create labels?"
            label_dir = Path(directory)
            start_dir = str(label_dir)
            try:
                records, normalized = persistence.load_labels(
                    image_dir, label_dir, self.engine.config.accepted_extensions
                )
            except persistence.BadLabelFormatError as exc:
                log.info("Bad label format in %s: %s", label_dir, exc)
                self._info_box("Bad Format", "This directory is not a valid label directory.")
                continue  # reopen the label-directory chooser
            except persistence.LabelMismatchError as exc:
                log.info("Label mismatch in %s: %s", label_dir, exc)
                self._info_box("Load error", "These labels do not match the images (names or resolutions differ).")
                continue
            self._apply_loaded(records, image_dir, label_dir, normalized)
            return True

    def _ask_load_or_create(self) -> str:
        """The "Load or create labels?" prompt; returns ``"load"`` / ``"create"`` / ``"cancel"``."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("ivcm")
        box.setText("Load or create labels?")
        load = box.addButton("Load", QMessageBox.ButtonRole.AcceptRole)
        create = box.addButton("Create", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is load:
            return "load"
        if clicked is create:
            return "create"
        return "cancel"

    def _create_labels(self, image_dir: Path) -> None:
        """Run the classify/segment batch on ``image_dir``, then record fresh (unsaved) dataset state."""
        if self._run_batch(image_dir):
            self._mark_dataset_state(image_dir, loaded_label_dir=None, dirty=True)

    def _apply_loaded(self, records: list[ImageRecord], image_dir: Path, label_dir: Path, normalized: bool) -> None:
        """Apply a loaded label directory: install the records and record the dataset state."""
        self.records[:] = records
        self._resort_and_refresh(lambda _recs: None)  # loaded dataset: leave no selection
        self._show_top3()
        self._mark_dataset_state(image_dir, loaded_label_dir=label_dir, dirty=normalized)

    def _mark_dataset_state(self, image_dir: Path, loaded_label_dir: Path | None, dirty: bool) -> None:
        """Record the dataset session state shared by Create and Load, and enable Save."""
        self._image_dir = image_dir
        self._loaded_label_dir = loaded_label_dir
        self._save_target = None  # first Save always asks for a location (load source may be read-only)
        self._dirty = dirty
        self._set_load_highlight(False)
        self._save_action.setEnabled(True)
        self._top3_action.setEnabled(True)
        # The status strip reflects the current dataset's save state; a freshly loaded/created
        # dataset has not been saved this session, so clear any prior save message.
        self.statusBar().clearMessage()

    # --- directory batch -------------------------------------------------

    def _run_batch(self, directory: Path) -> bool:
        """Classify/segment/measure ``directory`` into ``self.records``; ``False`` if a non-square abort."""
        extensions = self.engine.config.accepted_extensions
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions)

        # Enforce the square-input invariant across the whole directory *before* touching any state:
        # a non-square image is fatal (all inputs are guaranteed square), so log it, show a dialog,
        # and exit — leaving the previous view intact. Done before clearing so a fatal exit does not
        # leave a half-cleared UI. Unreadable files are not fatal here (see ``ensure_all_square``).
        try:
            ensure_all_square(files)
        except NonSquareImageError as exc:
            log.error("Non-square image in %s: %s", directory, exc)
            QMessageBox.critical(self, "ivcm — non-square image", str(exc))
            QApplication.exit(1)  # schedule non-zero exit; return so no state is cleared / batch run
            return False

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
        # top-3 window has no parent — only application modality blocks its Unpin / Don't Use clicks
        # from mutating self.records mid-build.
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

        # The status strip is reserved for the current dataset's save state, so batch problems are
        # surfaced in a dialog (only when there are any) rather than left in the strip.
        if skipped or seg_failed:
            parts: list[str] = []
            if skipped:
                parts.append(f"Skipped {len(skipped)} (load/classify failed): {', '.join(skipped)}.")
            if seg_failed:
                parts.append(f"Segmentation failed (left unchecked) for {len(seg_failed)}: {', '.join(seg_failed)}.")
            self._info_box("Some images were not processed", " ".join(parts))
        return True

    # --- save ------------------------------------------------------------

    def _on_save_shortcut(self) -> None:
        """The save shortcut fired in the top-3 window; save only when the Save action is enabled.

        Mirrors the main window's Save action gating (no data loaded, or an edit in progress), which
        a shortcut on a separate window would otherwise bypass.
        """
        if self._save_action.isEnabled():
            self._on_save()

    def _on_save(self) -> bool:
        """Save the current records; ``True`` if written, ``False`` if cancelled or failed."""
        if self._save_target is not None:
            # Silent re-save, but never delete a directory that is no longer in our label-dir shape.
            if not self._save_target.exists():
                return self._write_to(self._save_target)
            if persistence.is_overwritable_label_dir(self._save_target):
                return self._write_to(self._save_target)
            # The save target now holds non-label content: fall through to the interactive flow.
        return self._save_as()

    def _save_as(self) -> bool:
        """The first-save "Save As" interaction: choose a target directory and handle existing ones."""
        if self._loaded_label_dir is not None:
            start_dir = self._loaded_label_dir.parent
            suggested = self._loaded_label_dir.name
        elif self._image_dir is not None:
            start_dir = self._image_dir.parent
            suggested = f"{self._image_dir.name}_labels"
        else:
            return False  # nothing loaded: nothing to save
        while True:
            target = self._ask_save_target(start_dir, suggested)
            if target is None:
                return False  # cancelled
            if not target.exists():
                return self._write_to(target)
            if persistence.is_overwritable_label_dir(target):
                if self._confirm_overwrite(target):
                    return self._write_to(target)
            else:
                self._info_box(
                    "Cannot overwrite",
                    "The target directory already exists and is not a label directory, so it will not be overwritten.",
                )
            start_dir, suggested = target.parent, target.name  # reopen where we left off

    def _ask_save_target(self, start_dir: Path, suggested: str) -> Path | None:
        """Single-window Save dialog (navigate + type a name); the chosen path or ``None``.

        A plain file dialog in save mode reinterprets a typed name that matches an existing folder as
        "open it", so it can never return an existing directory for overwriting. ``SaveDirectoryDialog``
        keeps one always-"Save" button, so an existing name comes back as a normal path and the
        overwrite/update flow works.
        """
        dialog = SaveDirectoryDialog(self, start_dir, suggested)
        if dialog.exec():
            return dialog.selected_path()
        return None

    def _write_to(self, target: Path) -> bool:
        """Write the records to ``target``; on success confirm it and clear the dirty flag."""
        try:
            persistence.write_labels(target, self.records)
        except Exception:
            log.exception("Save failed for %s", target)
            self._info_box("Save failed", f"Could not save to {target}.")
            return False
        self._save_target = target
        self._dirty = False
        # Include a second-precision timestamp so each save updates the message (even a silent
        # re-save to the same target) and the last-save time is visible at a glance.
        self.statusBar().showMessage(f"Saved labels to {target} at {datetime.now():%Y-%m-%d %H:%M:%S}.")
        return True

    def _confirm_overwrite(self, target: Path) -> bool:
        """The "target directory already exists" Overwrite / Cancel prompt for an our-format dir."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("ivcm")
        box.setText("The target directory already exists")
        box.setInformativeText(str(target))
        overwrite = box.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is overwrite

    def _info_box(self, title: str, text: str) -> None:
        """A single-OK message box (bad-format / mismatch / dangerous-overwrite / save-failure)."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    # --- checkbox / top-3 ------------------------------------------------

    def _ensure_segmented(self, record: ImageRecord) -> bool:
        """Segment ``record`` (and compute its metrics) if not already done; ``False`` on failure.

        Cached results (``mask`` already present) are a no-op success. On a segmentation error the
        cached fields are cleared and a status message is shown; the caller reverts the checkbox.
        """
        if record.mask is not None:
            return True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            image = load_image(record.path)
            record.mask, record.metrics = self.engine.analyze(image)
            return True
        except Exception:
            log.exception("Segmentation failed for %s", record.name)
            record.mask = None
            record.metrics = None
            self.statusBar().showMessage(f"Segmentation failed for {record.name}.")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    def _on_toggle(self, record: ImageRecord) -> None:
        """Use changed: segment if newly checked and unmeasured, then re-sort and refresh.

        The prior current row is captured first (before segmentation/sort can move it). Checking
        follows the toggled record to its new position; unchecking keeps the prior row position
        (a failed check reverts to unchecked and so takes the keep-position branch).
        """
        prior_row = self.table.currentIndex().row()
        if record.is_nerve and not self._ensure_segmented(record):
            record.is_nerve = False
            record.pinned = False  # a pin can only ride on a use, so keep them consistent
        self._resort_and_refresh(lambda recs: selection_after_toggle(recs, record, record.is_nerve, prior_row))

    def _on_pin(self, record: ImageRecord, now_checked: bool) -> None:
        """Pin changed: on a successful pin the record is used and measured; re-sort and refresh.

        Pinning forces Use (already set by the model), so segment if needed; a failure reverts both
        Pin and Use. A successful pin follows the record to the top; an unpin (or a failed pin)
        keeps the prior row position.
        """
        prior_row = self.table.currentIndex().row()
        if now_checked and not self._ensure_segmented(record):
            record.pinned = False
            record.is_nerve = False
        self._resort_and_refresh(
            lambda recs: selection_after_toggle(recs, record, now_checked and record.pinned, prior_row)
        )

    def _on_pin_rejected(self) -> None:
        """A fourth pin was attempted: tell the user the limit.

        Builds the box explicitly and forces the warning pixmap: on macOS the message box is
        presented as a native alert, which otherwise falls back to the (generic, unbundled) app
        icon instead of a warning glyph.
        """
        box = QMessageBox(self)
        box.setWindowTitle("ivcm")
        box.setText("At most 3 images can be pinned!")
        box.setIconPixmap(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(64, 64))
        box.exec()

    def _on_pin_from_top3(self, record: ImageRecord) -> None:
        """ "Pin" in the top-3 window is exactly checking Pin in the table (subject to the limit).

        Enforces the three-pin cap (warning instead of pinning a fourth), forces Use and segments if
        needed, then follows the record to its new top position.
        """
        if not record.pinned and pinned_count(self.records) >= MAX_PINNED:
            self._on_pin_rejected()
            return
        self._mark_dirty()
        prior_row = self.table.currentIndex().row()
        record.pinned = True
        record.is_nerve = True  # pinning forces "use"
        if not self._ensure_segmented(record):
            record.pinned = False
            record.is_nerve = False
        self._resort_and_refresh(lambda recs: selection_after_toggle(recs, record, record.pinned, prior_row))

    def _on_unpin_from_top3(self, record: ImageRecord) -> None:
        """ "Unpin" in the top-3 window is exactly unchecking Pin in the table (keeps row position)."""
        self._mark_dirty()
        prior_row = self.table.currentIndex().row()
        record.pinned = False
        self._resort_and_refresh(lambda recs: selection_after_toggle(recs, record, False, prior_row))

    def _on_dont_use_from_top3(self, record: ImageRecord) -> None:
        """ "Don't Use" is exactly unchecking Use in the table (also unpins; keeps row position)."""
        self._mark_dirty()
        prior_row = self.table.currentIndex().row()
        record.is_nerve = False
        record.pinned = False
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
        self.table.setCurrentIndex(self.model.index(target, 2))

    def _on_mask_alpha_changed(self, value: int) -> None:
        # While editing, the slider is the blender for the in-progress mask: re-render from the
        # working mask rather than the saved record (which would drop the edit).
        if self._editing:
            self.preview.set_alpha(alpha_from_slider(value))
        else:
            self._update_preview()

    def _update_preview(self) -> None:
        record = self._current_record()
        if record is None:
            self.preview.show_message("Select an image")
            self._edit_button.setEnabled(False)
            self.mask_slider.setEnabled(False)  # the blender does nothing with no image showing
            return
        self._edit_button.setEnabled(True)
        self.mask_slider.setEnabled(True)
        self.preview.display(record, self.engine.config, alpha_from_slider(self.mask_slider.value()))

    def _open_large(self, index: QModelIndex) -> None:
        if not index.isValid() or not (0 <= index.row() < len(self.records)):
            return
        OverlayDialog(self.records[index.row()], self.engine.config, self).exec()

    # --- mask editing ----------------------------------------------------

    def _enter_edit(self) -> None:
        """Start editing the selected record's mask (works for any selected image)."""
        record = self._current_record()
        if record is None:
            return
        image = load_image(record.path)
        if record.mask is not None:
            working = record.mask.copy()
        else:
            working = np.zeros((int(image.shape[0]), int(image.shape[1])), dtype=np.uint8)

        self._editing = True
        self._edit_record = record
        self._original_mask = working.copy()
        self._history = EditHistory(working)
        alpha = alpha_from_slider(self.mask_slider.value())
        self.preview.begin_edit(image, working, alpha, float(self._brush_slider.value()))
        self._set_ui_frozen(True)
        self._controls.setCurrentIndex(1)
        self._refresh_undo_redo()

    def _set_ui_frozen(self, frozen: bool) -> None:
        """Grey out (or restore) everything but the preview, blender, and edit toolbar."""
        self.table.setEnabled(not frozen)
        self.toolbar.setEnabled(not frozen)
        self.top3_window.setEnabled(not frozen)
        # Disable the Save action itself (not just its toolbar button) so the Cmd+S shortcut is
        # inert during an edit; editing is only reachable once a dataset is loaded, so re-enabling
        # on unfreeze is always correct.
        self._save_action.setEnabled(not frozen)

    def _on_edit_started(self) -> None:
        """First press of a stroke: raise the blender to at least 0.5 so the mask is visible."""
        floor = slider_from_alpha(0.5)
        self.mask_slider.setValue(raise_slider_to_floor(self.mask_slider.value(), floor))

    def _on_stroke_committed(self) -> None:
        """A stroke that changed the mask ended: record one undo-history entry."""
        if self._history is None:
            return
        working = self.preview.working_mask()
        if working is not None:
            self._history.push(working)
            self._refresh_undo_redo()

    def _on_brush_changed(self, value: int) -> None:
        self.preview.set_brush_radius(float(value))

    def _undo_edit(self) -> None:
        if self._history is None:
            return
        self.preview.set_working_mask(self._history.undo())
        self._refresh_undo_redo()

    def _redo_edit(self) -> None:
        if self._history is None:
            return
        self.preview.set_working_mask(self._history.redo())
        self._refresh_undo_redo()

    def _refresh_undo_redo(self) -> None:
        self._undo_button.setEnabled(self._history is not None and self._history.can_undo())
        self._redo_button.setEnabled(self._history is not None and self._history.can_redo())

    def _keep_edit(self) -> bool:
        """Commit the post-processed working mask in memory: cells below the minimum area are removed
        and metrics computed first (so a failure leaves the record and the working mask intact).

        Returns whether the commit succeeded, so the close guard can abort the exit on a failure.
        """
        record = self._edit_record
        working = self.preview.working_mask()
        if record is None or working is None:
            return False
        try:
            clean, metrics = self.engine.postprocess(working)
        except Exception:
            log.exception("Morphometry failed for %s", record.name)
            self.statusBar().showMessage(f"Morphometry failed for {record.name}; edit left open.")
            return False  # stay in edit mode; the doctor's mask is preserved
        record.metrics = metrics
        record.mask = clean
        record.is_nerve = True
        self._mark_dirty()
        self._exit_edit()
        self._resort_and_refresh(lambda recs: recs.index(record))
        return True

    def _discard_edit(self) -> None:
        """Discard the working mask, confirming first if it differs from the edit-start mask."""
        working = self.preview.working_mask()
        if working is not None and self._original_mask is not None and mask_changed(working, self._original_mask):
            # Custom button text: a standard Discard button is relabelled "Don't Save" on macOS,
            # so the dialog reads exactly "Cancel" / "Discard".
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Discard edits?")
            box.setText("Discard your changes to this mask?")
            discard_button = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
            cancel_button = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(cancel_button)
            box.exec()
            if box.clickedButton() is not discard_button:
                return
        self._exit_edit()
        self._update_preview()  # re-render from the unchanged record

    def _exit_edit(self) -> None:
        self._editing = False
        self.preview.end_edit()
        self._set_ui_frozen(False)
        self._controls.setCurrentIndex(0)
        self._edit_record = None
        self._original_mask = None
        self._history = None
