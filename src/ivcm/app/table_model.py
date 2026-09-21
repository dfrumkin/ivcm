"""Qt table model over the record list (a thin read-adapter; the main window owns the list)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, QTimer

from ivcm.app.records import MAX_PINNED, ImageRecord, pinned_count

# Column layout: two checkboxes ("Pin", "Use"), name, then the four metrics.
_COLUMNS = ("Pin", "Use", "Name", "Count", "Density", "Area", "Perimeter")
_METRIC_KEYS = {3: "count", 4: "density", 5: "area", 6: "perimeter"}
_PIN_COLUMN = 0
_USE_COLUMN = 1
_NAME_COLUMN = 2
_COUNT_COLUMN = 3

_Index = QModelIndex | QPersistentModelIndex
_NO_PARENT = QModelIndex()  # module-level singleton: a function-call default would trip ruff B008


class ResultsTableModel(QAbstractTableModel):
    """Read-adapter over a shared ``list[ImageRecord]``.

    The list is owned (and re-sorted in place) by the main window; the model never mutates it
    except to flip ``is_nerve``/``pinned`` on a checkbox edit. ``setData`` only flips the flags and
    schedules the heavy segment-and-refresh work to run *after* it returns (via
    ``QTimer.singleShot(0)``), so the model is never reset re-entrantly from inside an edit.

    ``on_pin(record, now_checked)`` runs the pin-toggle's consequences; ``on_pin_rejected`` shows
    the "at most three pins" pop-up when a fourth pin is attempted (also deferred, so no modal
    dialog opens on the edit path). ``on_dirty`` is invoked **synchronously** the instant a checkbox
    edit flips a record field -- before the deferred ``on_toggle``/``on_pin`` handler runs -- so the
    main window's unsaved-changes flag reflects the change even if the app is closed before the
    timer callback fires.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        on_toggle: Callable[[ImageRecord], None],
        on_pin: Callable[[ImageRecord, bool], None],
        on_pin_rejected: Callable[[], None],
        on_dirty: Callable[[], None],
    ) -> None:
        super().__init__()
        self._records = records
        self._on_toggle = on_toggle
        self._on_pin = on_pin
        self._on_pin_rejected = on_pin_rejected
        self._on_dirty = on_dirty

    def refresh(self) -> None:
        """Re-read the (possibly re-sorted) record list into the view."""
        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent: _Index = _NO_PARENT) -> int:
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent: _Index = _NO_PARENT) -> int:
        return 0 if parent.isValid() else len(_COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return _COLUMNS[section]
        return None

    def data(self, index: _Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        record = self._records[index.row()]
        column = index.column()
        if column == _PIN_COLUMN:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if record.pinned else Qt.CheckState.Unchecked
            return None
        if column == _USE_COLUMN:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if record.is_nerve else Qt.CheckState.Unchecked
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            if column == _NAME_COLUMN:
                return record.name
            if record.is_nerve and record.metrics is not None:
                value = record.metrics[_METRIC_KEYS[column]]
                return f"{value:.0f}" if column == _COUNT_COLUMN else f"{value:.2f}"
            return ""
        return None

    def setData(self, index: _Index, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.CheckStateRole:
            return False
        checked = int(value) == int(Qt.CheckState.Checked.value)
        if index.column() == _PIN_COLUMN:
            return self._set_pin(index.row(), checked)
        if index.column() == _USE_COLUMN:
            return self._set_use(index.row(), checked)
        return False

    def _set_pin(self, row: int, checked: bool) -> bool:
        record = self._records[row]
        if checked and not record.pinned and pinned_count(self._records) >= MAX_PINNED:
            QTimer.singleShot(0, self._on_pin_rejected)  # deferred so no modal opens on the edit path
            return False
        record.pinned = checked
        if checked:
            record.is_nerve = True  # pinning forces "use"
        self._on_dirty()  # synchronous: mark unsaved now, before the deferred handler
        self._emit_row_changed(row)  # pin, use, and the metric columns can all change
        QTimer.singleShot(0, lambda: self._on_pin(record, checked))
        return True

    def _set_use(self, row: int, checked: bool) -> bool:
        record = self._records[row]
        record.is_nerve = checked
        if not checked:
            record.pinned = False  # unchecking Use always unpins
        self._on_dirty()  # synchronous: mark unsaved now, before the deferred handler
        self._emit_row_changed(row)
        QTimer.singleShot(0, lambda: self._on_toggle(record))
        return True

    def _emit_row_changed(self, row: int) -> None:
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(_COLUMNS) - 1))

    def flags(self, index: _Index) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() in (_PIN_COLUMN, _USE_COLUMN):
            base |= Qt.ItemFlag.ItemIsUserCheckable
        return base
