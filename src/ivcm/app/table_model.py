"""Qt table model over the record list (a thin read-adapter; the main window owns the list)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, QTimer

from ivcm.app.records import ImageRecord

# Column layout: checkbox, name, then the three metrics.
_COLUMNS = ("", "Name", "Density", "Area", "Perimeter")
_METRIC_KEYS = {2: "density", 3: "area", 4: "perimeter"}

_Index = QModelIndex | QPersistentModelIndex
_NO_PARENT = QModelIndex()  # module-level singleton: a function-call default would trip ruff B008


class ResultsTableModel(QAbstractTableModel):
    """Read-adapter over a shared ``list[ImageRecord]``.

    The list is owned (and re-sorted in place) by the main window; the model never mutates it
    except to flip ``is_nerve`` on a checkbox edit. ``setData`` only flips the flag and schedules
    the heavy segment-and-refresh work to run *after* it returns (via ``QTimer.singleShot(0)``),
    so the model is never reset re-entrantly from inside an edit.
    """

    def __init__(self, records: list[ImageRecord], on_toggle: Callable[[ImageRecord], None]) -> None:
        super().__init__()
        self._records = records
        self._on_toggle = on_toggle

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
        if column == 0:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if record.is_nerve else Qt.CheckState.Unchecked
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            if column == 1:
                return record.name
            if record.is_nerve and record.metrics is not None:
                return f"{record.metrics[_METRIC_KEYS[column]]:.2f}"
            return ""
        return None

    def setData(self, index: _Index, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            record = self._records[index.row()]
            record.is_nerve = int(value) == int(Qt.CheckState.Checked.value)
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            QTimer.singleShot(0, lambda: self._on_toggle(record))
            return True
        return False

    def flags(self, index: _Index) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            base |= Qt.ItemFlag.ItemIsUserCheckable
        return base
