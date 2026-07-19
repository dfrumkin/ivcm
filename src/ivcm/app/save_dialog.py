"""A single-window "save into a directory" dialog (imports Qt).

``QFileDialog``'s save mode reinterprets a typed name that matches an existing folder as "open it"
(the accept button turns into **Open** and navigates in), so it can never return an existing
directory as a save target. This small dialog keeps one always-**Save** button instead: double-click
a folder to navigate into it, single-click to put its name in the field, edit the name freely, then
**Save** returns ``<current directory>/<name>`` -- existing or not. The suggested name is pre-filled
and, when it already exists in the starting directory, pre-selected.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from PySide6.QtCore import QDir, QModelIndex
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SaveDirectoryDialog(QDialog):
    """Choose a parent directory (by navigating) and a folder name (by typing), in one window."""

    def __init__(self, parent: QWidget | None, start_dir: Path, suggested: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save labels")
        self._current = Path(start_dir)
        self._name_before_click = suggested  # restored when a click becomes a navigate (double-click)

        self._model = QFileSystemModel(self)
        self._model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot)
        self._model.setRootPath(str(self._current))
        self._model.directoryLoaded.connect(self._on_directory_loaded)

        self._view = QListView()
        self._view.setModel(self._model)
        self._view.setRootIndex(self._model.index(str(self._current)))
        self._view.clicked.connect(self._on_clicked)
        self._view.doubleClicked.connect(self._on_double_clicked)

        self._path_label = QLabel(str(self._current))
        up_button = QPushButton("Up")
        up_button.clicked.connect(self._go_up)

        self._name = QLineEdit(suggested)
        self._name.textChanged.connect(self._sync_save_enabled)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        header = QHBoxLayout()
        header.addWidget(self._path_label, 1)
        header.addWidget(up_button)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        name_row.addWidget(self._name, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self._view, 1)
        layout.addLayout(name_row)
        layout.addWidget(self._buttons)
        self.resize(520, 420)

        self._select_matching_name()
        self._sync_save_enabled()

    def selected_path(self) -> Path | None:
        """The chosen target: ``<current directory>/<name>``, or ``None`` when the name is empty."""
        name = self._name.text().strip()
        return self._current / name if name else None

    def _set_current(self, directory: Path) -> None:
        self._current = directory
        self._model.setRootPath(str(directory))
        self._view.setRootIndex(self._model.index(str(directory)))
        self._path_label.setText(str(directory))
        self._select_matching_name()

    def _go_up(self) -> None:
        parent = self._current.parent
        if parent != self._current:
            self._set_current(parent)

    def _on_clicked(self, index: QModelIndex) -> None:
        # Single click selects a folder: put its name in the field so it becomes the save target.
        self._name_before_click = self._name.text()
        self._name.setText(self._model.fileName(index))

    def _on_double_clicked(self, index: QModelIndex) -> None:
        # A single click fired first and filled the name; a double-click means "go in", not "target
        # this folder", so restore the pre-click name and navigate into the folder.
        self._name.setText(self._name_before_click)
        self._set_current(Path(self._model.filePath(index)))

    def _on_directory_loaded(self, path: str) -> None:
        # QFileSystemModel loads directory contents asynchronously; re-apply the selection once the
        # current directory's children are available.
        if Path(path) == self._current:
            self._select_matching_name()

    def _select_matching_name(self) -> None:
        name = self._name.text().strip()
        target = self._current / name if name else None
        if target is not None and target.is_dir():
            self._view.setCurrentIndex(self._model.index(str(target)))
        else:
            self._view.clearSelection()

    def _sync_save_enabled(self) -> None:
        # button() is typed non-optional in the stubs but can be None; guard defensively.
        button = cast("QPushButton | None", self._buttons.button(QDialogButtonBox.StandardButton.Save))
        if button is not None:
            button.setEnabled(bool(self._name.text().strip()))
