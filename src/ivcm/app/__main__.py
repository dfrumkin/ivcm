"""Entry point for the ``ivcm-ui`` console script.

Creates the ``QApplication`` first, then builds the ``InferenceEngine`` inside a try/except so
checkpoint/config/path errors surface as a dialog (and exit) rather than escaping to the console
before the GUI exists.
"""

from __future__ import annotations

import sys
from importlib import resources

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox

from ivcm.app.config import CONFIG
from ivcm.app.inference import InferenceEngine
from ivcm.app.main_window import MainWindow


def _app_icon() -> QIcon:
    """Load the packaged app icon from bytes.

    Reading the PNG as bytes (rather than via a filesystem path) avoids any ``as_file`` temp-path
    lifetime concern and works identically from a checkout or an installed wheel.
    """
    data = resources.files("ivcm.app").joinpath("assets", "icons", "icon.png").read_bytes()
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return QIcon(pixmap)


def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(_app_icon())  # propagates to every top-level window and dialog
    try:
        engine = InferenceEngine(CONFIG)
    except Exception as exc:  # surface any startup/model-load failure as a dialog, then exit
        QMessageBox.critical(None, "ivcm-ui — startup error", str(exc))
        return 1

    window = MainWindow(engine)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
