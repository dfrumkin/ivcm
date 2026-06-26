"""Entry point for the ``ivcm-ui`` console script.

Creates the ``QApplication`` first, then builds the ``InferenceEngine`` inside a try/except so
checkpoint/config/path errors surface as a dialog (and exit) rather than escaping to the console
before the GUI exists.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from ivcm.app.config import CONFIG
from ivcm.app.inference import InferenceEngine
from ivcm.app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
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
