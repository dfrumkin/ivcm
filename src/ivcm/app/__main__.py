"""Entry point for the ``ivcm-ui`` console script.

Creates the ``QApplication`` first, then builds the ``InferenceEngine`` inside a try/except so
checkpoint/config/path errors surface as a dialog (and exit) rather than escaping to the console
before the GUI exists.

The hidden ``--self-check`` flag drives the whole startup-and-first-use path headlessly (used by the
Windows CI smoke gate): it constructs the app exactly as normal startup does, runs one synthetic
image through the engine, and exits ``0`` before showing a window or entering the event loop. On any
failure it writes a traceback to a fixed log file and to stderr and exits non-zero, never opening a
blocking dialog (which would hang headless CI).
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox

from ivcm.app.config import CONFIG
from ivcm.app.inference import InferenceEngine, load_image
from ivcm.app.main_window import MainWindow

_SELF_CHECK_FLAG = "--self-check"
_SELF_CHECK_LOG = Path(tempfile.gettempdir()) / "ivcm-selfcheck.log"
# Segmentation encoder downsampling stride: the synthetic self-check image side must be a multiple
# of this so native-resolution sliding-window inference runs on a valid shape.
_ENCODER_STRIDE = 32


def _ensure_std_streams() -> None:
    """Give libraries a real sink when ``sys.stdout``/``sys.stderr`` are ``None``.

    A PyInstaller windowed (``--noconsole``) build has no console, so both streams are ``None``;
    any library that writes to them (a download progress bar, a warning) then raises
    ``AttributeError: 'NoneType' object has no attribute 'write'``. Point the missing streams at the
    null device so a stray write never crashes the app. Streams that already exist (e.g. redirected
    to a file by the CI smoke launcher) are left untouched.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 (open for process lifetime)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 (open for process lifetime)


def _app_icon() -> QIcon:
    """Load the packaged app icon from bytes.

    Reading the PNG as bytes (rather than via a filesystem path) avoids any ``as_file`` temp-path
    lifetime concern and works identically from a checkout or an installed wheel.
    """
    data = resources.files("ivcm.app").joinpath("assets", "icons", "icon.png").read_bytes()
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return QIcon(pixmap)


def _self_check_inference(engine: InferenceEngine) -> None:
    """Round-trip one synthetic image through ``load_image`` -> classify -> analyze.

    The image is written to a temp file and read back via the app's own ``load_image`` so PIL file
    decode (and its bundled image plugins) are exercised too, not just in-memory arrays. Its size is
    derived from the loaded engine (at least ``max(cls_size, seg_roi)``, rounded up to the encoder
    stride) so segmentation runs on a valid native-resolution shape.
    """
    side = max(engine.cls_size, engine.seg_roi)
    side = ((side + _ENCODER_STRIDE - 1) // _ENCODER_STRIDE) * _ENCODER_STRIDE
    array = np.zeros((side, side), dtype=np.uint8)
    array[side // 4 : 3 * side // 4, side // 4 : 3 * side // 4] = 255  # a simple bright square

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "self-check.tif"
        Image.fromarray(array).save(path)
        image = load_image(path)
        engine.classify(image)
        engine.analyze(image)


def _run_self_check() -> int:
    """Headless smoke check of the full startup + first-use path."""
    try:
        app = QApplication(sys.argv)
        app.setWindowIcon(_app_icon())
        engine = InferenceEngine(CONFIG)
        window = MainWindow(engine)  # construct the whole UI; never show() / exec()
        assert window is not None
        _self_check_inference(engine)
    except Exception:
        report = traceback.format_exc()
        try:
            _SELF_CHECK_LOG.write_text(report, encoding="utf-8")
        except OSError:
            pass
        print(report, file=sys.stderr)  # also to stderr for the workflow's redirected capture
        return 1
    print("ivcm-ui self-check: OK")
    return 0


def main() -> int:
    _ensure_std_streams()  # windowed builds have None stdout/stderr; make writes safe first
    if _SELF_CHECK_FLAG in sys.argv:
        # Strip the flag before Qt sees sys.argv (Qt consumes argv and a stray flag is brittle).
        sys.argv = [arg for arg in sys.argv if arg != _SELF_CHECK_FLAG]
        return _run_self_check()

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
