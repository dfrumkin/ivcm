"""Desktop clinical-review UI for the ivcm models.

Import-light on purpose: this package's ``__init__`` pulls in **no** Qt and **no** model code,
so the Qt-free modules (:mod:`ivcm.app.config`, :mod:`ivcm.app.records`,
:mod:`ivcm.app.render`, :mod:`ivcm.app.inference`) can be imported in the base environment
(without the ``ui`` extra installed) for unit testing. The Qt view modules and the ``ivcm-ui``
entry point import PySide6 themselves.
"""

from __future__ import annotations
