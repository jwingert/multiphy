# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Qt binding pin, from gui.py:70-94 (viewer-split Task 4) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: QtSvg renders the logo; the tag wraps the whole docstring (logo) | date: 2026-10-13]
"""The one Qt binding of the viewer: PySide6, pinned before matplotlib's Qt backend is imported.

Every viewer module that needs Qt imports it from here: ``QtCore``, ``QtWidgets``, ``QtGui``, ``QtSvg``,
``FigureCanvasQTAgg``, ``NavigationToolbar2QT`` and ``Figure``.
"""
# [AGENT EDIT END]
import os
import warnings

# Pin the Qt binding BEFORE matplotlib's Qt backend is imported. Several lab
# environments carry both PySide6 and PyQt5 -- psi_nems2 does, because enaml
# requires PyQt5 -- and matplotlib will happily choose PyQt5. Its canvas is then
# a PyQt5 QWidget while every other widget here is PySide6, and the layout
# rejects it with "addWidget called with wrong argument types".
#
# An existing QT_API is respected rather than overwritten, so importing this
# module cannot change the binding under code that already chose one. If that
# choice is incompatible, say so plainly instead of failing later inside Qt.
_WANT_QT = 'PySide6'
_have = os.environ.get('QT_API')
if _have and _have.lower() not in ('pyside6',):
    warnings.warn(
        f'QT_API is set to {_have!r}, but multiphy.viewer requires PySide6. '
        'Matplotlib canvases will not be accepted by the PySide6 layouts. '
        'Unset QT_API or set it to PySide6.', RuntimeWarning)
else:
    os.environ['QT_API'] = _WANT_QT
from PySide6 import QtCore, QtWidgets, QtGui  # noqa: E402
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: QtSvg renders the logo (logo) | date: 2026-10-13]
from PySide6 import QtSvg  # noqa: E402
# [AGENT EDIT END]

import matplotlib  # noqa: E402
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: QtSvg renders the logo (logo) | date: 2026-10-13]
__all__ = ['QtCore', 'QtWidgets', 'QtGui', 'QtSvg', 'FigureCanvasQTAgg', 'NavigationToolbar2QT', 'Figure']
# [AGENT EDIT END]
# [AGENT EDIT END]
