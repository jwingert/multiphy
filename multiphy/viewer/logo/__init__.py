# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the logo package: the window icon and the About picture, rendered from the committed SVGs (logo) | date: 2026-10-13]
"""The multiphy logo: the window icon and the About picture, rendered from the SVGs that ``make_logo`` writes
next to this file. Qt comes only from ``..qt``."""
from pathlib import Path

from multiphy.viewer.logo.make_logo import HERE, LOGO, WEIGHTS
from multiphy.viewer.qt import QtCore, QtGui, QtSvg

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def svg_path(px: int) -> Path:
    """The SVG drawn at ``px``: ``icon_<px>.svg`` where ``WEIGHTS`` gives that size heavier strokes, else
    ``logo.svg``."""
    return HERE / (f'icon_{px}.svg' if px in WEIGHTS else LOGO)


def pixmap(px: int, ratio: float = 1.0) -> QtGui.QPixmap:
    """``svg_path(px)`` rendered to a ``px`` × ``px`` pixmap of ``px * ratio`` device pixels; the tile's
    rounded corners stay transparent."""
    side = max(1, round(px * ratio))
    image = QtGui.QImage(side, side, QtGui.QImage.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    QtSvg.QSvgRenderer(str(svg_path(px))).render(painter, QtCore.QRectF(0, 0, side, side))
    painter.end()
    result = QtGui.QPixmap.fromImage(image)
    result.setDevicePixelRatio(ratio)
    return result


def icon() -> QtGui.QIcon:
    """The window icon: one pixmap per ``ICON_SIZES`` entry."""
    result = QtGui.QIcon()
    for px in ICON_SIZES:
        result.addPixmap(pixmap(px))
    return result
# [AGENT EDIT END]
