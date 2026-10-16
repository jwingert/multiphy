# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the logo: committed SVGs equal the generator, the trace, the icon sizes and pixmaps, and both entry points setting the window icon (logo) | date: 2026-10-13]
"""The logo: the committed SVGs equal what ``make_logo`` builds, the trace is the unit's waveform scaled, the
window icon carries every size, and the launcher and the viewer both set it. Settings and theme assets land under
``tmp_path`` (``theme_env``)."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from multiphy import __version__
from multiphy.launch import window as window_module
from multiphy.launch.plugins import load_plugins
from multiphy.viewer import app as viewer_app
from multiphy.viewer import logo
from multiphy.viewer.logo import make_logo

from .launch_window_fixtures import qt, runs  # noqa: F401 (fixtures by name)
from .theme_fixtures import theme_env  # noqa: F401  (a fixture, used by name)

SVGS = ['icon_16.svg', 'icon_24.svg', 'icon_32.svg', 'icon_48.svg', 'icon_64.svg', 'logo.svg']


def test_committed_svgs_match_the_generator():
    built = make_logo.build()
    assert sorted(built) == SVGS
    for name, text in built.items():
        assert (make_logo.HERE / name).read_text() == text, name
    assert sorted(path.name for path in make_logo.HERE.glob('*.svg')) == SVGS


def test_strokes_grow_heavier_as_icons_shrink():
    assert make_logo.WEIGHTS == {64: 1.3, 48: 1.45, 32: 1.7, 24: 2.0, 16: 2.3}
    built = make_logo.build()
    assert built['logo.svg'].count('stroke="#F472B6" stroke-width="11.00"') == 3
    assert built['icon_16.svg'].count('stroke="#F472B6" stroke-width="25.30"') == 3
    assert built['icon_16.svg'].count('<circle') == 3 and 'r="29.90"' in built['icon_16.svg']
    assert [f'stroke-opacity="{o}"' in built['logo.svg'] for o in (1.0, 0.6, 0.32)] == [True] * 3
    assert '-0.0,' not in built['logo.svg'] and ',-0.0 ' not in built['logo.svg']


def test_trace_is_the_unit_scaled():
    x, y = make_logo.waveform()
    assert len(make_logo.SAMPLES) == len(x) == len(y) == 53
    assert (x[0], x[-1]) == (-1.0, 1.0)
    assert y[0] == 0.0
    assert np.ptp(y) == pytest.approx(1.0)
    assert np.argmin(y) < np.argmax(y)  # the trough comes before the repolarisation peak
    assert np.argmin(make_logo.SAMPLES) == 12  # cropped 12 samples before the trough


def test_svg_path_per_size():
    assert [logo.svg_path(px).name for px in logo.ICON_SIZES] == [
        'icon_16.svg', 'icon_24.svg', 'icon_32.svg', 'icon_48.svg', 'icon_64.svg', 'logo.svg', 'logo.svg']


def test_icon_has_every_size(qt):
    sizes = logo.icon().availableSizes()
    assert [(s.width(), s.height()) for s in sizes] == [(px, px) for px in logo.ICON_SIZES]


def test_pixmap_draws_the_tile(qt):
    image = logo.pixmap(64).toImage()
    assert (image.width(), image.height()) == (64, 64)
    assert image.pixelColor(0, 0).alpha() == 0  # a rounded corner
    assert image.pixelColor(2, 32).name() == '#0f1114' and image.pixelColor(2, 32).alpha() == 255
    pink = QtGui.QColor('#F472B6')
    hits = sum(image.pixelColor(i, j) == pink for i in range(64) for j in range(64))
    assert hits > 20
    retina = logo.pixmap(64, 2.0)
    assert (retina.width(), retina.height(), retina.devicePixelRatio()) == (128, 128, 2.0)


def test_version():
    assert __version__ == '0.0.0'


def test_docs_name_the_logo():
    package = make_logo.HERE.parents[1]  # logo -> viewer -> multiphy
    viewer = (package / 'docs' / 'viewer.md').read_text(encoding='utf-8')
    for word in ('| `logo/` |', 'make_logo.py', '`About multiphy`', '`0.0.0`', 'PRN067a-338-1'):
        assert word in viewer, word
    assert '`About multiphy`' in (package / 'docs' / 'project_folder.md').read_text(encoding='utf-8')
    assert '<img src="viewer/logo/logo.svg"' in (package / 'README.md').read_text(encoding='utf-8')


def _keep_icon(app):
    return QtGui.QIcon(app.windowIcon())


def test_run_window_sets_the_icon(qt, runs, theme_env, monkeypatch):
    app = QtWidgets.QApplication.instance()
    kept = _keep_icon(app)
    app.setWindowIcon(QtGui.QIcon())
    opened = []
    monkeypatch.setattr(window_module, 'LauncherWindow', lambda *a, **k: opened.append(QtWidgets.QWidget()) or opened[-1])
    QtCore.QTimer.singleShot(0, QtWidgets.QApplication.quit)
    try:
        assert window_module.run_window(runs['root'], None, load_plugins(runs['root'])) == 0
        assert [s.width() for s in app.windowIcon().availableSizes()] == list(logo.ICON_SIZES)
    finally:
        app.setWindowIcon(kept)
        for w in opened:
            w.close()


def test_viewer_main_sets_the_icon(qt, theme_env, monkeypatch):
    app = QtWidgets.QApplication.instance()
    kept = _keep_icon(app)
    app.setWindowIcon(QtGui.QIcon())
    opened = []
    monkeypatch.setattr(viewer_app, 'open_run', lambda path: opened.append(QtWidgets.QWidget()) or opened[-1])
    QtCore.QTimer.singleShot(0, QtWidgets.QApplication.quit)
    try:
        assert viewer_app.main(['--run', str(theme_env / 'run')]) == 0
        assert [s.width() for s in app.windowIcon().availableSizes()] == list(logo.ICON_SIZES)
    finally:
        app.setWindowIcon(kept)
        for w in opened:
            w.close()
# [AGENT EDIT END]
