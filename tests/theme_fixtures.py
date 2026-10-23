"""``theme_env``: a theme test's settings file and tick and chevron SVGs land under ``tmp_path``; at teardown the
application's style sheet, palette, active theme, matplotlib settings and font, and the default unit palette, are
put back. Import it by name."""
import matplotlib
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from multiphy.viewer import display, theme
from multiphy.viewer import unit_palettes


@pytest.fixture
def theme_env(tmp_path, monkeypatch):
    """Point the user-scope INI settings and ``TMPDIR`` at ``tmp_path``; yield ``tmp_path``. Teardown ends by
    pointing the settings at ``tmp_path / 'after'``, so no later test reads this test's shortcuts."""
    saved = matplotlib.rcParams.copy()
    saved_font = QtGui.QFont(QtWidgets.QApplication.instance().font())
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    temporary_dir = QtCore.QTemporaryDir
    monkeypatch.setattr(QtCore, 'QTemporaryDir', lambda *args: temporary_dir(str(tmp_path / 'theme-XXXXXX')))
    monkeypatch.setattr(theme, '_assets', None)
    yield tmp_path
    app = QtWidgets.QApplication.instance()
    app.setStyleSheet('')
    app.setPalette(app.style().standardPalette())
    display.set_theme('graphite')
    matplotlib.rcParams.update(saved)
    app.setFont(saved_font)
    unit_palettes.set_palette(unit_palettes.DEFAULT)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path / 'after'))
