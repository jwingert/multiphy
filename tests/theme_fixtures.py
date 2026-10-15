# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the theme_env fixture keeps a theme test's settings file and SVG assets under tmp_path and restores the app's look (launcher-gui Task 8) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the application font among what teardown puts back (graphite-palettes Task 3) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the unit palette among what teardown puts back (graphite-palettes Task 4) | date: 2026-10-11]
"""``theme_env``: a theme test's settings file and tick and chevron SVGs land under ``tmp_path``; at teardown the
application's style sheet, palette, active theme, matplotlib settings and font, and the default unit palette, are
put back. Import it by name."""
# [AGENT EDIT END]
# [AGENT EDIT END]
import matplotlib
import pytest
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fixture saves and restores the application font (graphite-palettes Task 3) | date: 2026-10-11]
from PySide6 import QtCore, QtGui, QtWidgets
# [AGENT EDIT END]

from nems_lbhb.unit_match.viewer import display, theme
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: teardown restores the default unit palette (graphite-palettes Task 4) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer import unit_palettes
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: teardown points the settings at an empty folder, so no later test reads this test's shortcuts; tag wraps the whole fixture (day-table-keys Task 4) | date: 2026-10-14]
@pytest.fixture
def theme_env(tmp_path, monkeypatch):
    """Point the user-scope INI settings and ``TMPDIR`` at ``tmp_path``; yield ``tmp_path``. Teardown ends by
    pointing the settings at ``tmp_path / 'after'``, so no later test reads this test's shortcuts."""
    saved = matplotlib.rcParams.copy()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: apply_theme sets the application font; teardown puts the saved one back (graphite-palettes Task 3) | date: 2026-10-11]
    saved_font = QtGui.QFont(QtWidgets.QApplication.instance().font())
    # [AGENT EDIT END]
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path))
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setattr(theme, '_assets', None)
    yield tmp_path
    app = QtWidgets.QApplication.instance()
    app.setStyleSheet('')
    app.setPalette(app.style().standardPalette())
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: teardown restores the default theme, graphite (graphite-palettes Task 2) | date: 2026-10-11]
    display.set_theme('graphite')
    # [AGENT EDIT END]
    matplotlib.rcParams.update(saved)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: apply_theme sets the application font; teardown puts the saved one back (graphite-palettes Task 3) | date: 2026-10-11]
    app.setFont(saved_font)
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a Unit colours switch sets the active palette; teardown puts the default back (graphite-palettes Task 4) | date: 2026-10-11]
    unit_palettes.set_palette(unit_palettes.DEFAULT)
    # [AGENT EDIT END]
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(tmp_path / 'after'))
# [AGENT EDIT END]
# [AGENT EDIT END]
