# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tmp_settings points every test's user-scope settings at a fresh folder under the pytest base temp (day-table-keys Task 6) | date: 2026-10-14]
"""``tmp_settings``: every viewer window reads the user-scope settings file (``theme.settings()``: the theme, the
unit palette and, through ``keys.bindings``, the ``shortcuts/<action id>`` keys), so a test without this fixture
read ``~/.config/multiphy/multiphy.ini`` or the last test's folder. The fixture points the user-scope INI settings
at a fresh folder from ``tmp_path_factory`` (so a test's ``tmp_path`` holds only what the test writes) and, at
teardown, at that folder's ``after``. It is autouse wherever it is imported by name; pytest runs autouse fixtures
before requested ones, so ``theme_env`` points the path at ``tmp_path`` on top in the tests that take it.

Only ``pytest`` and ``PySide6.QtCore`` are imported: ``theme``, ``display`` and the other viewer modules load
``multiphy/viewer/qt.py``, which sets the matplotlib backend."""
import pytest
from PySide6 import QtCore


@pytest.fixture(autouse=True)
def tmp_settings(tmp_path_factory):
    """Point the user-scope INI settings at a fresh ``settings<N>`` folder under the pytest base temp; yield it.
    Teardown points them at ``<folder>/after``."""
    folder = tmp_path_factory.mktemp('settings')
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(folder))
    yield folder
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(folder / 'after'))
# [AGENT EDIT END]
