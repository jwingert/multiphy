from pathlib import Path
import os
import sys
import pytest
from .lab_blocker import LabBlocker
sys.meta_path.insert(0, LabBlocker())
# -B covers this interpreter only; every child Python a test starts inherits this and writes no bytecode.
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from .settings_fixtures import tmp_settings  # noqa: F401

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture(autouse=True)
def no_env_root(monkeypatch):
    """No test inherits ``MULTIPHY_CACHE_ROOT``; a test that needs a root passes one or sets the variable itself."""
    monkeypatch.delenv('MULTIPHY_CACHE_ROOT', raising=False)


def pytest_sessionfinish(session, exitstatus):
    """Destroy leftover test widgets while Qt and Python callbacks are still alive.

    Offscreen tests close windows without running the application's final event loop. Closed panels
    can remain owned by Qt; leaving their destruction to interpreter shutdown crashes PySide on
    Windows. This is test-session cleanup, without changing the application's close/reopen behavior.
    """
    widgets = sys.modules.get('PySide6.QtWidgets')
    core = sys.modules.get('PySide6.QtCore')
    if widgets is None or core is None:
        return
    app = widgets.QApplication.instance()
    if not isinstance(app, widgets.QApplication):
        return
    for widget in app.topLevelWidgets():
        widget.deleteLater()
    core.QCoreApplication.sendPostedEvents(None, core.QEvent.DeferredDelete)
