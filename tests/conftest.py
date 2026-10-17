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
