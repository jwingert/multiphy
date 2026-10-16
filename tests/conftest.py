from pathlib import Path
import sys
import pytest
from .lab_blocker import LabBlocker
sys.meta_path.insert(0, LabBlocker())
from .settings_fixtures import tmp_settings  # noqa: F401

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def fixtures_dir():
    return FIXTURES
