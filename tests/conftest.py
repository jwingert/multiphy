from pathlib import Path
import sys
import pytest
from .lab_blocker import LabBlocker
sys.meta_path.insert(0, LabBlocker())
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every test in tests reads settings from a fresh folder under the pytest base temp (day-table-keys Task 6) | date: 2026-10-14]
from .settings_fixtures import tmp_settings  # noqa: F401
# [AGENT EDIT END]

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def fixtures_dir():
    return FIXTURES
