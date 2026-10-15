# tests/unit_match/conftest.py
from pathlib import Path
import pytest
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every test in tests/unit_match reads settings from a fresh folder under the pytest base temp (day-table-keys Task 6) | date: 2026-10-14]
from .settings_fixtures import tmp_settings  # noqa: F401
# [AGENT EDIT END]

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def fixtures_dir():
    return FIXTURES
