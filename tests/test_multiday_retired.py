# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: multiday archives and their runner are retired; the run dialog forwards its cache root (Plan B Task 8) | date: 2026-09-28]
import importlib.util
from pathlib import Path


def test_multiday_modules_gone():
    assert importlib.util.find_spec('multiphy.multiday') is None
    assert importlib.util.find_spec('multiphy.multiday_runner') is None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Run... dialog module is deleted (run-curation Task 4) | date: 2026-10-05]
def test_backend_job_module_gone():
    assert importlib.util.find_spec('multiphy.backend_job') is None
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every Task 8 edit is a START/END block, never a standalone marker (Plan B Task 8, fix round 1) | date: 2026-09-28]
def test_task_8_edit_tags_are_start_end_blocks():
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / 'multiphy').rglob('*.py')) + sorted((root / 'tests').rglob('*.py')):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith('# [AGENT EDIT') and '(Plan B Task 8' in line:
                assert '[AGENT EDIT START |' in line, f'{path}:{i + 1}'
                assert any('[AGENT EDIT END]' in later for later in lines[i + 1:]), f'{path}:{i + 1}'
# [AGENT EDIT END]
