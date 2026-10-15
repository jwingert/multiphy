# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the project-folder docs stay true: the adding-a-source example runs, and the project-folder guide names every command and rule (multiphy project folder Task 16; fix round 2: the file holds the brief's two tests, and the guide's code claims are pinned in the launcher and extraction suites) | date: 2026-09-25]
"""The user docs for the project folder and new sources stay true.

``adding_a_source.md`` holds one ``python`` fence defining ``main(root)``; it is executed as
written and must extract its day in the raw tier. ``project_folder.md`` must name the
launcher's commands and the rules a user meets. The guide's claims about the code are pinned
where that code is tested: the declared channel count in ``test_launch_commands.py``, the
baseline clipping and the pooled full-day waveform in ``test_extract_full_day.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from nems_lbhb.unit_match.sources.base import clear_readers_for_tests

DOCS = Path(__file__).resolve().parents[2] / 'nems_lbhb' / 'unit_match' / 'docs'
PYTHON_FENCE = re.compile(r'^```python\n(.*?)^```[ \t]*$', re.MULTILINE | re.DOTALL)


@pytest.fixture(autouse=True)
def fresh_registry():
    """The example registers a reader; nothing it registers reaches another test."""
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


def _python_fences(path: Path) -> list:
    return PYTHON_FENCE.findall(path.read_text(encoding='utf-8'))


def test_adding_a_source_example_runs(tmp_path):
    path = DOCS / 'adding_a_source.md'
    fences = _python_fences(path)
    assert len(fences) == 1  # the doc's one python fence
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 (Task 16): the example's code carries the repository's AGENT EDIT tags | date: 2026-09-25]
    code = fences[0].strip().split('\n')
    assert code[0].startswith('# [AGENT EDIT START | agent: ') and code[-1] == '# [AGENT EDIT END]'
    # [AGENT EDIT END]
    namespace = {'__name__': 'adding_a_source_example'}
    exec(compile(fences[0], str(path), 'exec'), namespace)

    result = namespace['main'](tmp_path)

    assert result['tier'] == 'raw'
    assert result['day'] == 'D1'
    assert result['features']['halves'] is not None and result['features']['full'] is not None
    manifest = json.loads((tmp_path / 'lab' / 'demo' / 'project.json').read_text(encoding='utf-8'))
    day = manifest['days']['D1']
    assert day['readers'] == {'sort': 'example.npz@1', 'events': 'events_csv@1', 'raw': 'binary@1'}
    assert day['tier'] == 'raw' and day['features'] == result['features']
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 (Task 16): the example declares its channel count, and both of its units reach the full-day table, as the doc says | date: 2026-09-25]
    assert day['reader_options']['raw_n_channels'] == 4
    units = pd.read_csv(tmp_path / 'lab' / 'demo' / 'days' / 'D1' / 'features' / 'full' / 'units.csv')
    assert units['cellid'].tolist() == ['D1:0', 'D1:1']
    assert units['eligible'].tolist() == [True, True]
    # [AGENT EDIT END]


def test_project_folder_doc_names_commands():
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('init', 'extract', 'status', '--from', 'retargeted', 'condition', 'temp_wh.dat', 'new project'):
        assert word in text, word
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the guide names launch run, init --probe, the curation folder and stripped condition labels (Plan B Task 10) | date: 2026-09-28]
    for word in ('### run', 'launch run', '--probe', 'curation/', 'stripped of surrounding spaces'):
        assert word in text, word
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docs name both curation homes, say held-out sessions, and close the run section's tag before the launcher log (Follow-up Task 4) | date: 2026-09-29]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the project-folder guide names one curation history per run and launch export; no doc names the retired curated/pairs home (run-curation Task 10) | date: 2026-10-05]
def test_docs_name_run_history():
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('curation/runs/', 'history.csv', 'launch export'):
        assert word in text, word
    for path in sorted(DOCS.parent.rglob('*.md')):
        assert path.read_text(encoding='utf-8').count('curated/pairs') == 0, path
# [AGENT EDIT END]


def test_pair_logistic_texts_say_held_out_sessions():
    import inspect

    import nems_lbhb.unit_match.backends.local as local
    from nems_lbhb.unit_match import drift_validation
    from nems_lbhb.unit_match.backends.local import pair_logistic
    from nems_lbhb.unit_match.contract.names import NAMES
    texts = {
        'backends.local': local.__doc__,
        'pair_logistic': inspect.getsource(pair_logistic),
        "NAMES['backend']['pair_logistic'].help": NAMES['backend']['pair_logistic'].help,
        'drift_validation': inspect.getsource(drift_validation),
        'README.md': (DOCS.parent / 'README.md').read_text(encoding='utf-8'),
    }
    for name, text in texts.items():
        assert 'held-out days' not in text, name
        assert 'held-out sessions' in text, name


def test_run_section_tag_block_ends_before_launcher_log():
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    heading = '### `logs/launcher.log`'
    assert text.count(heading) == 1
    marker = text.index('the guide documents launch run and its output line (Plan B Task 10)')
    start = text.rindex('\n', 0, marker) + 1
    block = text[start:text.index('[AGENT EDIT END]', marker)]
    assert '### run' in block
    assert heading not in block
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docs name launch gui, cancelled, the queue and the theme (launcher-gui Task 11) | date: 2026-10-07]
def test_docs_name_launcher_gui():
    package = DOCS.parent
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('launch [--root DIR] gui [PROJECT]', '### gui', 'cancelled', 'Curate…', 'Inspect…', 'CLI only',
                 'Extract features', '.extract.lock', 'exits 130', 'Lab root…', 'waiting', 'Clear',
                 'viewer.md#theme'):
        assert word in text, word
    readme = (package / 'README.md').read_text(encoding='utf-8')
    for word in ('launch gui', '| `launch/` |'):
        assert word in readme, word
    source = (DOCS / 'adding_a_source.md').read_text(encoding='utf-8')
    for word in ('`export`, `gui`', 'CLI only'):
        assert word in source, word
    assert 'launch gui' in (DOCS / 'viewer.md').read_text(encoding='utf-8')
    development = (package / 'DEVELOPMENT.md').read_text(encoding='utf-8')
    for word in ('launch/window.py', 'viewer/theme.py'):
        assert word in development, word
# [AGENT EDIT END]
