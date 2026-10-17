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

from multiphy.sources.base import clear_readers_for_tests

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / 'multiphy' / 'docs'
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
    assert day['reader_options']['raw_n_channels'] == 4
    units = pd.read_csv(tmp_path / 'lab' / 'demo' / 'days' / 'D1' / 'features' / 'full' / 'units.csv')
    assert units['cellid'].tolist() == ['D1:0', 'D1:1']
    assert units['eligible'].tolist() == [True, True]


def test_project_folder_doc_names_commands():
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('init', 'extract', 'status', '--from', 'retargeted', 'condition', 'temp_wh.dat', 'new project'):
        assert word in text, word
    for word in ('### run', 'launch run', '--probe', 'curation/', 'stripped of surrounding spaces'):
        assert word in text, word


def test_docs_name_run_history():
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('curation/runs/', 'history.csv', 'launch export'):
        assert word in text, word
    for path in sorted(DOCS.parent.rglob('*.md')):
        assert path.read_text(encoding='utf-8').count('curated/pairs') == 0, path


def test_pair_logistic_texts_say_held_out_sessions():
    import inspect

    import multiphy.backends.local as local
    from multiphy import drift_validation
    from multiphy.backends.local import pair_logistic
    from multiphy.contract.names import NAMES
    texts = {
        'backends.local': local.__doc__,
        'pair_logistic': inspect.getsource(pair_logistic),
        "NAMES['backend']['pair_logistic'].help": NAMES['backend']['pair_logistic'].help,
        'drift_validation': inspect.getsource(drift_validation),
        'README.md': (REPO / 'README.md').read_text(encoding='utf-8'),
    }
    for name, text in texts.items():
        assert 'held-out days' not in text, name
        assert 'held-out sessions' in text, name


def test_docs_name_launcher_gui():
    package = DOCS.parent
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('launch [--root DIR] gui [PROJECT]', '### gui', 'cancelled', 'Curate…', 'Inspect…', 'CLI only',
                 'Extract features', '.extract.lock', 'exits 130', 'Lab root…', 'waiting', 'Clear',
                 'viewer.md#theme'):
        assert word in text, word
    readme = (REPO / 'README.md').read_text(encoding='utf-8')
    for word in ('launch gui', '| `launch/` |'):
        assert word in readme, word
    source = (DOCS / 'adding_a_source.md').read_text(encoding='utf-8')
    for word in ('`export`, `gui`', 'CLI only'):
        assert word in source, word
    assert 'launch gui' in (DOCS / 'viewer.md').read_text(encoding='utf-8')
    development = (package / 'DEVELOPMENT.md').read_text(encoding='utf-8')
    for word in ('launch/window.py', 'viewer/theme.py'):
        assert word in development, word
