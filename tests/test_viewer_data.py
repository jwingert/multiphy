# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Qt-free display texts, RunData and Selection.mode (viewer-split Task 2) | date: 2026-10-06]
"""The viewer's Qt-free modules: display texts, the typed `RunData` panels read, and a selection's mode."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from multiphy.backends import result_from_pairs
from multiphy.contract.provenance import Provenance
from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.viewer import data

from .synthetic_day import extracted_project
from .test_run_provenance_panels import _group_run

REPO = Path(__file__).resolve().parents[1]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Words has four fields, no pair words (native-same-day Task 10) | date: 2026-10-11]
def _words():
    return data.Words(track_column='track', member_column='days', observations='days', import_tip='')
# [AGENT EDIT END]


def _run_data(**overrides):
    fields = dict(mode='across_days', result=None, words=_words(), title='', info='', info_tip='',
                  unit_table=pd.DataFrame({'cellid': ['a-1'], 'siteid': ['D1']}),
                  unit_map={'a-1': data.UnitData(cellid='a-1', siteid='D1')},
                  pair_table=pd.DataFrame(columns=list(data.PAIR_COLUMNS)), day_pair_list=(),
                  geometries={}, geometry_sources={}, day_names={}, freq_axes={},
                  min_fingerprint_repeats=4, parts=frozenset())
    fields.update(overrides)
    return data.RunData(**fields)


def test_qt_free_modules_import_no_qt():
    code = ('import sys\n'
            'import multiphy.viewer.display\n'
            'import multiphy.viewer.data\n'
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the drift frame reader imports no Qt either (drift-view Task 1) | date: 2026-10-08]
            'import multiphy.viewer.drift_frame\n'
            # [AGENT EDIT END]
            "print(sorted(m for m in sys.modules if m.startswith(('PySide6', 'PyQt5', "
            "'matplotlib.backends.backend_qt'))))\n")
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == '[]'


def test_run_data_refuses_missing_pair_column():
    pairs = pd.DataFrame(columns=[c for c in data.PAIR_COLUMNS if c != 'score_kind'])
    with pytest.raises(ValueError, match='score_kind'):
        _run_data(pair_table=pairs)


def test_run_data_refuses_unknown_part():
    with pytest.raises(ValueError, match='unknown parts'):
        _run_data(parts=frozenset({'psth'}))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: offsets is no longer a part (drift-view Task 3) | date: 2026-10-08]
    with pytest.raises(ValueError, match='unknown parts'):
        _run_data(parts=frozenset({'offsets'}))
    # [AGENT EDIT END]


def test_run_data_defaults():
    run = _run_data()
    assert run.day_name('D1') == 'D1'
    assert run.geometry_source('D1') == 'unknown'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: RunData has no day_offset (drift-view Task 3) | date: 2026-10-08]
    assert not hasattr(run, 'day_offset')
    # [AGENT EDIT END]
    assert run.freq_axis('D1') is None
    assert run.sites() == {'a-1': 'D1'}


def test_display_pairs_sets_score_kind():
    units = pd.DataFrame({'cellid': ['a-01', 'b-01'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'], 'p_global': [.99]})
    result = result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', run_id='r1')
    assert 'score_kind' not in result.pairs
    shown = data.display_pairs(result, {'a-01': 'a', 'b-01': 'b'})
    assert (shown['score_kind'] == result.score.kind.value).all()
    assert len(shown) == 1


def test_apply_run_provenance_leaves_input():
    day_pairs = [{'siteid_a': 'D1', 'siteid_b': 'D2', 'days_apart': 1}]
    provenance = Provenance(extra={'group_runs': {'P': _group_run()}})
    out = data.apply_run_provenance(day_pairs, provenance)
    assert out[0]['streams']['stim']['pi'] == 0.25
    assert 'streams' not in day_pairs[0]


def test_selection_mode(tmp_path):
    folder = extracted_project(tmp_path)
    project = Project.open(folder.parent, folder.name)
    one = Selection(project, ('D1',))
    two = Selection(project, ('D1', 'D2'))
    assert one.mode == 'same_day'
    assert two.mode == 'across_days'
    assert one.record()['mode'] == one.mode
    assert two.record()['mode'] == two.mode
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: complete_day_pairs skips a site pair within one site (native-same-day Task 7) | date: 2026-10-11]
def test_complete_day_pairs_skips_same_site():
    out = data.complete_day_pairs([{'siteid_a': 'A', 'siteid_b': 'A', 'days_apart': 0}],
                                  pd.DataFrame({'siteid_a': ['A', 'A'], 'siteid_b': ['A', 'B']}))
    assert out == [{'siteid_a': 'A', 'siteid_b': 'B', 'days_apart': 'unknown'}]
# [AGENT EDIT END]
