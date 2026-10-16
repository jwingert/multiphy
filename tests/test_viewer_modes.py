# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run modes load a run into RunData; fix round 1: test_modes_registered names the eight panels for each mode (viewer-split Task 3) | date: 2026-10-06]
"""The across-days and same-day modes: each loads a run into the typed ``RunData`` the panels read."""
import copy
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from multiphy.project.selection import selection_of_run
from multiphy.run.run_dir import load_result
from multiphy.viewer import data
from multiphy.viewer.modes import MODES, across_days, same_day

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an across-days run of D1, D2, D3 and a same-day run of D1, from extracted_project and run_selection; the old viewers and their harness are gone (viewer-split Task 9) | date: 2026-10-07]
from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection

from .synthetic_day import extracted_project
# [AGENT EDIT END]

REPO = Path(__file__).resolve().parents[1]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an across-days run of D1, D2, D3 and a same-day run of D1, from extracted_project and run_selection (viewer-split Task 9) | date: 2026-10-07]
@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    path = extracted_project(tmp_path_factory.mktemp('modes'), days=('D1', 'D2', 'D3'))
    project = Project.open(path.parent, path.name)
    return {'across_days': run_selection(Selection(project, ('D1', 'D2', 'D3')), 'functional_likelihood'),
            'same_day': run_selection(Selection(project, ('D1',)), 'functional_likelihood')}
# [AGENT EDIT END]


def _opened(run: Path):
    selection = selection_of_run(run)
    return selection, selection.dataset(), load_result(run / 'result.json')


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no day-pair label to assert (native-same-day Task 10 fix round 1) | date: 2026-10-11]
def test_appended_day_pair_has_well_and_dates(runs):
    selection, dataset, result = _opened(runs['across_days'])
    dataset.units = dataset.units.copy()
    dataset.units.loc[dataset.units.session_id == 'D3', 'well'] = 'Q'   # no (D1, D3) day pair: different wells
    result = copy.deepcopy(result)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the plain run scores no pair, so the appended (D1:0, D3:0) row is built from the result's columns with the harness's below-threshold values (viewer-split Task 9) | date: 2026-10-07]
    row = {'cellid_a': 'D1:0', 'cellid_b': 'D3:0', 'p_global': .2, 'p_stim': .2, 'p_phys': .2, 'match_score': .2,
           'selected': False, 'decision': 'below_threshold'}
    extra = pd.DataFrame([{c: row.get(c) for c in result.pairs.columns}])
    # [AGENT EDIT END]
    result.pairs = pd.concat([result.pairs, extra], ignore_index=True)
    loaded = across_days.load(selection, dataset, result, selection.day_extras())
    (dp,) = [d for d in loaded.day_pairs() if {d['siteid_a'], d['siteid_b']} == {'D1', 'D3'}]
    assert 'well' in dp
    assert dp['date_a'] == '2026-01-01' and dp['date_b'] == '2026-01-03'
    assert dp['days_apart'] == 2
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-day proposed_link row loads in across-days mode and adds no same-site day pair (native-same-day Task 7) | date: 2026-10-11]
def test_across_days_accepts_same_day_pairs(runs):
    selection, dataset, result = _opened(runs['across_days'])
    result = copy.deepcopy(result)
    row = {'cellid_a': 'D1:0', 'cellid_b': 'D1:1', 'p_global': .2, 'p_stim': .2, 'p_phys': .2, 'match_score': .2,
           'selected': False, 'decision': 'proposed_link'}
    extra = pd.DataFrame([{c: row.get(c) for c in result.pairs.columns}])
    result.pairs = pd.concat([result.pairs, extra], ignore_index=True)
    loaded = across_days.load(selection, dataset, result, selection.day_extras())
    shown = loaded.pairs()
    same = shown[(shown.cellid_a == 'D1:0') & (shown.cellid_b == 'D1:1')]
    assert len(same) == 1
    assert same.iloc[0].siteid_a == 'D1' and same.iloc[0].siteid_b == 'D1'
    assert same.iloc[0].decision == 'proposed_link'
    assert not [d for d in loaded.day_pairs() if d['siteid_a'] == d['siteid_b']]


def test_same_day_mode_accepts_same_part_pairs(runs):
    _, dataset, result = _opened(runs['same_day'])
    result = copy.deepcopy(result)
    first = result.units.session_id.iloc[0]
    a, b = list(result.units.loc[result.units.session_id == first, 'cellid'])[:2]
    row = {'cellid_a': a, 'cellid_b': b, 'p_global': .2, 'p_stim': .2, 'p_phys': .2, 'match_score': .2,
           'selected': False, 'decision': 'proposed_link'}
    extra = pd.DataFrame([{c: row.get(c) for c in result.pairs.columns}])
    result.pairs = pd.concat([result.pairs, extra], ignore_index=True)
    loaded = same_day.run_data(dataset, result)
    shown = loaded.pairs()
    assert ((shown.cellid_a == a) & (shown.cellid_b == b)).sum() == 1
# [AGENT EDIT END]


def test_same_day_load_checks_selection_hash(runs):
    _, dataset, result = _opened(runs['same_day'])
    with pytest.raises(ValueError, match='input_manifest_sha256'):
        same_day.run_data(dataset, result, input_manifest_sha256='0' * 64)


def test_modes_import_no_qt():
    code = ('import sys\n'
            'import multiphy.viewer.modes\n'
            "print(sorted(m for m in sys.modules if m.startswith(('PySide6', 'PyQt5', "
            "'matplotlib.backends.backend_qt'))))\n")
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONPATH=str(REPO)))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == '[]'


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram are two plots, ten plots each; the tag wraps the whole function (browse-same-day Task 5 fix round 1) | date: 2026-10-12]
def test_modes_registered():
    assert set(MODES) == {'across_days', 'same_day'}
    assert MODES['same_day'].curates is False
    panels = {'waveform', 'wfmap', 'isi', 'acg', 'footprint', 'probe', 'fingerprint', 'tuning', 'mixture', 'drift'}
    for mode in MODES.values():
        assert len(mode.plots) == 10
        assert set(mode.plots) == panels
        assert 'drift' in mode.plots
# [AGENT EDIT END]


def test_across_merge_refusal():
    units = pd.DataFrame({'cellid': ['D1:0', 'D2:0', 'D3:0'], 'siteid': ['D1', 'D2', 'D3'],
                          'well': ['P', 'P', 'Q'], 'session_date': ['2026-01-01', '2026-01-02', '2026-01-03']})
    loaded = data.RunData(
        mode='across_days', result=None, words=across_days.WORDS, title='', info='', info_tip='',
        unit_table=units, unit_map={}, pair_table=pd.DataFrame(columns=list(data.PAIR_COLUMNS)), day_pair_list=(),
        geometries={}, geometry_sources={}, day_names={}, freq_axes={}, min_fingerprint_repeats=4, parts=frozenset())
    assert across_days.merge_refusal(loaded, ['D1:0', 'D3:0']) == (
        'Merge rejected: selected observations belong to different wells or lack well/date identity.')
    assert across_days.merge_refusal(loaded, ['D1:0', 'D2:0']) is None
# [AGENT EDIT END]
