# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer names the run's streams and evidence tags from its saved weights and run_evidence, and refuses a run saved without run_evidence (event-optional Task 12) | date: 2026-10-10]
"""The viewer shows the streams and evidence tags a run saved: the mixture columns, the ``wave`` table columns,
the evidence streams line and the header tags come from each group's saved ``weight_<stream>`` keys and
``run_evidence``; a run with mixture references but no valid ``run_evidence`` is refused."""
import dataclasses
import os
import shutil
from types import SimpleNamespace

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixture fills test reads face colours as hex (graphite-palettes Task 2) | date: 2026-10-11]
import matplotlib.colors
# [AGENT EDIT END]
import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match import backends
from nems_lbhb.unit_match.contract.provenance import Provenance
from nems_lbhb.unit_match.project.folder import Project
from nems_lbhb.unit_match.project.selection import Selection
from nems_lbhb.unit_match.run.project_run import run_selection
from nems_lbhb.unit_match.run.run_dir import load_result, save_result
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixture ticks test applies the graphite theme (graphite-palettes Task 3) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer import data, display, theme
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.app import App, open_run
from nems_lbhb.unit_match.viewer.panels.plots.mixture import MixturePanel
from nems_lbhb.unit_match.viewer.panels.tables import EvidencePanel
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt through viewer/qt.py, the binding pin (event-optional Task 12, fix round 1) | date: 2026-10-10]
from nems_lbhb.unit_match.viewer.qt import QtWidgets
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.state import Actions, PairState

from .synthetic_day import extracted_project
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: theme_env restores the theme the mixture fills test sets (graphite-palettes Task 2) | date: 2026-10-11]
from .theme_fixtures import theme_env  # noqa: F401  (a fixture, used by name)
# [AGENT EDIT END]
from .viewer_fakes import FAKE_MODE, DictData

SMALL = {'min_spikes': 2, 'n_splits': 2, 'min_pairs': 1}
NO_EVENTS = 'no events · shared conditions per day pair: D1+D2 0'
WHOLE = 'spont · whole recording · days without baseline windows: D2'


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope='module')
def mixed_run(tmp_path_factory):
    """A saved D1+D2 run of a project whose D2 has no events."""
    path = extracted_project(tmp_path_factory.mktemp('mixed'), no_event_days=('D2',))
    project = Project.open(path.parent, path.name)
    return run_selection(Selection(project, ('D1', 'D2')), 'functional_likelihood', dict(SMALL))


def _actions() -> Actions:
    """An ``Actions`` whose every call does nothing."""
    return Actions(select_pair=lambda a, b: None, set_verdict=lambda _: None, curation=lambda: None,
                   track_state=lambda: None, take_over=lambda: None, update_controls=lambda: None,
                   cluster_lookup=lambda: {}, review_threshold=lambda: .9, merge_refusal=lambda members: None)


def _evidence(*reasons) -> dict:
    return {'events': not any(r.startswith('no events') for r in reasons), 'spont': 'baseline', 'shared': {},
            'reasons': list(reasons)}


def _refs(*streams) -> dict:
    """A group's saved mixture references with a weight per stream and five-value references."""
    return {**{f'weight_{s}': 0.5 for s in streams}, 'bias': 0.0,
            'streams': {s: {'same': [1.0, 1.2, 1.4, 1.6, 1.8], 'diff': [-1.0, -0.8, -0.6, -0.4, -0.2]}
                        for s in [*streams, 'global']}}


def _group_run(group, streams, reasons=(), day_a='D1', day_b='D2') -> dict:
    """One ``group_runs`` entry: the group's mixture references, a fitted session pair and its run evidence."""
    fitted = {s: {'observed': [0.0, 0.5, 1.0], 'pi': 0.4} for s in [*streams, 'global']}
    return {'run_id': 'r-' + group, 'provenance': {
        'mixture_references': {group: _refs(*streams)},
        'session_pairs': [{'recording_group': group, 'session_a': day_a, 'session_b': day_b, 'streams': fitted}],
        'run_evidence': {group: _evidence(*reasons)}}}


def test_combined_label_from_saved_weights():
    def grouped(*streams):
        return Provenance(extra={'group_runs': {'P': _group_run('P', streams)}})

    assert display.combined_label(grouped('stim', 'phys', 'wave')) == 'Event response + physiology + waveform'
    assert display.combined_label(grouped('phys', 'wave')) == 'Physiology + waveform'
    top = {'mixture_references': {'P': _refs('phys', 'wave')}, 'feature_stream': 'combined',
           'evidence_display_name': 'Functional likelihood'}
    result = SimpleNamespace(backend_id='functional_likelihood', provenance=top)
    assert display.result_input_label(result) == 'Functional likelihood: Physiology + waveform'


def test_run_evidence_tags_from_groups():
    other = 'no events · shared conditions per day pair: D3+D4 0'
    provenance = Provenance(extra={'group_runs': {'P': _group_run('P', ('phys', 'wave'), (NO_EVENTS, WHOLE)),
                                                  'Q': _group_run('Q', ('phys', 'wave'), (other,), 'D3', 'D4')}})
    tags = display.run_evidence_tags(provenance)
    assert [tag for tag, _ in tags] == ['no events', 'spont · whole recording']
    assert tags == [('no events', NO_EVENTS + '\n' + other), ('spont · whole recording', WHOLE)]


def test_mixture_columns_follow_weights(qt):
    provenance = Provenance(extra={'group_runs': {'P': _group_run('P', ('phys', 'wave'), (NO_EVENTS,))}})
    (dp,) = data.apply_run_provenance([{'siteid_a': 'D1', 'siteid_b': 'D2', 'days_apart': 1}], provenance)
    assert {k for k in dp if k.startswith('weight_')} == {'weight_phys', 'weight_wave'}
    state = PairState()
    state.set_data(DictData(['D1:0', 'D2:0'], {}, {}, {'D1:0': 'D1', 'D2:0': 'D2'}))
    state.day_pair = dp
    state.row = pd.Series({'s_stim': np.nan, 'p_stim': np.nan, 's_phys': 1.0, 'p_phys': .7,
                           's_wave': 1.2, 'p_wave': .8, 's_global': 1.5, 'p_global': .9})
    panel = MixturePanel(state)
    panel.refresh()
    assert [ax.get_title().split('\n')[0] for ax in panel.axes] == ['physiology', 'waveform', 'combined']
    weights = panel.fig.get_suptitle()
    assert 'physiology 0.50' in weights and 'waveform 0.50' in weights and 'events' not in weights
    qt.processEvents()          # the panel's queued redraw runs while its canvas still exists
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the two mixture references fill in the status colours accept and reject (graphite-palettes Task 2) | date: 2026-10-11]
def test_mixture_fills_status_colours(qt, theme_env):
    display.set_theme('graphite')
    provenance = Provenance(extra={'group_runs': {'P': _group_run('P', ('phys', 'wave'), (NO_EVENTS,))}})
    (dp,) = data.apply_run_provenance([{'siteid_a': 'D1', 'siteid_b': 'D2', 'days_apart': 1}], provenance)
    state = PairState()
    state.set_data(DictData(['D1:0', 'D2:0'], {}, {}, {'D1:0': 'D1', 'D2:0': 'D2'}))
    state.day_pair = dp
    state.row = pd.Series({'s_stim': np.nan, 'p_stim': np.nan, 's_phys': 1.0, 'p_phys': .7,
                           's_wave': 1.2, 'p_wave': .8, 's_global': 1.5, 'p_global': .9})
    panel = MixturePanel(state)
    panel.refresh()
    assert len(panel.axes) == 3
    for ax in panel.axes:
        assert [matplotlib.colors.to_hex(c.get_facecolor()[0]) for c in ax.collections] == ['#3dd68c', '#ff7b72']
    qt.processEvents()          # the panel's queued redraw runs while its canvas still exists
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixture panel's tick labels, its twin probability axes included, are JetBrains Mono at its own 6 pt (graphite-palettes Task 3) | date: 2026-10-11]
def test_mixture_ticks_mono(qt, theme_env):
    theme.apply_theme(QtWidgets.QApplication.instance(), 'graphite')
    display.set_theme('graphite')
    provenance = Provenance(extra={'group_runs': {'P': _group_run('P', ('phys', 'wave'), (NO_EVENTS,))}})
    (dp,) = data.apply_run_provenance([{'siteid_a': 'D1', 'siteid_b': 'D2', 'days_apart': 1}], provenance)
    state = PairState()
    state.set_data(DictData(['D1:0', 'D2:0'], {}, {}, {'D1:0': 'D1', 'D2:0': 'D2'}))
    state.day_pair = dp
    state.row = pd.Series({'s_stim': np.nan, 'p_stim': np.nan, 's_phys': 1.0, 'p_phys': .7,
                           's_wave': 1.2, 'p_wave': .8, 's_global': 1.5, 'p_global': .9})
    panel = MixturePanel(state)
    panel.refresh()
    assert len(panel.axes) == 3
    for ax in panel.axes:
        label = ax.xaxis.get_major_ticks()[0].label1
        assert label.get_fontfamily() == ['JetBrains Mono']
        assert label.get_fontsize() == 6
    twins = [a for a in panel.fig.axes if a not in panel.axes]
    assert len(twins) == 3
    for a in twins:
        label = a.yaxis.get_major_ticks()[0].label2
        assert label.get_fontfamily() == ['JetBrains Mono']
        assert label.get_fontsize() == 6
    qt.processEvents()          # the panel's queued redraw runs while its canvas still exists
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the rivals table alone carries the wave column (native-same-day Task 10) | date: 2026-10-11]
def test_rivals_wave_column(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    table = window.rivals.table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers[headers.index('phys') + 1] == 'wave'
    units = ['a-01', 'b-01', 'b-02']
    pairs = pd.DataFrame({'cellid_a': ['a-01', 'a-01'], 'cellid_b': ['b-01', 'b-02'], 'p_global': [.9, .2],
                          'p_stim': [np.nan, np.nan], 'p_phys': [.6, .1], 'p_wave': [.7, .3]})
    window.set_data(DictData(units, {}, {}, {u: u.split('-')[0] for u in units}, pairs))
    window.select_pair('a-01', 'b-01')
    rivals = window.rivals.table
    expected = [['--', '10%', '30%'], ['--', '60%', '70%']]      # evt, phys, wave of each candidate
    assert sorted([rivals.item(i, c).text() for c in (2, 3, 4)] for i in range(rivals.rowCount())) == expected
    window._save_layout = lambda: None
    window.close()
# [AGENT EDIT END]


def test_evidence_streams_line_names_run_streams(qt):
    units = pd.DataFrame({'cellid': ['a-01', 'b-01', 'c-01', 'd-01'], 'session_id': ['a', 'b', 'c', 'd'],
                          'su_type': 'SU', 'siteid': ['a', 'b', 'c', 'd']})
    pairs = pd.DataFrame({'cellid_a': ['a-01', 'c-01'], 'cellid_b': ['b-01', 'd-01'], 'recording_group': ['E', 'N'],
                          'p_global': [.9, .8], 'p_stim': [.8, np.nan], 'p_phys': [.6, .5], 'p_wave': [.7, .9]})
    provenance = {'group_runs': {'E': _group_run('E', ('stim', 'phys', 'wave'), (), 'a', 'b'),
                                 'N': _group_run('N', ('phys', 'wave'), (NO_EVENTS,), 'c', 'd')}}
    result = backends.result_from_pairs('mixed', '1', units, pairs, score_column='p_global', run_id='mixed',
                                        provenance=provenance)
    state = PairState()
    state.set_data(DictData(units, {}, {}, dict(zip(units.cellid, units.siteid)), result=result))
    panel = EvidencePanel(state, _actions())
    lines = {}
    for _, row in state.pairs.iterrows():
        state.row = row
        panel.refresh()
        lines[row.recording_group] = panel.streams.text()
    assert lines == {'E': 'events 80% · physiology 60% · waveform 70%', 'N': 'physiology 50% · waveform 90%'}
    panel.close()


def test_run_without_evidence_refused(qt, mixed_run):
    run = mixed_run.parent / (mixed_run.name + '_v2')
    shutil.copytree(mixed_run, run)
    for path in (run, *run.rglob('*')):
        path.chmod(0o755 if path.is_dir() else 0o644)
    result = load_result(run / 'result.json')
    group_runs = result.provenance.get('group_runs')
    assert all('mixture_references' in entry['provenance'] for entry in group_runs.values())
    saved_v2 = {group: {**entry, 'provenance': {k: v for k, v in entry['provenance'].items() if k != 'run_evidence'}}
                for group, entry in group_runs.items()}
    save_result(dataclasses.replace(result, version='2',
                                    provenance=result.provenance.merged(extra={'group_runs': saved_v2})),
                run / 'result.json', overwrite=True)
    with pytest.raises(ValueError) as caught:
        open_run(run)
    assert str(caught.value) == 'run saved before event-optional matchers · rerun'


def test_header_tags_from_run_evidence(qt, mixed_run):
    window = open_run(mixed_run)
    try:
        assert window.evidence_tags_shown() == [('no events', NO_EVENTS), ('spont · whole recording', WHOLE)]
        assert [label.property('tone') for label in window._evidence_labels] == ['unsure', 'unsure']
    finally:
        window._save_layout = lambda: None
        window.close()
        qt.processEvents()
# [AGENT EDIT END]
