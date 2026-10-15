# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify direct benchmark viewing, measured coordinates and isolated curation | date: 2026-09-17]
import copy
import os

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match.benchmark import SourceRecording, build_dataset
from nems_lbhb.unit_match import backends
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the load and validate tests read the same-day mode (viewer-split Task 3) | date: 2026-10-06]
from nems_lbhb.unit_match.viewer.modes import same_day
from nems_lbhb.unit_match.viewer.modes.same_day import validate_benchmark_result
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every test reads settings from a fresh folder under the pytest base temp (day-table-keys Task 6) | date: 2026-10-14]
from tests.unit_match.settings_fixtures import tmp_settings  # noqa: F401
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no registered backend tracks cached pair scores; fabricate results with the common tracker under a real id (Task 3) | date: 2026-09-27]
def track(units, pairs, **kwargs):
    from nems_lbhb.unit_match.backends import result_from_pairs
    return result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', **kwargs)
# [AGENT EDIT END]


@pytest.fixture
def example():
    voltage = np.zeros((2000, 6))
    spikes = {'secret-source-a': np.arange(10, 1990, 20),
              'secret-source-b': np.arange(20, 1980, 20)}
    for source, channel in zip(spikes, [1, 4]):
        for sample in spikes[source]:
            voltage[sample - 1:sample + 2, channel] = [.2, -2., 1.]
    stimuli = [dict(stimulus='sound:left' if trial < 4 else 'sound:right',
                    start=part * 1000 + trial * 100, stop=part * 1000 + trial * 100 + 100)
               for part in range(2) for trial in range(8)]
    source = SourceRecording('source_recording', 'animal:3', 'test', 1000.,
        np.column_stack((np.zeros(6), 1000 + np.arange(6) * 20)), np.array([[0, 1000], [1000, 2000]]),
        spikes, tuple(spikes), lambda a, z: voltage[a:z], 'synthetic common sample clock',
        {'raw_source': 'synthetic voltage fixture'}, stimuli=stimuli)
    dataset = build_dataset([source], dataset_id='viewer-fixture',
                            extraction_options={'before': 2, 'after': 3, 'bin_seconds': .02})
    labels = dataset.labels.set_index('cellid').source_unit_id
    pairs = dataset.pairs.copy()
    pairs['p_global'] = [.99 if labels[a] == labels[z] else .01
                          for a, z in zip(pairs.cellid_a, pairs.cellid_b)]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(dataset.units, pairs,
    # [AGENT EDIT END]
                                  provenance=dict(dataset.metadata, benchmark_partition='test'), run_id='fixture-run')
    return dataset, result


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day mode builds the run's RunData (viewer-split Task 3) | date: 2026-10-06]
def test_display_cache_keeps_measured_geometry_and_centered_waveform_patch(example):
    dataset, result = example
    data = same_day.run_data(dataset, result)
    assert set(data.units().cellid) == set(result.units.cellid)
    assert all(data.geometry(site)[:, 1].min() == 1000 for site in set(data.sites().values()))
    for cellid in data.units().cellid:
        unit = data.unit(cellid)
        assert unit.patch.shape == (41, 5)
        np.testing.assert_allclose(unit.patch[20], unit.waveform)
        assert unit.fingerprint_bin_seconds == .02
        assert set(unit.fingerprints) == {'sound:left', 'sound:right'}
    assert 'source_unit_id' not in data.units()
    assert all(d['days_apart'] == 0 for d in data.day_pairs())
# [AGENT EDIT END]


def test_result_must_cover_exact_dataset_partition_and_sessions(example):
    dataset, result = example
    altered = copy.deepcopy(result)
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write through .merged (Task 12) | date: 2026-09-22]
    altered.provenance = altered.provenance.merged(extra={'cache_built': 'wrong-cache'})
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='cache_built'):
        validate_benchmark_result(dataset, altered)
    altered = copy.deepcopy(result)
    altered.units.loc[0, 'session_id'] = 'wrong-session'
    with pytest.raises(ValueError, match='session identities'):
        validate_benchmark_result(dataset, altered)
    partial_ids = list(result.units.cellid)[:2]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: compat run_backend takes a JSON mapping; Provenance is typed (Task 12) | date: 2026-09-22]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    partial = track(result.units[result.units.cellid.isin(partial_ids)],
    # [AGENT EDIT END]
                                   pd.DataFrame(), provenance=result.provenance.to_json())
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='every unit'):
        validate_benchmark_result(dataset, partial)


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: PySide6 is required, so the GUI checks never pass as skips (Plan B Task 6) | date: 2026-09-28]
    from PySide6 import QtWidgets as widgets
    # [AGENT EDIT END]
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield app
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: restore the validate_benchmark_result recording-identity test Task 4 does not retire (run-curation Task 4 fix round 1) | date: 2026-10-05]
def test_import_cannot_cross_original_recordings_or_change_group_identity():
    from test_unit_match_benchmark import source
    dataset = build_dataset([source(), source(well='animal:2', session='source2')],
        dataset_id='independent-recordings', extraction_options={'before': 2, 'after': 3})
    result = track(dataset.units, dataset.pairs.assign(p_global=0.),
        provenance=dict(dataset.metadata, benchmark_partition='test'))
    groups = list(dataset.units.groupby('recording_group'))
    a, b = groups[0][1].cellid.iloc[0], groups[1][1].cellid.iloc[0]
    wrong = copy.deepcopy(result)
    wrong.tracks = ((a, b),)
    wrong.unmatched = tuple(c for c in wrong.units.cellid if c not in (a, b))
    with pytest.raises(ValueError, match='tracks cannot cross'):
        validate_benchmark_result(dataset, wrong)
    wrong = copy.deepcopy(result)
    pair = wrong.pairs.iloc[0].copy()
    pair['cellid_a'], pair['cellid_b'] = a, b
    wrong.pairs = pd.concat([wrong.pairs, pd.DataFrame([pair])], ignore_index=True)
    with pytest.raises(ValueError, match='within one original recording'):
        validate_benchmark_result(dataset, wrong)
    wrong = copy.deepcopy(result)
    wrong.units.loc[0, 'recording_group'] = 'wrong-original-recording'
    with pytest.raises(ValueError, match='original recording identities'):
        validate_benchmark_result(dataset, wrong)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the halves viewer opens same-day project runs from their run folder (Plan B Task 6) | date: 2026-09-28]
FL = 'functional_likelihood'


@pytest.fixture(scope='module')
def halves_project(tmp_path_factory):
    """A project of D1 and D2 with a functional_likelihood run of D1 alone."""
    from nems_lbhb.unit_match.project.folder import Project
    from nems_lbhb.unit_match.project.selection import Selection
    from nems_lbhb.unit_match.run.project_run import run_selection
    from tests.unit_match.synthetic_day import extracted_project
    folder = extracted_project(tmp_path_factory.mktemp('project'))
    project = Project.open(folder.parent, folder.name)
    selection = Selection(project, ('D1',))
    return dict(project=project, selection=selection, run=run_selection(selection, FL))


def test_halves_open_run_uses_project_curation(qt, halves_project):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer opens the run (viewer-split Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer.app import open_run
    # [AGENT EDIT END]
    window = open_run(halves_project['run'])
    curation = halves_project['project'].path / 'curation'
    result = backends.load_result(halves_project['run'] / 'result.json')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-day run opens with no curation controller (run-curation Task 5) | date: 2026-10-05]
    assert window.curation is None
    # [AGENT EDIT END]
    assert window.match_result.run_id == result.run_id
    window.close()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day mode's layout file (viewer-split Task 8) | date: 2026-10-07]
    assert (halves_project['project'].root / 'viewer' / 'same_day_layout.json').is_file()
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: test_halves_open_run_refuses_across_days_run is deleted: open_run opens an across-days run in its own mode (viewer-split Task 8) | date: 2026-10-07]
# [AGENT EDIT END]


def test_halves_result_checked_against_selection(qt, halves_project):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day mode's load checks the result against the selection hash (viewer-split Task 8) | date: 2026-10-07]
    selection = halves_project['selection']
    result = backends.load_result(halves_project['run'] / 'result.json')
    result.provenance = result.provenance.merged(input_manifest_sha256='0' * 64)
    with pytest.raises(ValueError, match='input_manifest_sha256'):
        same_day.load(selection, selection.dataset(), result, {})
    # [AGENT EDIT END]


def test_benchmark_viewer_main_modes(qt, halves_project, tmp_path, capsys, monkeypatch):
    from PySide6 import QtWidgets
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer command (viewer-split Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer import app as viewer_app
    # [AGENT EDIT END]
    run = str(halves_project['run'])
    monkeypatch.setattr(QtWidgets.QApplication, 'exec', lambda self: 0)
    for argv in ([], ['--run', run, '--input', str(tmp_path)], ['--run', run, '--result', str(tmp_path / 'r.json')],
                 ['--run', run, '--result', '']):
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer command (viewer-split Task 8) | date: 2026-10-07]
        with pytest.raises(SystemExit) as caught:
            viewer_app.main(argv)
        # [AGENT EDIT END]
        assert caught.value.code == 2
    assert '--result' in capsys.readouterr().err
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: --input is retired: an argparse error, exit 2 (run-curation Task 4) | date: 2026-10-05]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer command (viewer-split Task 8) | date: 2026-10-07]
    with pytest.raises(SystemExit) as caught:
        viewer_app.main(['--input', 'x'])
    # [AGENT EDIT END]
    assert caught.value.code == 2
    # [AGENT EDIT END]
    opened = []

    class Window:
        def show(self):
            opened.append('shown')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer command opens through viewer.app.open_run (viewer-split Task 8) | date: 2026-10-07]
    monkeypatch.setattr(viewer_app, 'open_run', lambda path: opened.append(path) or Window())
    assert viewer_app.main(['--run', run]) == 0
    # [AGENT EDIT END]
    assert opened == [halves_project['run'], 'shown']
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project run's window imports only completed runs of its own selection (Plan B Task 6) | date: 2026-09-28]
def test_halves_import_only_from_selection_runs(qt, tmp_path):
    import shutil
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer opens the run (viewer-split Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer.app import open_run
    # [AGENT EDIT END]
    from nems_lbhb.unit_match.project.folder import Project
    from nems_lbhb.unit_match.project.selection import Selection
    from nems_lbhb.unit_match.run.project_run import run_selection
    from tests.unit_match.synthetic_day import extracted_project
    folder = extracted_project(tmp_path / 'p')
    project = Project.open(folder.parent, folder.name)
    selection = Selection(project, ('D1',))
    shown, other = run_selection(selection, FL), run_selection(selection, FL)
    window = open_run(shown)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shown tracks come from track_state(), not a ledger (run-curation Task 5) | date: 2026-10-05]
    shown_state = window.track_state()
    # [AGENT EDIT END]
    loose = tmp_path / 'loose'
    loose.mkdir()
    shutil.copyfile(other / 'result.json', loose / 'result.json')
    with pytest.raises(FileNotFoundError, match='has no selection.json'):
        window.import_match_result(loose / 'result.json')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shown tracks come from track_state(), not a ledger (run-curation Task 5) | date: 2026-10-05]
    assert window.track_state() == shown_state
    # [AGENT EDIT END]
    imported = window.import_match_result(other / 'result.json')
    assert window.match_result.run_id == imported.run_id != backends.load_result(shown / 'result.json').run_id
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-day run is inspect-only: no controller, mutation controls disabled, one tag (run-curation Task 5, D22) | date: 2026-10-05]
def test_halves_run_is_inspect_only(qt, halves_project):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer opens the run (viewer-split Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer.app import open_run
    # [AGENT EDIT END]
    window = open_run(halves_project['run'])
    assert window.curation is None
    assert window.tracked.buttons['Accept'].isEnabled() is False
    assert window.tracked.inspect_tag.text() == 'inspect-only'
    assert not window.tracked.inspect_tag.isHidden()
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pins tmp_settings: with no fixture requested, the settings file lies in a fresh settings folder under the pytest base temp, never ~/.config/multiphy or an after folder (day-table-keys Task 6) | date: 2026-10-14]
def test_settings_live_in_a_fresh_tmp_folder():
    import re
    from pathlib import Path
    from nems_lbhb.unit_match.viewer import theme
    p = Path(theme.settings().fileName())
    assert p.name == 'multiphy.ini'
    assert p.parent.name == 'multiphy'
    assert re.fullmatch(r'settings\d+', p.parent.parent.name) is not None
# [AGENT EDIT END]
