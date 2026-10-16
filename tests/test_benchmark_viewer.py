import copy
import os

import numpy as np
import pandas as pd
import pytest

from multiphy.benchmark import SourceRecording, build_dataset
from multiphy import backends
from multiphy.viewer.modes import same_day
from multiphy.viewer.modes.same_day import validate_benchmark_result
from tests.settings_fixtures import tmp_settings  # noqa: F401


def track(units, pairs, **kwargs):
    from multiphy.backends import result_from_pairs
    return result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', **kwargs)


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
    result = track(dataset.units, pairs,
                                  provenance=dict(dataset.metadata, benchmark_partition='test'), run_id='fixture-run')
    return dataset, result


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


def test_result_must_cover_exact_dataset_partition_and_sessions(example):
    dataset, result = example
    altered = copy.deepcopy(result)
    altered.provenance = altered.provenance.merged(extra={'cache_built': 'wrong-cache'})
    with pytest.raises(ValueError, match='cache_built'):
        validate_benchmark_result(dataset, altered)
    altered = copy.deepcopy(result)
    altered.units.loc[0, 'session_id'] = 'wrong-session'
    with pytest.raises(ValueError, match='session identities'):
        validate_benchmark_result(dataset, altered)
    partial_ids = list(result.units.cellid)[:2]
    partial = track(result.units[result.units.cellid.isin(partial_ids)],
                                   pd.DataFrame(), provenance=result.provenance.to_json())
    with pytest.raises(ValueError, match='every unit'):
        validate_benchmark_result(dataset, partial)


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtWidgets as widgets
    app = widgets.QApplication.instance() or widgets.QApplication([])
    yield app


def test_import_cannot_cross_original_recordings_or_change_group_identity():
    from tests.test_benchmark import source
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


FL = 'functional_likelihood'


@pytest.fixture(scope='module')
def halves_project(tmp_path_factory):
    """A project of D1 and D2 with a functional_likelihood run of D1 alone."""
    from multiphy.project.folder import Project
    from multiphy.project.selection import Selection
    from multiphy.run.project_run import run_selection
    from tests.synthetic_day import extracted_project
    folder = extracted_project(tmp_path_factory.mktemp('project'))
    project = Project.open(folder.parent, folder.name)
    selection = Selection(project, ('D1',))
    return dict(project=project, selection=selection, run=run_selection(selection, FL))


def test_halves_open_run_uses_project_curation(qt, halves_project):
    from multiphy.viewer.app import open_run
    window = open_run(halves_project['run'])
    curation = halves_project['project'].path / 'curation'
    result = backends.load_result(halves_project['run'] / 'result.json')
    assert window.curation is None
    assert window.match_result.run_id == result.run_id
    window.close()
    assert (halves_project['project'].root / 'viewer' / 'same_day_layout.json').is_file()




def test_halves_result_checked_against_selection(qt, halves_project):
    selection = halves_project['selection']
    result = backends.load_result(halves_project['run'] / 'result.json')
    result.provenance = result.provenance.merged(input_manifest_sha256='0' * 64)
    with pytest.raises(ValueError, match='input_manifest_sha256'):
        same_day.load(selection, selection.dataset(), result, {})


def test_benchmark_viewer_main_modes(qt, halves_project, tmp_path, capsys, monkeypatch):
    from PySide6 import QtWidgets
    from multiphy.viewer import app as viewer_app
    run = str(halves_project['run'])
    monkeypatch.setattr(QtWidgets.QApplication, 'exec', lambda self: 0)
    for argv in ([], ['--run', run, '--input', str(tmp_path)], ['--run', run, '--result', str(tmp_path / 'r.json')],
                 ['--run', run, '--result', '']):
        with pytest.raises(SystemExit) as caught:
            viewer_app.main(argv)
        assert caught.value.code == 2
    assert '--result' in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        viewer_app.main(['--input', 'x'])
    assert caught.value.code == 2
    opened = []

    class Window:
        def show(self):
            opened.append('shown')
    monkeypatch.setattr(viewer_app, 'open_run', lambda path: opened.append(path) or Window())
    assert viewer_app.main(['--run', run]) == 0
    assert opened == [halves_project['run'], 'shown']


def test_halves_import_only_from_selection_runs(qt, tmp_path):
    import shutil
    from multiphy.viewer.app import open_run
    from multiphy.project.folder import Project
    from multiphy.project.selection import Selection
    from multiphy.run.project_run import run_selection
    from tests.synthetic_day import extracted_project
    folder = extracted_project(tmp_path / 'p')
    project = Project.open(folder.parent, folder.name)
    selection = Selection(project, ('D1',))
    shown, other = run_selection(selection, FL), run_selection(selection, FL)
    window = open_run(shown)
    shown_state = window.track_state()
    loose = tmp_path / 'loose'
    loose.mkdir()
    shutil.copyfile(other / 'result.json', loose / 'result.json')
    with pytest.raises(FileNotFoundError, match='has no selection.json'):
        window.import_match_result(loose / 'result.json')
    assert window.track_state() == shown_state
    imported = window.import_match_result(other / 'result.json')
    assert window.match_result.run_id == imported.run_id != backends.load_result(shown / 'result.json').run_id
    window.close()


def test_halves_run_is_inspect_only(qt, halves_project):
    from multiphy.viewer.app import open_run
    window = open_run(halves_project['run'])
    assert window.curation is None
    assert window.tracked.buttons['Accept'].isEnabled() is False
    assert window.tracked.inspect_tag.text() == 'inspect-only'
    assert not window.tracked.inspect_tag.isHidden()
    window.close()


def test_settings_live_in_a_fresh_tmp_folder():
    import re
    from pathlib import Path
    from multiphy.viewer import theme
    p = Path(theme.settings().fileName())
    assert p.name == 'multiphy.ini'
    assert p.parent.name == 'multiphy'
    assert re.fullmatch(r'settings\d+', p.parent.parent.name) is not None
