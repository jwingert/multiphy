# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: selections of extracted project days (Plan B Task 1) | date: 2026-09-28]
"""A ``Selection`` names one day, or two or more days, of a project and builds their dataset."""
from __future__ import annotations

import hashlib
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the signature test reads parameter order (event-optional Task 2) | date: 2026-10-09]
import inspect
# [AGENT EDIT END]
import json
from pathlib import Path
import typing
from typing import Optional, Union

import pandas as pd
import pytest

from multiphy import benchmark
from multiphy.benchmark import BenchmarkDataset, fingerprint_matrix
from multiphy.contract.inputs import RawWaveformBundle
from multiphy.launch import commands
from multiphy.project import folder
from multiphy.project.folder import RUNS_DIR, Project
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: selection_inputs, read_source_entry and the runner they are checked against (launcher-gui Task 1) | date: 2026-10-07]
import numpy as np

from multiphy.contract.inputs import Input
from multiphy.project.selection import (AcrossDaysDataset, Selection, default_days,
                                                     selection_inputs, selection_of_run)
from multiphy.run import runner
from multiphy.sources.normalize import read_source, read_source_entry

from .synthetic_day import SyntheticDay, extracted_project, write_day, write_legacy_inputs
# [AGENT EDIT END]

WEAK_REASON = 'Too few spikes in temporal split.'


def _open(project_dir: Path) -> Project:
    return Project.open(project_dir.parent, project_dir.name)


@pytest.fixture(scope='module')
def weak(tmp_path_factory) -> Project:
    """D1 and D2 extracted from raw; D1 holds the weak unit ``D1:4``."""
    return _open(extracted_project(tmp_path_factory.mktemp('weak'), weak_units=('D1',)))


def _set_probe(project: Project, probe: str) -> None:
    project.update(lambda manifest: manifest['identity'].update(probe=probe))


def test_default_days_are_extracted_days(tmp_path):
    project = _open(extracted_project(tmp_path))
    sd = write_day(tmp_path / 'in', 'D3', seed=2, date='2026-01-03')
    commands.init_day(project.root, 'P', day='D3', sort=str(sd.sort_dir), date=sd.date,
                      events=str(sd.events_path), raw=str(sd.raw_path), uv_per_bit=sd.uv_per_bit)
    assert default_days(project) == ('D1', 'D2')


def test_days_sorted_and_record(weak):
    selection = Selection(weak, ('D2', 'D1'))
    assert selection.days == ('D1', 'D2')
    assert selection.dataset_id == 'P/D1+D2'
    assert not selection.same_day
    assert selection.runs_dir() == weak.path / RUNS_DIR
    record = selection.record()
    assert record['mode'] == 'across_days'
    assert [d['day'] for d in record['days']] == ['D1', 'D2']
    assert {d['features'] for d in record['days']} == {'full'}
    assert selection.sha256() == hashlib.sha256(selection.record_bytes()).hexdigest()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - the whole record and its bytes against an expectation built from the project folder, not from the implementation (Plan B Task 1) | date: 2026-09-28]
    identity = weak.read()['identity']
    expected = {
        'schema_version': 1, 'kind': 'multiphy_selection', 'project': 'P',
        'group': identity['group'], 'probe': None, 'mode': 'across_days',
        'days': [{'day': day, 'date': date, 'features': 'full',
                  'features_id': hashlib.sha256(
                      (weak.day_dir(day) / 'features' / 'full' / 'manifest.json').read_bytes()).hexdigest()}
                 for day, date in (('D1', '2026-01-01'), ('D2', '2026-01-02'))],
        'excluded': [{'cellid': 'D1:4', 'day': 'D1', 'reason': WEAK_REASON}]}
    assert identity['group']
    assert record == expected
    expected_bytes = (json.dumps(expected, indent=2, sort_keys=True) + '\n').encode()
    assert selection.record_bytes() == expected_bytes
    assert selection.sha256() == hashlib.sha256(expected_bytes).hexdigest()
    # [AGENT EDIT END]


def test_across_days_dataset_pairs_every_cross_day_unit(weak):
    dataset = Selection(weak, ('D1', 'D2')).dataset()
    assert isinstance(dataset, AcrossDaysDataset)
    assert len(dataset.units) == 6
    assert len(dataset.pairs) == 9
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pairs are exactly every eligible D1 unit with every D2 unit, no duplicates (Plan B Task 1 fix round 1) | date: 2026-09-28]
    got = [frozenset((a, b)) for a, b in zip(dataset.pairs.cellid_a, dataset.pairs.cellid_b)]
    assert len(got) == len(set(got))
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: unit ids and pairs against the synthetic days' known clusters 0-2 (D1:4 weak, cluster 3 mua), not against the dataset under test (Plan B Task 1 fix round 1, a2) | date: 2026-09-28]
    d1 = ('D1:0', 'D1:1', 'D1:2')
    d2 = ('D2:0', 'D2:1', 'D2:2')
    assert sorted(dataset.units.cellid) == sorted(d1 + d2)
    assert dict(zip(dataset.units.cellid, dataset.units.session_id)) == {
        **{c: 'D1' for c in d1}, **{c: 'D2' for c in d2}}
    assert set(got) == {frozenset((a, b)) for a in d1 for b in d2}
    # [AGENT EDIT END]
    assert dataset.metadata['selection'] == 'across_days'
    assert dataset.metadata['input_manifest_sha256'] == Selection(weak, ('D1', 'D2')).sha256()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: day names, insertion order and dates disagree, so only a date sort passes (Plan B Task 1 fix round 1) | date: 2026-09-28]
def test_days_sort_by_date_not_name_or_order(tmp_path):
    project = _open(extracted_project(tmp_path, days=('C', 'A')))  # C on 2026-01-01, A on 2026-01-02
    sd = write_day(tmp_path / 'in', 'B', seed=2, date='2025-12-31')
    commands.init_day(project.root, 'P', day='B', sort=str(sd.sort_dir), date=sd.date,
                      events=str(sd.events_path), raw=str(sd.raw_path), uv_per_bit=sd.uv_per_bit)
    commands.extract(project.root, 'P', days=['B'], scratch_root=tmp_path / 'scratch')
    assert default_days(project) == ('B', 'C', 'A')
    assert Selection(project, ('A', 'B', 'C')).days == ('B', 'C', 'A')
    assert [d['day'] for d in Selection(project, ('A', 'C')).record()['days']] == ['C', 'A']


def test_dataset_hash_and_metadata_come_from_one_project_read(weak, monkeypatch):
    selection = Selection(weak, ('D1', 'D2'))
    expected = selection.sha256()
    reads = []
    real_read = weak.read

    def read():
        reads.append(1)
        manifest = real_read()
        if len(reads) > 1:  # a later read sees a project changed after the first
            manifest['identity']['probe'] = 'NP2'
        return manifest

    monkeypatch.setattr(weak, 'read', read)
    for chosen in (selection, Selection(weak, ('D1',))):
        reads.clear()
        dataset = chosen.dataset()
        assert len(reads) == 1
        if chosen is selection:
            assert dataset.metadata['probe_type'] is None
            assert dataset.metadata['input_manifest_sha256'] == expected


def test_same_day_timing_does_not_depend_on_eligible_units(weak, monkeypatch):
    from multiphy.project import selection as selection_module

    def refuse(*args, **kwargs):
        raise AssertionError('timing must not read the halves units')

    monkeypatch.setattr(selection_module, 'load_dataset', refuse)
    assert set(Selection(weak, ('D1',)).timing()) == {'recording0000_part000', 'recording0000_part001'}
# [AGENT EDIT END]


def test_across_days_bundle_shape_and_timing(weak):
    selection = Selection(weak, ('D1', 'D2'))
    bundle = selection.dataset().waveform_bundle()
    assert isinstance(bundle, RawWaveformBundle)
    assert bundle.waveforms.shape == (6, 82, 8, 2)
    assert set(selection.timing()) == {'D1', 'D2'}


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an across-days bundle names each day's source manifest hash as raw_source, so UnitMatch, DANT and DeepUnitMatch accept it (e2e fix) | date: 2026-10-11]
def test_across_days_bundle_passes_upstream_check(weak):
    from multiphy.backends._upstream import prepare_bundle

    dataset = Selection(weak, ('D1', 'D2')).dataset()
    bundle = dataset.waveform_bundle()
    expected = {day: hashlib.sha256((weak.day_dir(day) / 'source' / 'extraction_manifest.json').read_bytes()).hexdigest()
                for day in ('D1', 'D2')}
    assert bundle.provenance['raw_source'] == expected
    eligible = prepare_bundle(dataset.units, bundle)[0]
    assert len(eligible) == 6
    assert sorted(eligible.cellid) == ['D1:0', 'D1:1', 'D1:2', 'D2:0', 'D2:1', 'D2:2']
    bundle.provenance.pop('raw_source')
    with pytest.raises(ValueError, match='^Raw waveform provenance requires raw_source and split_method$'):
        prepare_bundle(dataset.units, bundle)
# [AGENT EDIT END]


def test_same_day_timing_keys_both_halves(weak):
    selection = Selection(weak, ('D1',))
    assert selection.same_day
    assert selection.dataset_id == 'P/D1'
    assert selection.runs_dir() == weak.day_dir('D1') / RUNS_DIR
    timing = selection.timing()
    assert set(timing) == {'recording0000_part000', 'recording0000_part001'}
    for value in timing.values():
        assert value.source_session == 'D1'
        assert len(value.stimuli) == 10


def test_ineligible_unit_excluded_and_listed(weak):
    across = Selection(weak, ('D1', 'D2'))
    assert across.record()['excluded'] == [{'cellid': 'D1:4', 'day': 'D1', 'reason': WEAK_REASON}]
    assert 'D1:4' not in set(across.dataset().units.cellid)
    same = Selection(weak, ('D1',))
    assert [e['reason'] for e in same.record()['excluded']] == [f'part 1: {WEAK_REASON}']
    assert same.record()['mode'] == 'same_day'
    assert same.record()['days'][0]['features'] == 'halves'


def test_selection_refusals(tmp_path):
    project = _open(extracted_project(tmp_path, tiers={'D2': 'templates'}))
    sd = write_day(tmp_path / 'in', 'D3', seed=2, date='2026-01-03')
    commands.init_day(project.root, 'P', day='D3', sort=str(sd.sort_dir), date=sd.date,
                      events=str(sd.events_path), raw=str(sd.raw_path), uv_per_bit=sd.uv_per_bit)
    for days in ((), ('D1', 'D1')):
        with pytest.raises(ValueError, match='^choose one day, or two or more distinct days$'):
            Selection(project, days)
    with pytest.raises(ValueError, match='^no day D9 in project P$'):
        Selection(project, ('D9',))
    with pytest.raises(ValueError, match='^day D3 has no features; run extract first$'):
        Selection(project, ('D1', 'D3'))
    with pytest.raises(ValueError) as refused:
        Selection(project, ('D1', 'D2'))
    assert str(refused.value) == ('day D2 is templates tier: no waveform halves, so no geometry for any backend; '
                                  'add the day with its raw file to a new project')


def test_selection_of_run_roundtrip_and_refusal(tmp_path):
    project = _open(extracted_project(tmp_path))
    selection = Selection(project, ('D1', 'D2'))
    run_dir = selection.runs_dir() / 'run1'
    run_dir.mkdir(parents=True)
    saved = selection.record_bytes()
    (run_dir / 'selection.json').write_bytes(saved)
    reopened = selection_of_run(run_dir)
    assert reopened.record_bytes() == saved
    assert reopened.days == ('D1', 'D2')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a changed features id gets the run-mismatch refusal (Plan B Task 1 fix round 1) | date: 2026-09-28]
    original = project.read()['days']['D1']['features']['full']
    project.update(lambda manifest: manifest['days']['D1']['features'].update(full='f' * 64))
    with pytest.raises(ValueError) as refused:
        selection_of_run(run_dir)
    assert str(refused.value) == (f'{run_dir.resolve()}/selection.json no longer matches project P; '
                                  'the project changed after this run')
    project.update(lambda manifest: manifest['days']['D1']['features'].update(full=original))
    assert selection_of_run(run_dir).record_bytes() == saved
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a cleared features id and a removed day get the same run-mismatch refusal (Plan B Task 1 review 3) | date: 2026-09-28]
    mismatch = (f'{run_dir.resolve()}/selection.json no longer matches project P; '
                'the project changed after this run')
    project.update(lambda manifest: manifest['days']['D1']['features'].update(full=None))
    with pytest.raises(ValueError) as refused:
        selection_of_run(run_dir)
    assert str(refused.value) == mismatch
    project.update(lambda manifest: manifest['days']['D1']['features'].update(full=original))
    removed = project.read()['days']['D2']
    project.update(lambda manifest: manifest['days'].pop('D2'))
    with pytest.raises(ValueError) as refused:
        selection_of_run(run_dir)
    assert str(refused.value) == mismatch
    project.update(lambda manifest: manifest['days'].update(D2=removed))
    assert selection_of_run(run_dir).record_bytes() == saved
    # [AGENT EDIT END]
    _set_probe(project, 'NP1')
    with pytest.raises(ValueError, match='no longer matches project P'):
        selection_of_run(run_dir)
    empty = selection.runs_dir() / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match='has no selection.json; it is not a project run'):
        selection_of_run(empty)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a folder outside any project with no selection.json gets the prescribed missing-selection refusal (Plan B Task 1 fix round 2) | date: 2026-09-28]
    outside = tmp_path / 'outside'
    outside.mkdir()
    with pytest.raises(FileNotFoundError) as refused_outside:
        selection_of_run(outside)
    assert str(refused_outside.value) == f'{outside.resolve()} has no selection.json; it is not a project run'
    # [AGENT EDIT END]


def test_probe_reaches_both_bundles(tmp_path):
    project = _open(extracted_project(tmp_path))
    _set_probe(project, 'NP1')
    across = Selection(project, ('D1', 'D2'))
    assert across.record()['probe'] == 'NP1'
    assert across.dataset().waveform_bundle().provenance['probe_type'] == 'NP1'
    same = Selection(project, ('D2',)).dataset()
    assert isinstance(same, BenchmarkDataset)
    assert same.metadata['selection'] == 'same_day'
    assert same.waveform_bundle().provenance['probe_type'] == 'NP1'


def test_across_days_subset_refuses_unknown(weak):
    dataset = Selection(weak, ('D1', 'D2')).dataset()
    for cellids in (['X:9'], []):
        with pytest.raises(ValueError, match='^subset needs known unit ids$'):
            dataset.subset(cellids)
    subset = dataset.subset(['D1:0', 'D2:1'])
    assert list(subset.units.cellid) == ['D1:0', 'D2:1']
    assert len(subset.pairs) == 1
    assert set(subset.geometry) == {'D1', 'D2'}
    assert set(dataset.subset(['D1:0', 'D1:1']).geometry) == {'D1'}


def test_conditions_per_day(tmp_path):
    project = _open(extracted_project(tmp_path, conditions={'D2': ('C', 'D')}))
    assert Selection(project, ('D1', 'D2')).conditions() == {'D1': frozenset({'A', 'B'}),
                                                             'D2': frozenset({'C', 'D'})}


def test_selection_signatures():
    hints = typing.get_type_hints
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: write_day takes events and extracted_project takes no_event_days, each as its last parameter (event-optional Task 2) | date: 2026-10-09]
    assert hints(write_day) == {
        'folder': Path, 'day': str, 'seed': int, 'date': str, 'weak_unit': bool, 'gzip_raw': bool,
        'with_templates': bool, 'conditions': tuple, 'events': bool, 'return': SyntheticDay}
    assert hints(extracted_project) == {
        'folder': Path, 'days': tuple, 'project': str, 'conditions': Optional[dict], 'tiers': Optional[dict],
        'weak_units': tuple, 'no_event_days': tuple, 'return': Path}
    assert list(inspect.signature(write_day).parameters)[-1] == 'events'
    assert list(inspect.signature(extracted_project).parameters)[-1] == 'no_event_days'
    # [AGENT EDIT END]
    assert hints(fingerprint_matrix) == {'rows': list, 'return': tuple}
    assert folder.RUNS_DIR == 'runs'
    assert "RUNS_DIR = " not in Path(commands.__file__).read_text()  # commands imports the one in folder.py
    assert hints(Selection) == {'project': Project, 'days': tuple}
    assert hints(Selection.same_day.fget) == {'return': bool}
    assert hints(Selection.dataset_id.fget) == {'return': str}
    assert hints(Selection.runs_dir) == {'return': Path}
    assert hints(Selection.record) == {'return': dict}
    assert hints(Selection.record_bytes) == {'return': bytes}
    assert hints(Selection.sha256) == {'return': str}
    assert hints(Selection.dataset) == {'return': Union[BenchmarkDataset, AcrossDaysDataset]}
    assert hints(Selection.timing) == {'return': dict}
    assert hints(Selection.conditions) == {'return': dict}
    assert hints(default_days) == {'project': Project, 'return': tuple}
    assert hints(selection_of_run) == {'run_dir': Path, 'return': Selection}
    assert hints(AcrossDaysDataset) == {'units': pd.DataFrame, 'pairs': pd.DataFrame, 'features': dict,
                                        'geometry': dict, 'metadata': dict}
    assert hints(AcrossDaysDataset.subset) == {'cellids': list, 'return': AcrossDaysDataset}
    assert hints(AcrossDaysDataset.waveform_bundle) == {'return': RawWaveformBundle}
    assert benchmark.fingerprint_matrix is fingerprint_matrix
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the signatures of selection_inputs and read_source_entry (launcher-gui Task 1) | date: 2026-10-07]
    assert hints(selection_inputs) == {'project': Project, 'days': tuple, 'return': frozenset}
    assert hints(read_source_entry) == {'source_dir': Path, 'return': dict}
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a selection's inputs predicted from its source manifests, checked against the built inputs (launcher-gui Task 1) | date: 2026-10-07]
EIGHT_INPUTS = frozenset({Input.UNITS, Input.CANDIDATE_PAIRS, Input.FEATURES, Input.WAVEFORM_HALVES,
                          Input.GEOMETRY, Input.SPIKE_TIMES, Input.TIMING, Input.FINGERPRINT})


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: D1 with trial events, D2 without; the timing and evidence of a selection that mixes them (event-optional Task 4) | date: 2026-10-09]
@pytest.fixture(scope='module')
def mixed(tmp_path_factory) -> Project:
    """D1 with trial events (conditions A and B); D2 written and added without events."""
    return _open(extracted_project(tmp_path_factory.mktemp('mixed'), no_event_days=('D2',)))


def test_selection_timing_skips_days_without_events(mixed):
    assert list(Selection(mixed, ('D1', 'D2')).timing()) == ['D1']
    assert Selection(mixed, ('D2',)).timing() == {}


def test_selection_evidence_mixed(mixed, tmp_path):
    evidence = Selection(mixed, ('D1', 'D2')).evidence()
    assert evidence.events is False
    assert evidence.reasons == ('no events · shared conditions per day pair: D1+D2 0',
                                'spont · whole recording · days without baseline windows: D2')
    shared = _open(extracted_project(tmp_path))
    assert Selection(shared, ('D1',)).evidence().shared == {'recording0000_part000+recording0000_part001': 2}
# [AGENT EDIT END]


def _built_inputs(selection: Selection) -> frozenset:
    """The intersection over the selection's groups of the inputs ``runner.build_input`` gives, less checkpoint."""
    dataset = runner._select_partition(selection.dataset(), 'all' if selection.same_day else None)
    timing = selection.timing()
    sets = [runner.build_input(dataset.subset(rows.cellid.tolist()), group,
                               extra_inputs={'timing': timing}).available() - {Input.CHECKPOINT}
            for group, rows in dataset.units.groupby('recording_group', sort=True)]
    assert sets
    return frozenset.intersection(*sets)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixed project adds a row with one event-free day and a row with only that day (event-optional Task 4) | date: 2026-10-09]
def test_selection_inputs_match_built_input(tmp_path, mixed):
    projects = {
        'shared': _open(extracted_project(tmp_path / 'shared')),
        'partial': _open(extracted_project(tmp_path / 'partial', conditions={'D2': ('B', 'C')})),
        'disjoint': _open(extracted_project(tmp_path / 'disjoint', conditions={'D2': ('C', 'D')})),
        'mixed': mixed,
    }
    withheld = {('disjoint', ('D1', 'D2')): {Input.FINGERPRINT},
                ('mixed', ('D1', 'D2')): {Input.FINGERPRINT},
                ('mixed', ('D2',)): {Input.TIMING, Input.FINGERPRINT}}
    for name, project in projects.items():
        for days in (('D1', 'D2'), ('D2',)):
            expected = EIGHT_INPUTS - withheld.get((name, days), set())
            # [AGENT EDIT END]
            predicted = selection_inputs(project, days)
            assert predicted == expected, (name, days)
            assert predicted == _built_inputs(Selection(project, days)), (name, days)


def test_selection_inputs_read_no_arrays(weak, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('selection_inputs must read no arrays')

    monkeypatch.setattr(np, 'load', refuse)
    assert selection_inputs(weak, ('D1', 'D2')) == EIGHT_INPUTS
    assert Selection(weak, ('D1', 'D2')).conditions() == {'D1': frozenset({'A', 'B'}), 'D2': frozenset({'A', 'B'})}


def test_read_source_entry_equals_read_source(weak, tmp_path):
    folder = weak.day_dir('D1') / 'source'
    entry = read_source_entry(folder)
    assert entry == read_source(folder)['entry']
    assert entry == json.loads((folder / 'extraction_manifest.json').read_text())['recordings'][0]

    sd = write_day(tmp_path / 'in', 'D1')
    legacy = tmp_path / 'legacy'
    write_legacy_inputs(sd, legacy, dataset_id='legacy', well='G')
    entry = read_source_entry(legacy)
    oracle = json.loads((legacy / 'extraction_manifest.json').read_text())['recordings'][0]
    assert {'partition', 'well'} <= set(oracle)
    assert entry == read_source(legacy)['entry']
    assert entry == {key: value for key, value in oracle.items() if key not in ('partition', 'well')}
    assert 'partition' not in entry and 'well' not in entry
# [AGENT EDIT END]
