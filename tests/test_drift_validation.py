"""Tests of the drift validation harness: its outputs, its refusal of an existing folder, recovery
of an injected shift, the whole-row shift rule and recorded refusals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dataclasses
import json

from multiphy import drift_validation as dv
from multiphy.contract import registry
from multiphy.run.run_dir import load_result

from .synthetic_drift import ground_truth, make_drift_days

OUTPUT_FILES = ('summary.csv', 'calibration.csv', 'drift_recovery.png', 'calibration.png',
                'drift_recovery_legend.md', 'calibration_legend.md', 'METHODS.md', 'manifest.json')


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # a stray relative write lands in tmp_path, never in the repo
    monkeypatch.chdir(tmp_path)


def _dataset(name='synthetic', **kwargs) -> dv.DriftDataset:
    days = make_drift_days(**kwargs)
    return dv.DriftDataset(name, days.units, days.features, days.geometry, days.timing, ground_truth(days),
                           dict(days.truth_offsets))


@pytest.fixture(scope='module')
def small_run(tmp_path_factory):
    output = tmp_path_factory.mktemp('harness') / 'out'
    summary = dv.validate_datasets([_dataset(n_neurons=16, seed=0)], output, shifts_um=(0, 40),
                                   backends=('functional_likelihood', 'pair_logistic'),
                                   registrations=('matches', 'none'))
    return output, summary


def test_harness_writes_outputs(small_run):
    output, summary = small_run
    table = pd.read_csv(output / 'summary.csv')
    assert len(table) == 8
    assert set(table.columns) >= {'dataset', 'backend', 'registration', 'shift_um', 'status', 'reason',
                                  'offset_error_um', 'precision', 'recall', 'n_true_pairs', 'brier', 'n_scored_pairs'}
    assert sorted(set(zip(table.backend, table.registration, table.shift_um))) == sorted(
        (b, r, s) for b in ('functional_likelihood', 'pair_logistic') for r in ('matches', 'none') for s in (0, 40))
    for name in OUTPUT_FILES:
        assert (output / name).is_file(), name
    assert len(summary) == 8
    # no correction: the error is the largest true relative offset, 40 - (-60) = 100 µm at shift 0
    none0 = table[(table.registration == 'none') & (table.shift_um == 0)]
    np.testing.assert_allclose(none0.offset_error_um, 100.0)
    calibration = pd.read_csv(output / 'calibration.csv')
    assert len(calibration) == 8 * 10
    # the artifacts hold content: real PNGs, legends naming the dataset and every family, the methods and manifest
    for name in ('drift_recovery.png', 'calibration.png'):
        data = (output / name).read_bytes()
        assert data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) > 10_000, name
    for name in ('drift_recovery_legend.md', 'calibration_legend.md'):
        text = (output / name).read_text()
        assert 'nothing is pooled across datasets' in text, name
        for backend in ('functional_likelihood', 'pair_logistic'):
            assert f'synthetic · {backend}' in text, (name, backend)
    methods = (output / 'METHODS.md').read_text()
    assert '| synthetic | in memory (no path) |' in methods and 'radius_um = 80' in methods
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['inputs'][0]['shifted_session'] == 'day2'
    assert set(OUTPUT_FILES) <= set(manifest['files'])
    assert sum(f.endswith('result.json') for f in manifest['files']) == 8


def test_harness_refuses_existing_output(tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'keep.txt').write_text('mine')

    def no_work(*args, **kwargs):
        raise AssertionError('matcher work started before the output folder was refused')
    monkeypatch.setattr(registry, 'get', no_work)
    monkeypatch.setattr(dv, 'inject_shift', no_work)
    with pytest.raises(FileExistsError):
        dv.validate_datasets([_dataset(n_neurons=16, seed=0)], output, shifts_um=(0,))
    assert sorted(p.name for p in output.iterdir()) == ['keep.txt']
    assert (output / 'keep.txt').read_text() == 'mine'


def _capturing(monkeypatch, fail=lambda call: None):
    """Wrap every backend so its inputs are recorded; ``fail(call)`` may raise to refuse call ``call``."""
    real_get, calls = registry.get, []

    class Proxy:
        def __init__(self, backend):
            self.backend, self.spec = backend, backend.spec

        def run(self, inp, options, context):
            calls.append(dict(backend=self.spec.id, geometry={s: np.array(g) for s, g in inp.geometry.items()},
                              options=dict(options)))
            fail(len(calls) - 1)
            return self.backend.run(inp, options, context)
    monkeypatch.setattr(registry, 'get', lambda backend_id: Proxy(real_get(backend_id)))
    return calls


def test_harness_recovers_injected_shift(tmp_path, monkeypatch):
    dataset = _dataset(seed=0)
    calls = _capturing(monkeypatch)
    summary = dv.validate_datasets([dataset], tmp_path / 'out', shifts_um=(80,),
                                   backends=('functional_likelihood',), registrations=('matches',))
    row = summary.iloc[0]
    assert row.status == 'ok', row.reason
    assert row.offset_error_um <= 10
    # the matcher saw day2 (last by date) moved up 80 µm and the other days unmoved
    (call,) = calls
    assert call['options']['registration'] == 'matches'
    np.testing.assert_allclose(call['geometry']['day2'][:, 1], dataset.geometry['day2'][:, 1] + 80)
    for day in ('day0', 'day1'):
        np.testing.assert_allclose(call['geometry'][day], dataset.geometry[day])
    # independently: the saved offsets match the original truth with day2's offset lowered by 80 µm
    (path,) = (tmp_path / 'out' / 'runs').glob('*/result.json')
    offsets = load_result(path).provenance.get('registration')['offsets_um']
    truth = dict(dataset.truth_offsets, day2=dataset.truth_offsets['day2'] - 80.0)
    errors = [abs((offsets[b] - offsets[a]) - (truth[b] - truth[a]))
              for a, b in (('day0', 'day1'), ('day0', 'day2'), ('day1', 'day2'))]
    assert max(errors) <= 10
    np.testing.assert_allclose(row.offset_error_um, max(errors))


def test_inject_shift_needs_whole_rows():
    dataset = _dataset(n_neurons=4, seed=0)
    with pytest.raises(ValueError, match='20'):
        dv.inject_shift(dataset, 'day2', 30.0)
    shifted = dv.inject_shift(dataset, 'day2', 40.0)
    np.testing.assert_allclose(shifted.geometry['day2'][:, 1], dataset.geometry['day2'][:, 1] + 40)
    np.testing.assert_allclose(shifted.geometry['day2'][:, 0], dataset.geometry['day2'][:, 0])
    assert shifted.truth_offsets['day2'] == dataset.truth_offsets['day2'] - 40
    assert shifted.truth_offsets['day0'] == dataset.truth_offsets['day0']
    # the input is left unchanged
    assert dataset.geometry['day2'][:, 1].min() == 0
    with pytest.raises(ValueError, match='20'):
        dv.inject_shift(dataset, 'day2', float('inf'))
    # another probe: 15 µm rows take 45 but not 20 (the pitch is read from the geometry)
    fine = dataclasses.replace(dataset, geometry={s: g * np.array([1.0, 0.75]) for s, g in dataset.geometry.items()})
    with pytest.raises(ValueError, match='15'):
        dv.inject_shift(fine, 'day2', 20.0)
    np.testing.assert_allclose(dv.inject_shift(fine, 'day2', 45.0).geometry['day2'][:, 1],
                               fine.geometry['day2'][:, 1] + 45)


def test_refusal_recorded(tmp_path):
    summary = dv.validate_datasets([_dataset(n_neurons=16, seed=0)], tmp_path / 'out', shifts_um=(0,),
                                   backends=('functional_likelihood', 'pair_logistic'),
                                   registrations=('matches', 'none'), radius_um=1.0)
    assert len(summary) == 4
    assert sorted(zip(summary.backend, summary.registration)) == sorted(
        (b, r) for b in ('functional_likelihood', 'pair_logistic') for r in ('matches', 'none'))
    assert (summary.status == 'refused').all()
    assert summary.reason.str.contains('Rerun with a wider radius').all()
    assert (tmp_path / 'out' / 'summary.csv').is_file()
    calibration = pd.read_csv(tmp_path / 'out' / 'calibration.csv')
    assert len(calibration) == 4 * 10 and (calibration.n == 0).all()


def test_refusal_does_not_stop_later_runs(tmp_path, monkeypatch):
    """A matcher refusal and an evaluation refusal each give one refused row; the other runs still score."""
    import multiphy.evaluation as evaluation
    real_evaluate, evaluated = evaluation.evaluate_result, []

    def refuse_first(call):
        if call == 0:
            raise ValueError('injected matcher refusal')

    def evaluate(*args, **kwargs):
        evaluated.append(1)
        if len(evaluated) == 2:
            raise ValueError('injected evaluation refusal')
        return real_evaluate(*args, **kwargs)
    calls = _capturing(monkeypatch, refuse_first)
    monkeypatch.setattr(evaluation, 'evaluate_result', evaluate)
    summary = dv.validate_datasets([_dataset(n_neurons=16, seed=0)], tmp_path / 'out', shifts_um=(0, 40),
                                   backends=('functional_likelihood',), registrations=('matches', 'none'))
    assert len(calls) == 4
    assert list(summary.status) == ['refused', 'ok', 'refused', 'ok']
    assert summary.reason.iloc[0] == 'ValueError: injected matcher refusal'
    assert summary.reason.iloc[2] == 'ValueError: injected evaluation refusal'
    assert summary.recall.iloc[[1, 3]].notna().all() and summary.recall.iloc[[0, 2]].isna().all()
    saved = sorted(p.parent.name for p in (tmp_path / 'out' / 'runs').glob('*/result.json'))
    assert saved == sorted(f'synthetic__functional_likelihood__none__shift{s}' for s in (0, 40))
    calibration = pd.read_csv(tmp_path / 'out' / 'calibration.csv')
    assert len(calibration) == 4 * 10
    per_run = calibration.groupby(['shift_um', 'registration']).n.sum()
    assert per_run[(0, 'matches')] == 0 and per_run[(40, 'matches')] == 0
    assert per_run[(0, 'none')] > 0 and per_run[(40, 'none')] > 0


def test_sessions_follow_matcher_day_order():
    """The shifted session is the last by session_date (else first start_sample), not by name."""
    dataset = _dataset(n_neurons=4, seed=0)
    names = {'day0': 'z_early', 'day1': 'm_mid', 'day2': 'a_late'}
    units = dataset.units.assign(session_id=dataset.units.session_id.map(names))
    assert dv._sessions(dataclasses.replace(dataset, units=units)) == ['z_early', 'm_mid', 'a_late']
    starts = {'z_early': 100, 'm_mid': 200, 'a_late': 300}
    session_of = units.set_index('cellid').session_id
    features = {c: dict(f, start_sample=starts[session_of[c]]) for c, f in dataset.features.items()}
    undated = dataclasses.replace(dataset, units=units.drop(columns='session_date'), features=features)
    assert dv._sessions(undated) == ['z_early', 'm_mid', 'a_late']


def test_non_finite_shift_refused_and_manifest_written(tmp_path, monkeypatch):
    calls = _capturing(monkeypatch)
    output = tmp_path / 'out'
    summary = dv.validate_datasets([_dataset(n_neurons=16, seed=0)], output,
                                   shifts_um=(0, float('inf'), float('nan')),
                                   backends=('functional_likelihood',), registrations=('matches',))
    assert list(summary.status) == ['ok', 'refused', 'refused']
    assert summary.reason.iloc[1:].str.contains('not a whole number of rows').all()
    assert len(calls) == 1
    for name in OUTPUT_FILES:
        assert (output / name).is_file(), name
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['options']['shifts_um'] == [0.0, 'inf', 'nan']


def _two_groups() -> dv.DriftDataset:
    """Four days: day0/day1 in group g1, day2/day3 in group g2 (truth offsets -10, -50 | -10, 70 µm)."""
    dataset = _dataset(n_days=4, n_neurons=4, offsets_um=(0.0, 40.0, 0.0, -80.0), seed=0)
    groups = {'day0': 'g1', 'day1': 'g1', 'day2': 'g2', 'day3': 'g2'}
    return dataclasses.replace(dataset, units=dataset.units.assign(
        recording_group=dataset.units.session_id.map(groups)))


def test_offset_error_compares_sessions_within_a_recording_group():
    dataset = _two_groups()
    truth = dataset.truth_offsets
    assert truth == {'day0': -10.0, 'day1': -50.0, 'day2': -10.0, 'day3': 70.0}
    pairs = dv._session_pairs(dataset)
    assert pairs == [('day0', 'day1'), ('day2', 'day3')]
    # each group recovered exactly and centred on its own mean, as both backends report it
    recovered = {'day0': 20.0, 'day1': -20.0, 'day2': -40.0, 'day3': 40.0}
    assert dv._offset_error(recovered, truth, pairs, 'day3') == (0.0, '')
    # across groups the frames differ: day0 -> day2 would read |(-40 - 20) - 0| = 60 µm of false error
    assert abs((recovered['day2'] - recovered['day0']) - (truth['day2'] - truth['day0'])) == 60.0
    assert dv._offset_error(dict(recovered, day3=60.0), truth, pairs, 'day3') == (20.0, '')
    # a session missing from a compared pair is named; a session in no pair is not needed
    error, reason = dv._offset_error({k: v for k, v in recovered.items() if k != 'day1'}, truth, pairs, 'day3')
    assert np.isnan(error) and 'day1' in reason
    single = dataclasses.replace(dataset, units=dataset.units.assign(
        recording_group=dataset.units.session_id.map({'day0': 'g1', 'day1': 'g2', 'day2': 'g3', 'day3': 'g4'})))
    assert dv._session_pairs(single) == []
    error, reason = dv._offset_error(recovered, truth, [], 'day3')
    assert np.isnan(error) and 'recording_group' in reason


def test_drift_legend_counts_session_pairs_within_groups(tmp_path):
    output = tmp_path / 'out'
    dv.validate_datasets([_two_groups()], output, shifts_um=(0,), backends=('functional_likelihood',),
                         registrations=('matches',), radius_um=1.0)
    legend = (output / 'drift_recovery_legend.md').read_text()
    assert 'Session pairs compared per offset-error point:' in legend and 'share a `recording_group`' in legend
    assert 'synthetic · functional_likelihood: matches shift 0: refused' in legend
    assert 'share a `recording_group`' in (output / 'METHODS.md').read_text()

def _last_day_alone() -> dv.DriftDataset:
    """Three days: day0/day1 in group g1, the shifted last day day2 alone in g2."""
    dataset = _dataset(n_days=3, n_neurons=30, offsets_um=(0.0, 40.0, -60.0), seed=0)
    return dataclasses.replace(dataset, units=dataset.units.assign(
        recording_group=dataset.units.session_id.map({'day0': 'g1', 'day1': 'g1', 'day2': 'g2'})))


def test_offset_error_needs_the_shifted_session():
    dataset = _last_day_alone()
    pairs = dv._session_pairs(dataset)
    assert pairs == [('day0', 'day1')]
    shifted = dv.inject_shift(dataset, 'day2', 320.0)
    exact = {s: t for s, t in shifted.truth_offsets.items() if s != 'day2'}
    assert dv._offset_error(exact, shifted.truth_offsets, pairs, 'day0') == (0.0, '')
    error, reason = dv._offset_error(exact, shifted.truth_offsets, pairs, 'day2')
    assert np.isnan(error) and 'day2' in reason


def test_harness_does_not_report_recovery_of_an_unscored_shift(tmp_path):
    output = tmp_path / 'out'
    summary = dv.validate_datasets([_last_day_alone()], output, shifts_um=(40,),
                                   backends=('functional_likelihood', 'pair_logistic'), registrations=('matches',))
    assert len(summary) == 2
    assert (summary.status == 'ok').all()  # day0/day1 match; the only gap is day2's unmeasured shift
    assert summary.offset_error_um.isna().all()
    assert summary.reason.str.contains('shifted session day2').all()

def _fake_result(offsets: dict, groups: list):
    from types import SimpleNamespace
    return SimpleNamespace(provenance=dict(registration=dict(
        offsets_um=offsets, diagnostics={'g': dict(groups=groups)})))


def test_offset_error_compares_sessions_within_a_registration_component():
    truth = {'day0': 0.0, 'day1': 40.0, 'day2': 0.0, 'day3': -80.0}
    pairs = [(a, b) for i, a in enumerate(truth) for b in list(truth)[i + 1:]]
    # two components, each recovered exactly and centred on its own mean, as register_depths reports them
    offsets = {'day0': -20.0, 'day1': 20.0, 'day2': 40.0, 'day3': -40.0}
    split = _fake_result(offsets, [['day0', 'day1'], ['day2', 'day3']])
    error, reason = dv._registration_error(split, 'matches', truth, pairs, 'day3')
    assert error == 0.0
    assert "('day0', 'day2')" in reason and 'unmeasured' in reason
    # the recording_group rule alone would report the frame difference as error
    assert dv._offset_error(offsets, truth, pairs, 'day3')[0] == 60.0
    # an isolated shifted session is never scored, though it shares the recording_group
    isolated = _fake_result({'day0': -20.0, 'day1': 20.0, 'day2': 0.0, 'day3': 0.0},
                            [['day0', 'day1', 'day2'], ['day3']])
    error, reason = dv._registration_error(isolated, 'matches', truth, pairs, 'day3')
    assert np.isnan(error) and 'shifted session day3 was registered to no other session' in reason
    # one component: every pair is scored, no reason
    whole = _fake_result({'day0': 10.0, 'day1': 50.0, 'day2': 10.0, 'day3': -70.0},
                         [['day0', 'day1', 'day2', 'day3']])
    assert dv._registration_error(whole, 'matches', truth, pairs, 'day3') == (0.0, '')
    # no registration: zero offsets in one frame; singleton components do not hide the uncorrected drift
    none = _fake_result({s: 0.0 for s in truth}, [[s] for s in truth])
    assert dv._registration_error(none, 'none', truth, pairs, 'day3') == (120.0, '')


def test_harness_scores_offsets_within_registration_components(tmp_path, monkeypatch):
    """A run whose registration splits the days into two components is scored within each only."""
    dataset = _dataset(n_days=4, n_neurons=16, offsets_um=(0.0, 40.0, 0.0, -80.0), seed=0)
    real_get = registry.get

    class Split:
        def __init__(self, backend):
            self.backend, self.spec = backend, backend.spec

        def run(self, inp, options, context):
            result = self.backend.run(inp, options, context)
            registration = result.provenance.get('registration')
            (group, diag), = registration['diagnostics'].items()
            truth = dv.inject_shift(dataset, 'day3', 40.0).truth_offsets
            # recovered exactly, centred per component: day0/day1 and day2/day3 unlinked
            registration['offsets_um'] = {'day0': truth['day0'] - (truth['day0'] + truth['day1']) / 2,
                                          'day1': truth['day1'] - (truth['day0'] + truth['day1']) / 2,
                                          'day2': truth['day2'] - (truth['day2'] + truth['day3']) / 2,
                                          'day3': truth['day3'] - (truth['day2'] + truth['day3']) / 2}
            diag['groups'] = [['day0', 'day1'], ['day2', 'day3']]
            return result
    monkeypatch.setattr(registry, 'get', lambda backend_id: Split(real_get(backend_id)))
    summary = dv.validate_datasets([dataset], tmp_path / 'out', shifts_um=(40,),
                                   backends=('functional_likelihood',), registrations=('matches',))
    row = summary.iloc[0]
    assert row.status == 'ok'
    assert row.offset_error_um == pytest.approx(0.0)
    assert 'unmeasured' in row.reason and "('day1', 'day3')" in row.reason
    assert row.n_offset_pairs == 2
    legend = (tmp_path / 'out' / 'drift_recovery_legend.md').read_text()
    assert 'synthetic · functional_likelihood: matches shift 40: 2\n' in legend.split('Known same-unit')[0]
