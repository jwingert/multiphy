import importlib.util
import numpy as np
import pandas as pd
import pytest

from multiphy.contract import registry
from multiphy.contract.backend import RunContext
from multiphy.contract.errors import BackendUnavailable
from multiphy.contract.inputs import Input, MatchingInput, RawWaveformBundle, CheckpointRef
from multiphy.contract.results import ScoreKind, TrackKind


def test_external_specs_declare_inputs_and_emissions():
    s = {x.id: x for x in registry.list_specs()}
    assert s['unitmatch'].requires == {Input.UNITS, Input.WAVEFORM_HALVES, Input.GEOMETRY}
    assert s['dant'].requires == {Input.UNITS, Input.WAVEFORM_HALVES, Input.GEOMETRY, Input.SPIKE_TIMES}
    assert s['dant'].optional == {Input.FINGERPRINT}
    assert s['dant'].provides.score_kinds == {ScoreKind.NONE} and s['dant'].provides.similarity
    assert s['dant'].provides.track_kinds == {TrackKind.NATIVE}
    assert s['deepunitmatch'].requires == {Input.UNITS, Input.WAVEFORM_HALVES, Input.GEOMETRY, Input.CHECKPOINT}
    assert s['unitmatch'].provides.track_kinds == {TrackKind.NATIVE, TrackKind.COMMON}
    assert {o.name for o in s['dant'].options} == {'preset', 'settings', 'seed'}
    assert {'tracking', 'native_threshold', 'threshold', 'params'} <= {o.name for o in s['unitmatch'].options}
    assert {'checkpoint', 'checkpoint_probe', 'device'} <= {o.name for o in s['deepunitmatch'].options}


@pytest.mark.parametrize('backend_id,package', [('unitmatch', 'UnitMatchPy'), ('dant', 'pyDANT'), ('deepunitmatch', 'DeepUnitMatch')])
def test_availability_reflects_installed_packages(backend_id, package):
    avail = registry.get(backend_id).available()
    installed = importlib.util.find_spec(package) is not None
    assert avail.ok == installed
    if not installed:
        assert package in avail.reason


def test_unavailable_backend_raises_before_touching_data(tmp_path):
    backend = registry.get('dant')
    if backend.available().ok:
        pytest.skip('pyDANT installed; unavailability path not exercisable here')
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    inp = MatchingInput(units, group='g')
    with pytest.raises(BackendUnavailable, match='pyDANT'):
        backend.run(inp, backend.spec.defaults(), RunContext('r', tmp_path))


def test_deepunitmatch_probe_mismatch_is_a_contract_violation(tmp_path, monkeypatch):
    from multiphy.contract.errors import ContractViolation
    backend = registry.get('deepunitmatch')
    monkeypatch.setattr(backend, 'available', lambda: type('A', (), {'ok': True, 'version': 'x', 'reason': None})())
    ids = ('A-1', 'B-1')
    units = pd.DataFrame({'cellid': ids, 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    geom = np.column_stack([np.zeros(384), np.arange(384) * 10.])
    bundle = RawWaveformBundle(ids, np.zeros((2, 82, 384, 2)), ('A', 'B'), {'A': geom, 'B': geom}, (),
                               {'raw_source': 't', 'split_method': 'h', 'probe_type': 'np1'})
    ckpt = tmp_path / 'ckpt'; ckpt.write_bytes(b'x')
    inp = MatchingInput.from_bundle(units, bundle, group='g', checkpoint=CheckpointRef(str(ckpt), 'np2_4shank'))
    with pytest.raises(ContractViolation, match='probe'):
        backend.run(inp, {**backend.spec.defaults(), 'checkpoint': str(ckpt), 'checkpoint_probe': 'np2_4shank'},
                    RunContext('r', tmp_path))


def _resolver_score(n, values):
    score = np.full((n, n), np.nan)
    np.fill_diagonal(score, 1.0)
    for (i, j), v in values.items():
        score[i, j] = v
        score[j, i] = v
    return score


def _first_resolver_groups():
    from multiphy.backends.native_groups import resolve_groups
    score = np.array([[1, .99, .8, .5], [.99, 1, .7, .1], [.8, .7, 1, .8], [.5, .1, .8, 1]], dtype=float)
    return resolve_groups(['a', 'b', 'c', 'd'], ['s1', 's2', 's2', 's3'], [1, 1, 1, 1], score, [10, 10, 10, 10])


def test_resolver_keeps_best_mean_score():
    groups = _first_resolver_groups()
    assert groups.tracks == (('a', 'c', 'd'),)
    assert groups.proposed == frozenset({frozenset({'a', 'b'}), frozenset({'b', 'c'}), frozenset({'b', 'd'})})


@pytest.mark.parametrize('n_spikes,expected', [
    ([5, 50, 80], (('a', 'c'),)),
    ([5, 80, 80], (('a', 'b'),)),
    ([5, float('nan'), 1], (('a', 'c'),)),
])
def test_resolver_tie_goes_to_more_spikes_then_cellid(n_spikes, expected):
    from multiphy.backends.native_groups import resolve_groups
    score = np.array([[1, .9, .9], [.9, 1, .5], [.9, .5, 1]], dtype=float)
    groups = resolve_groups(['a', 'b', 'c'], ['s1', 's2', 's2'], [1, 1, 1], score, n_spikes)
    assert groups.tracks == expected


def test_resolver_one_session_group_is_all_proposed():
    from multiphy.backends.native_groups import resolve_groups
    score = np.array([[1, .9, .9], [.9, 1, .9], [.9, .9, 1]], dtype=float)
    groups = resolve_groups(['a', 'b', 'c'], ['s1', 's1', 's1'], [1, 1, 1], score, [1, 1, 1])
    assert groups.tracks == ()
    assert groups.proposed == frozenset({frozenset({'a', 'b'}), frozenset({'a', 'c'}), frozenset({'b', 'c'})})


def test_resolver_skips_missing_scores():
    from multiphy.backends.native_groups import resolve_groups
    ids = ['a', 'b', 'c', 'd', 'x', 'y', 'z']
    ix = {c: i for i, c in enumerate(ids)}
    score = _resolver_score(7, {(ix['b'], ix['d']): .3, (ix['c'], ix['a']): .25,
                                (ix['c'], ix['d']): .25, (ix['z'], ix['x']): .1})
    groups = resolve_groups(ids, ['s1', 's2', 's2', 's3', 's1', 's2', 's2'], [1, 1, 1, 1, 2, 2, 2], score,
                            [1, 1, 1, 1, 1, 500, 10])
    assert groups.tracks == (('a', 'b', 'd'), ('x', 'z'))


def test_resolver_rejects_mismatched_inputs():
    from multiphy.backends.native_groups import resolve_groups
    with pytest.raises(ValueError) as err:
        resolve_groups(['a', 'b'], ['s1'], [1, 1], np.eye(2), [1, 1])
    assert str(err.value) == 'resolve_groups inputs disagree in length'


def test_label_pairs_marks_kept_proposed_same_session():
    from multiphy.backends.native_groups import label_pairs
    groups = _first_resolver_groups()
    pairs = pd.DataFrame({'cellid_a': ['a', 'a', 'b', 'a', 'e'], 'cellid_b': ['c', 'b', 'c', 'e', 'f']})
    sessions = {'a': 's1', 'b': 's2', 'c': 's2', 'd': 's3', 'e': 's4', 'f': 's4'}
    out = label_pairs(pairs, sessions, groups, kept='native_track', unmatched='unmatched_by_native_tracking')
    assert len(out) == 5
    assert list(out.decision) == ['native_track', 'proposed_link', 'proposed_link',
                                  'unmatched_by_native_tracking', 'same_session']
    assert list(out.selected) == [True, False, False, False, False]
    assert 'decision' not in pairs.columns
