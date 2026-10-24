import copy
import json
import platform
from pathlib import Path

import pandas as pd
import pytest

import multiphy.backends  # noqa: F401  (registers the built-ins)
from multiphy.backends.local import functional_likelihood
from multiphy.backends.local.functional_likelihood import run_functional_likelihood
from multiphy.contract import inputs, registry
from multiphy.contract.backend import RunContext
from multiphy.contract.inputs import Input, MatchingInput
from multiphy.contract.provenance import json_safe
from multiphy.contract.results import ScoreKind, TrackKind
from multiphy.evaluation import evaluate_result
import dataclasses
import itertools

import numpy as np

from multiphy.backends.local import evidence as ev
from multiphy.backends.local.examples import fit_ensemble, labeled_examples
from multiphy.backends.local.references import group_observations

from .synthetic_drift import ground_truth, make_drift_days


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # every test runs in its own tmp_path, so a stray relative write never lands in the repo
    monkeypatch.chdir(tmp_path)


def _run(days, **options):
    return run_functional_likelihood(days.units, days.features, days.geometry, days.timing, **options)


@pytest.fixture(scope='module')
def drift_run():
    days = make_drift_days(seed=0)
    return days, _run(days)


@pytest.fixture(scope='module')
def observed_group(drift_run):
    days, _ = drift_run
    (group,), _ = group_observations(days.units, days.features, days.geometry, days.timing)
    return group


@pytest.fixture(scope='module')
def references(observed_group):
    """``(same, diff)`` reference scores of an ensemble fitted here from the same days, outside the backend."""
    group = observed_group
    ensemble, same_ref, diff_ref = fit_ensemble(
        labeled_examples(group.observations, group.session_of, group.keys), ev.run_streams(True, waveform=True))
    return ensemble.reference_scores(same_ref), ensemble.reference_scores(diff_ref)


def _rows(result, entry):
    pairs = result.pairs
    return pairs[(pairs.siteid_a == entry['session_a']) & (pairs.siteid_b == entry['session_b'])]


def _rounded(values):
    values = np.asarray(values, dtype=float)
    return [round(float(x), 4) for x in values[np.isfinite(values)]]


def _matching_failure_context(result):
    """Compact failure evidence, not the large reference/candidate score arrays."""
    import scipy

    provenance = result.provenance.to_json()
    pairs = result.pairs
    details = {key: provenance.get(key) for key in (
        'score_column', 'threshold', 'ambiguity_margin', 'registration', 'skipped_groups',
        'reference_counts', 'ensemble_fits', 'input_pair_digest', 'run_config_digest')}
    details.update(
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                     scipy=scipy.__version__),
        decisions=pairs.decision.value_counts().to_dict(),
        score_summary=pairs.match_score.describe().to_dict(),
        ensemble_weights={group: {key: value for key, value in refs.items() if key != 'streams'}
                          for group, refs in provenance.get('mixture_references', {}).items()},
        session_pairs=[{**entry, 'streams': {
            stream: {key: value for key, value in fit.items() if key != 'observed'}
            for stream, fit in entry.get('streams', {}).items()}}
            for entry in provenance.get('session_pairs', [])])
    return json.dumps(json_safe(details), indent=2, sort_keys=True, allow_nan=False)


def test_functional_likelihood_is_registered_with_feature_inputs():
    spec = registry.get('functional_likelihood').spec
    assert spec.requires == {Input.UNITS, Input.FEATURES, Input.GEOMETRY}
    assert spec.optional == {Input.TIMING}
    assert spec.version == '3'
    assert spec.provides.score_kinds == {ScoreKind.UNCALIBRATED} and spec.provides.track_kinds == {TrackKind.COMMON}
    assert [o.name for o in spec.options] == ['min_repeats', 'min_spikes', 'n_splits', 'seed', 'radius_um',
                                              'min_pairs', 'registration', 'feature_stream', 'threshold',
                                              'contradiction_threshold', 'ambiguity_margin']
    options = {o.name: o for o in spec.options}
    assert options['min_pairs'].default == 30
    assert options['feature_stream'].choices == ('combined', 'stimulus', 'physiology', 'waveform')
    assert options['registration'].choices == ('matches', 'profile', 'none')
    assert options['registration'].default == 'matches'
    assert (options['min_repeats'].default, options['min_spikes'].default, options['n_splits'].default,
            options['seed'].default, options['radius_um'].default) == (5, 200, 20, 0, 80.0)


def test_real_days_allowed():
    spec = registry.get('functional_likelihood').spec
    assert spec.version == '3' and spec.real_days is True


def test_functional_likelihood_backend_matches_module_function(tmp_path):
    days = make_drift_days(n_neurons=20, seed=1)
    direct = _run(days, n_splits=2)

    spec = registry.get('functional_likelihood').spec
    inp = MatchingInput(days.units, group='g', features=days.features, geometry=days.geometry,
                        timing=days.timing).validate()
    result = registry.get('functional_likelihood').run(inp, dict(spec.defaults(), n_splits=2),
                                                       RunContext('r', tmp_path))

    assert result.tracks == direct.tracks and result.unmatched == direct.unmatched
    assert result.backend_id == 'functional_likelihood' and result.version == '3'
    assert result.provenance.get('evidence_display_name') == 'Functional likelihood'
    assert result.provenance.get('legacy_shared_input') is None and result.provenance.get('session_pairs') is not None
    assert result.validate(spec)


def test_functional_likelihood_recovers_synthetic_tracks(drift_run):
    days, result = drift_run
    scores = evaluate_result(result, ground_truth(days), expected_partition=None)
    assert scores['pair_precision'] >= 0.9, scores
    assert scores['pair_recall'] >= 0.6, scores


def test_registration_beats_none_on_drift():
    days = make_drift_days(offsets_um=(0.0, 100.0, -100.0))
    truth = ground_truth(days)
    recall = {method: evaluate_result(_run(days, registration=method), truth, expected_partition=None)['pair_recall']
              for method in ('matches', 'none')}
    assert recall['matches'] - recall['none'] >= 0.3, recall


def test_functional_likelihood_tracks_one_per_day():
    days = make_drift_days(seed=0)
    original = days.units.cellid[days.units.session_id == 'day1'].iloc[0]
    clone = original + '-clone'
    units = pd.concat([days.units, days.units[days.units.cellid == original].assign(cellid=clone)],
                      ignore_index=True)
    features = dict(days.features, **{clone: copy.deepcopy(days.features[original])})
    result = run_functional_likelihood(units, features, days.geometry, days.timing, ambiguity_margin=0.0)
    session_of = dict(zip(units.cellid, units.session_id))
    assert result.tracks, _matching_failure_context(result)
    assert all(len({session_of[c] for c in track}) == len(track) for track in result.tracks)
    twins = result.pairs[result.pairs.cellid_a.isin([original, clone]) | result.pairs.cellid_b.isin([original, clone])]
    assert (twins.decision == 'selected').any()
    assert (twins.decision == 'session_collision').any()
    assert sum(any(c in track for c in (original, clone)) for track in result.tracks) == 1


def test_mixture_fits_every_gated_candidate(drift_run, references, observed_group):
    days, result = drift_run
    entries = result.provenance.get('session_pairs')
    n_units = days.units.session_id.value_counts().to_dict()
    assert entries and all(e['n_mixture_observations'] == e['n_candidates'] for e in entries)
    assert any(e['n_candidates'] > n_units[e['session_a']] for e in entries)
    group, gate_um = observed_group, ev.Criteria().gate_um
    offsets = result.provenance.get('registration')['offsets_um']
    for entry in entries:
        sa, sb = entry['session_a'], entry['session_b']
        rel = offsets[sb] - offsets[sa]
        expected = {(a, b) for a, x in group.observations.items() if group.session_of[a] == sa
                    for b, y in group.observations.items() if group.session_of[b] == sb
                    if abs(x['depth'] - (y['depth'] + rel)) <= gate_um}
        rows = _rows(result, entry)
        assert expected and set(zip(rows.cellid_a, rows.cellid_b)) == expected and len(rows) == len(expected)
    same, diff = references
    fitted, maxima_differ = 0, False
    for entry in entries:
        rows = _rows(result, entry)
        assert len(rows) == entry['n_candidates']
        for stream in ('stim', 'phys', 'wave', 'global'):
            saved, col = entry['streams'][stream], 's_' + stream
            if 'pi' not in saved:
                continue
            every_row = ev.Mixture(same[col], diff[col]).fit(rows[col].to_numpy(dtype=float))
            assert saved['pi'] == pytest.approx(every_row.pi, abs=1e-9)
            np.testing.assert_allclose(rows['p_' + stream].to_numpy(dtype=float),
                                       every_row.posterior(rows[col].to_numpy(dtype=float)), atol=1e-9)
            maxima = rows.groupby('cellid_a')[col].max().to_numpy(dtype=float)
            per_unit = ev.Mixture(same[col], diff[col]).fit(maxima)
            maxima_differ |= not np.isfinite(per_unit.pi) or abs(per_unit.pi - saved['pi']) > 0.01
            fitted += 1
    assert fitted and maxima_differ


def test_mixture_references_in_provenance(drift_run, references):
    _, result = drift_run
    saved_references = result.provenance.get('mixture_references')
    entries = [e for e in result.provenance.get('session_pairs') if 'streams' in e]
    assert entries
    same, diff = references
    for entry in entries:
        for stream in ('stim', 'phys', 'wave', 'global'):
            assert len(entry['streams'][stream]['observed']) == entry['n_candidates']
            pile = saved_references[entry['recording_group']]['streams'][stream]
            assert len(pile['same']) >= 5 and len(pile['diff']) >= 5
            col = 's_' + stream
            assert entry['streams'][stream]['observed'] == _rounded(_rows(result, entry)[col])
            assert pile['same'] == _rounded(same[col]) and pile['diff'] == _rounded(diff[col])
    json.dumps(result.provenance.get('mixture_references'))
    json.dumps(result.provenance.get('session_pairs'))


def test_ensemble_fit_diagnostics_in_provenance(drift_run):
    _, result = drift_run
    fits = result.provenance.get('ensemble_fits')
    assert set(fits) == set(result.provenance.get('mixture_references'))
    for group in fits.values():
        assert set(group) == {'scoring', 'registration'}
        for fit in group.values():
            assert fit['attempted'] is True and fit['method'] == 'L-BFGS-B'
            assert isinstance(fit['success'], bool) and isinstance(fit['status'], int)
            assert isinstance(fit['iterations'], int) and fit['iterations'] >= 0
            assert isinstance(fit['message'], str)
            assert fit['objective'] is None or np.isfinite(fit['objective'])
    assert json.loads(json.dumps(fits, allow_nan=False)) == fits
    context = json.loads(_matching_failure_context(result))
    assert context['ensemble_fits'] == fits
    assert sum(context['decisions'].values()) == len(result.pairs)
    assert all('observed' not in fit for entry in context['session_pairs'] for fit in entry['streams'].values())
    assert all('streams' not in entry for entry in context['ensemble_weights'].values())


@pytest.mark.parametrize('empty', [False, True])
def test_matching_failure_context_handles_unavailable_scores(empty):
    from multiphy.backends.local.tracker import result_from_pairs

    units = pd.DataFrame(dict(cellid=['a', 'b'], session_id=['A', 'B'], recording_group='g', su_type='SU'))
    pairs = pd.DataFrame(dict(cellid_a=['a'], cellid_b=['b'], match_score=[np.nan]))
    result = result_from_pairs('diagnostic', '1', units, pairs.iloc[:0] if empty else pairs)
    context = json.loads(_matching_failure_context(result))
    assert not result.tracks
    assert context['score_summary']['count'] == 0 and context['score_summary']['min'] is None
    assert context['decisions'] == ({} if empty else {'missing_evidence': 1})
    assert context['ensemble_weights'] == {} and context['session_pairs'] == []


def test_offsets_in_provenance(drift_run):
    days, result = drift_run
    assert set(result.provenance.get('registration')['offsets_um']) == {'day0', 'day1', 'day2'}
    # the injected shifts are 40, 60 and 100 µm; 10 µm is half the 20 µm probe row pitch, the
    # bound the Task 6 match-refinement tests hold registration to
    offsets = result.provenance.get('registration')['offsets_um']
    truth = days.truth_offsets
    errors = {(a, b): abs((offsets[b] - offsets[a]) - (truth[b] - truth[a]))
              for a, b in itertools.combinations(sorted(truth), 2)}
    assert max(errors.values()) <= 10.0, errors


def test_different_bin_widths_are_refused():
    days = make_drift_days(n_neurons=12, seed=2)
    features = {c: dict(f, fingerprint_bin_seconds=2 * f['fingerprint_bin_seconds']) if c.startswith('day2') else f
                for c, f in days.features.items()}
    with pytest.raises(ValueError, match='bin widths'):
        run_functional_likelihood(days.units, features, days.geometry, days.timing, n_splits=2)


def test_different_condition_lengths_are_refused():
    days = make_drift_days(n_neurons=12, seed=2)
    short = days.timing['day2']
    stimuli = tuple(dict(s, stop=s['start'] + (s['stop'] - s['start']) // 2) for s in short.stimuli)
    timing = dict(days.timing, day2=dataclasses.replace(short, stimuli=stimuli).validate())
    with pytest.raises(ValueError, match='fingerprint lengths'):
        run_functional_likelihood(days.units, days.features, days.geometry, timing, n_splits=2)


def test_day_without_kept_units_is_named():
    days = make_drift_days(n_neurons=20, seed=1)
    features = {c: dict(f, spike_samples=f['spike_samples'][:10]) if c.startswith('day2') else f
                for c, f in days.features.items()}
    result = run_functional_likelihood(days.units, features, days.geometry, days.timing, n_splits=2)
    named = [s for s in result.provenance.get('skipped_groups') if s.get('session') == 'day2']
    assert len(named) == 1 and named[0]['group'] == 'synthetic' and '0 of' in named[0]['reason']
    assert {(e['session_a'], e['session_b']) for e in result.provenance.get('session_pairs')} == {('day0', 'day1')}


def test_spacing_multiplicities_must_match():
    days = make_drift_days(n_neurons=12, seed=2)

    def raised(geometry, rows, step):
        geometry = np.array(geometry, dtype=float)
        tail = np.isin(geometry[:, 1], np.unique(geometry[:, 1])[-rows:])
        geometry[tail, 1] += step
        return geometry

    # spacings [20 x 22, 40] versus [20 x 21, 40, 40]: the same set of values, different multiplicities
    one_gap = {s: raised(g, 1, 20.0) for s, g in days.geometry.items()}
    two_gaps = dict(one_gap, day2=raised(raised(days.geometry['day2'], 2, 20.0), 1, 20.0))
    assert {tuple(np.unique(np.diff(np.unique(g[:, 1])))) for g in two_gaps.values()} == {(20.0, 40.0)}
    group_observations(days.units, days.features, one_gap, days.timing, n_splits=2)
    with pytest.raises(ValueError, match='same channel columns and spacing'):
        group_observations(days.units, days.features, two_gaps, days.timing, n_splits=2)


def test_depth_offsets_is_an_unknown_option():
    assert 'depth_offsets' not in {o.name for o in registry.get('functional_likelihood').spec.options}
    with pytest.raises(TypeError):
        run_functional_likelihood(pd.DataFrame(), {}, {}, {}, depth_offsets={'A': 0.0})


def test_old_ids_unknown(tmp_path):
    for old in ('constrained', 'legacy_fresh'):
        with pytest.raises(KeyError):
            registry.get(old)


def test_trial_timing_name(tmp_path):
    assert not hasattr(inputs, 'LegacyTiming')
    assert hasattr(inputs, 'TrialTiming')


def _anchored_errors(offsets: dict, truth: dict, anchor: str = 'day0') -> dict:
    """Each day's offset error once both the run's offsets and the true ones put ``anchor`` at 0."""
    return {s: abs((offsets[s] - offsets[anchor]) - (truth[s] - truth[anchor])) for s in truth}


def test_no_event_drift_recovered():
    days = make_drift_days(n_neurons=20, seed=1, events=False)
    result = _run(days)
    errors = _anchored_errors(result.provenance.get('registration')['offsets_um'], days.truth_offsets)
    assert max(errors.values()) <= 1.0, errors
    scores = evaluate_result(result, ground_truth(days), expected_partition=None)
    assert scores['pair_recall'] >= 0.85, scores
    assert scores['pair_precision'] >= 0.95, scores


def test_event_run_keeps_stim_registration(monkeypatch):
    real, registrations = functional_likelihood.functional_offsets, []

    def spy(obs_by_session, ensemble, diff_ref, keys, **kwargs):
        registrations.append((list(ensemble.streams), kwargs.get('aligned', False)))
        return real(obs_by_session, ensemble, diff_ref, keys, **kwargs)

    monkeypatch.setattr(functional_likelihood, 'functional_offsets', spy)
    days = make_drift_days(seed=0)
    result = _run(days)
    assert registrations == [(['stim', 'phys'], False)]
    errors = _anchored_errors(result.provenance.get('registration')['offsets_um'], days.truth_offsets)
    assert max(errors.values()) <= 10.0, errors


def test_stimulus_stream_refused_without_events():
    days = make_drift_days(n_neurons=20, seed=1, events=False)
    with pytest.raises(ValueError) as caught:
        _run(days, feature_stream='stimulus')
    assert str(caught.value) == ("feature_stream 'stimulus' needs events; group synthetic: no events · shared "
                                 'conditions per day pair: day0+day1 0 · day0+day2 0 · day1+day2 0')


def test_waveform_stream_choice():
    days = make_drift_days(n_neurons=20, seed=1)
    result = _run(days, n_splits=2, feature_stream='waveform')
    assert result.provenance.get('score_column') == 'p_wave'
    pairs = result.pairs
    assert pairs.p_wave.notna().any()
    np.testing.assert_array_equal(pairs.match_score.to_numpy(dtype=float), pairs.p_wave.to_numpy(dtype=float))


def test_fl_refuses_units_without_halves(monkeypatch):
    days = make_drift_days(n_neurons=12, seed=2)
    cellid = days.units.cellid.iloc[0]
    features = dict(days.features, **{cellid: {k: v for k, v in days.features[cellid].items()
                                               if k != 'waveform_halves'}})

    def observed(*args, **kwargs):
        raise AssertionError('units were observed before the waveform-halves refusal')

    monkeypatch.setattr(functional_likelihood, 'group_observations', observed)
    with pytest.raises(ValueError, match='^functional_likelihood needs waveform halves'):
        run_functional_likelihood(days.units, features, days.geometry, days.timing, n_splits=2)


def test_fl_chunk_seconds_only_when_a_group_observed_chunks():
    days = make_drift_days(seed=0)
    lone = make_drift_days(n_neurons=6, seed=3, events=False)
    lone_units = lone.units[lone.units.session_id == 'day0']
    rename = {c: f'lone-{c}' for c in lone_units.cellid}
    lone_units = lone_units.assign(cellid=lone_units.cellid.map(rename), session_id='lone0', recording_group='lone')
    units = pd.concat([days.units, lone_units], ignore_index=True)
    features = {**days.features, **{rename[c]: lone.features[c] for c in rename}}
    geometry = {**days.geometry, 'lone0': lone.geometry['day0']}
    provenance = run_functional_likelihood(units, features, geometry, days.timing, n_splits=2).provenance.to_json()
    evidence = provenance['run_evidence']
    assert (evidence['synthetic']['events'], evidence['synthetic']['spont']) == (True, 'baseline')
    assert evidence['lone']['events'] is False
    assert dict(group='lone', reason='1 session with single units; matching needs at least 2') in \
        provenance['skipped_groups']
    assert 'chunk_seconds' not in provenance
