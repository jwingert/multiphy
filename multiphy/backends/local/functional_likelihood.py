# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: functional_likelihood v2: project-wide references, match-driven registration before the proximity gate, a mixture on every gated candidate (functional-likelihood Task 7) | date: 2026-09-27]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names event-response likelihood ratios (launcher-gui Task 10) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the run's evidence, the waveform stream and the registration each run uses (event-optional Task 9) | date: 2026-10-10]
"""Backend ``functional_likelihood``: firing, waveform and optional event-response likelihood ratios.

Per recording group (one project): ``references.evidence_by_group`` decides once what the
group's days measured (``run_evidence``): events when every day has trial events and a
condition shared by every day, else a no-event run. Every single unit of every day is observed
on its day's resampled splits (``references.group_observations``) of its trials in an event run,
else of 5 s chunks of its window. The same/different reference pile comes from every day at once
(``examples.labeled_examples``: each unit against its own other half, its waveform half 1
jittered, and same-day neighbours within ``radius_um``) and fits one ``ev.Ensemble`` on the run's
streams (``ev.run_streams``: the event response only with events, physiology, waveform). Each day
then gets a rigid depth offset (``registration.functional_offsets``): an event run registers on
event response + physiology, unaligned; a no-event run on physiology + waveform, with day B
shifted by each offset tried. For each day pair, every unit pair within the proximity gate on
the registered depths is a candidate, scored half against half with day B's waveform aligned; a
per-stream ``ev.Mixture`` fits its mixing fraction on every candidate and gives the posterior.
Tracks come from ``tracker.assign`` (mutual margins, one unit per session, measured
contradictions). Trial timing, where a day has it, arrives as a ``TrialTiming``; this module
reads no lab format.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import evidence as ev
from ...contract.backend import COMMON_TRACKER_OPTIONS, Availability, BackendSpec, Option, Provides
from ...contract.inputs import Input
from ...contract.results import MatchResult, ScoreKind, TrackKind, single_units
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run decides its evidence per group, refuses units without waveform halves and records the jitter, pair minimum and chunk length (event-optional Task 9) | date: 2026-10-10]
from .examples import MIN_NEIGHBOUR_PAIRS, WAVEFORM_JITTER_UM, fit_ensemble, labeled_examples, pair_properties
from .references import CHUNK_SECONDS, evidence_by_group, group_observations, require_waveform_halves
# [AGENT EDIT END]
from .registration import functional_offsets
from .tracker import result_from_pairs, score_column_for

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams come from each run's evidence; feature_stream gains the waveform posterior (event-optional Task 9) | date: 2026-10-10]
FEATURE_STREAMS = ('combined', 'stimulus', 'physiology', 'waveform')
# [AGENT EDIT END]
NULL_QUANTILE = 0.99
REGISTRATION_METHODS = ('matches', 'profile', 'none')


def _rounded(values) -> list:
    values = np.asarray(values, dtype=float)
    return [round(float(x), 4) for x in values[np.isfinite(values)]]


def _with_psth(observation: dict, keys) -> dict:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: without shared conditions the observation has no psth (event-optional Task 5) | date: 2026-10-09]
    if not len(keys):
        return observation
    # [AGENT EDIT END]
    return dict(observation, psth=np.concatenate([observation['fingerprints'][k] for k in keys]))


def _candidates(obs_a: dict, obs_b: dict, keys, ensemble, crit, rel: float) -> pd.DataFrame:
    """Every unit pair of two days inside the proximity gate on registered depths, scored half against half."""
    rows = []
    for a, x in obs_a.items():
        full_a = _with_psth(x, keys)
        for b, y in obs_b.items():
            evidence = ev.pair_evidence(full_a, _with_psth(y, keys), offset_um=rel)
            gates = ev.gates_passed(evidence, crit)
            if not gates.get('prox', True):
                continue
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the waveform properties read day B's waveform at the registered offset; the other properties ignore it (event-optional Task 9, D9) | date: 2026-10-10]
            properties = pair_properties(x, y, keys, 0, offset_um=rel)
            # [AGENT EDIT END]
            scores = ensemble.score(properties)
            # the row keeps the evidence, then the half-versus-half properties that were scored replace
            # the full-versus-full values of the same name
            rows.append({'cellid_a': a, 'cellid_b': b, **evidence, **properties,
                         **{k: float(np.ravel(v)[0]) for k, v in scores.items()},
                         'gates_passed': ','.join(k for k, v in gates.items() if v),
                         'passes_all': bool(all(gates.values())), **{'gate_' + k: v for k, v in gates.items()}})
    return pd.DataFrame(rows)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixture fits the run's streams and the global score, not two fixed streams (event-optional Task 9) | date: 2026-10-10]
def _mixture(table: pd.DataFrame, same_scores: dict, diff_scores: dict, names) -> dict:
    """Fit each of ``names`` (the run's streams, then ``'global'``) on every candidate; write
    ``p_<stream>`` into ``table``; return the streams entry."""
    streams = {}
    for stream in names:
# [AGENT EDIT END]
        col = 's_' + stream
        observed = table[col].to_numpy(dtype=float) if col in table else np.array([])
        entry = dict(observed=_rounded(observed))
        n_same = int(np.isfinite(np.asarray(same_scores[col], dtype=float)).sum())
        n_diff = int(np.isfinite(np.asarray(diff_scores[col], dtype=float)).sum())
        try:
            mixture = ev.Mixture(same_scores[col], diff_scores[col]).fit(observed)
        except (ValueError, np.linalg.LinAlgError) as exc:
            table['p_' + stream] = np.nan
            entry.update(unavailable=f'{type(exc).__name__}: {exc}',
                         reason='the mixture fit failed on these candidates; no substitute score')
            streams[stream] = entry
            continue
        if not np.isfinite(mixture.pi):
            table['p_' + stream] = np.nan
            entry.update(unavailable='mixture not fitted',
                         reason=(f'{int(np.isfinite(observed).sum())} finite candidate scores, {n_same} same and '
                                 f'{n_diff} different reference scores; the mixture needs at least 8 candidates '
                                 'and 5 of each reference, and a density on each'))
            streams[stream] = entry
            continue
        table['p_' + stream] = mixture.posterior(observed)
        entry.update(pi=mixture.pi, pi_lo=mixture.pi_lo, pi_hi=mixture.pi_hi)
        streams[stream] = entry
    return streams


def _score_session_pair(obs_a: dict, obs_b: dict, keys, ensemble, crit, rel: float, same_scores: dict,
                        diff_scores: dict) -> tuple:
    """``(table, streams)``: one session pair's gated candidates with their posteriors, and its mixture entry."""
    table = _candidates(obs_a, obs_b, keys, ensemble, crit, rel)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixture fits the ensemble's own streams and the global score (event-optional Task 9) | date: 2026-10-10]
    return table, _mixture(table, same_scores, diff_scores, [*ensemble.streams, 'global'])
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each group's reference split is named from its evidence: trials or 5 s chunks, baseline windows or chunks for spontaneous rate (event-optional Task 9) | date: 2026-10-10]
def _reference_split(e) -> str:
    """How a group with evidence ``e`` split its references: its trials or 5 s chunks, and its spont windows."""
    chunks = f'{CHUNK_SECONDS:g} s chunks'
    if e.events:
        spont = 'baseline windows' if e.spont == 'baseline' else f'{chunks} of the recording window'
        return (f'Seeded trial splits: whole trials for ISI/ACG and rate, {spont} for spontaneous '
                'rate, repetitions per exact stimulus; split 0 alternates')
    spont = 'baseline windows' if e.spont == 'baseline' else 'the chunks'
    return (f'Seeded splits of {chunks} of the recording window: whole chunks for ISI/ACG and rate, '
            f'{spont} for spontaneous rate; no event responses; split 0 alternates')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run decides its evidence per group before any work, fits the run's streams with the waveform, registers on event response + physiology with events and on aligned physiology + waveform without, and records what it measured (event-optional Task 9) | date: 2026-10-10]
def run_functional_likelihood(units, features, geometry, timing_by_session, *, threshold=.9,
                              contradiction_threshold=.1, ambiguity_margin=.02, min_repeats=5, min_spikes=200,
                              n_splits=20, seed=0, radius_um=80.0, min_pairs=MIN_NEIGHBOUR_PAIRS,
                              registration='matches', feature_stream='combined', criteria=None, provenance=None,
                              run_id=None) -> MatchResult:
    """Match within each ``recording_group`` from project-wide references and registered depths.

    ``timing_by_session`` maps matcher session ids to verified ``TrialTiming``; a day it lacks
    has no events (``None``: no day has timing). Geometry is measured channel order. Every
    explicit single unit needs waveform halves. Uses no source-cluster truth labels.
    """
    if feature_stream not in FEATURE_STREAMS:
        raise ValueError(f'feature_stream is {feature_stream!r}; choose one of {FEATURE_STREAMS}')
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 1:
        raise ValueError(f'min_pairs must be an int >= 1, not {min_pairs!r}')
    if registration not in REGISTRATION_METHODS:
        raise ValueError(f'registration is {registration!r}; choose one of {REGISTRATION_METHODS}')
    require_waveform_halves(units, features, 'functional_likelihood')
    timing_by_session = timing_by_session or {}
    evidence = evidence_by_group(units, features, timing_by_session, min_repeats=min_repeats)
    if feature_stream == 'stimulus':
        for group, e in evidence.items():
            if not e.events:
                raise ValueError(f"feature_stream 'stimulus' needs events; group {group}: {e.reasons[0]}")
    crit = criteria or ev.Criteria()
    groups, skipped = group_observations(units, features, geometry, timing_by_session, min_repeats=min_repeats,
                                         min_spikes=min_spikes, min_reliability=crit.min_reliability,
                                         n_splits=n_splits, seed=seed, evidence=evidence)
    scored, session_pairs = [], []
    offsets_um, diagnostics, reference_counts, mixture_references = {}, {}, {}, {}
    for g in groups:
        e = evidence[g.group]
        streams = ev.run_streams(e.events, waveform=True)
        examples = labeled_examples(g.observations, g.session_of, g.keys, radius_um=radius_um, min_pairs=min_pairs,
                                    seed=seed)
        ensemble, same_ref, diff_ref = fit_ensemble(examples, streams)
        obs_by_session = {s: {c: o for c, o in g.observations.items() if g.session_of[c] == s} for s in g.sessions}
        if e.events:
            # an event run registers on the event response and physiology alone, unaligned
            registering, _, registering_diff = fit_ensemble(examples, ev.run_streams(True, waveform=False))
            offsets, diagnostics[g.group] = functional_offsets(obs_by_session, registering, registering_diff, g.keys,
                                                               method=registration, null_quantile=NULL_QUANTILE)
        else:
            # a no-event run registers on physiology and waveform, day B's waveform read at each offset tried
            offsets, diagnostics[g.group] = functional_offsets(obs_by_session, ensemble, diff_ref, g.keys,
                                                               method=registration, null_quantile=NULL_QUANTILE,
                                                               aligned=True)
        offsets_um.update({s: float(offsets[s]) for s in g.sessions})
        different = examples[~examples.same_unit.astype(bool)]
        reference_counts[g.group] = dict(
            same=int(examples.same_unit.astype(bool).sum()), different=len(different),
            neighbour_pairs=len(different[['cellid_a', 'cellid_b']].drop_duplicates()))
        same_scores, diff_scores = ensemble.reference_scores(same_ref), ensemble.reference_scores(diff_ref)
        mixture_references[g.group] = {
            **{f'weight_{s}': float(ensemble.weights[s]) for s in streams}, 'bias': float(ensemble.bias),
            'streams': {s: dict(same=_rounded(same_scores['s_' + s]), diff=_rounded(diff_scores['s_' + s]))
                        for s in [*streams, 'global']}}
        for sa, sb in itertools.combinations(g.sessions, 2):
            table, pair_streams = _score_session_pair(obs_by_session[sa], obs_by_session[sb], g.keys, ensemble, crit,
                                                      offsets[sb] - offsets[sa], same_scores, diff_scores)
            session_pairs.append(dict(recording_group=g.group, session_a=sa, session_b=sb,
                                      n_candidates=len(table), n_mixture_observations=len(table),
                                      streams=pair_streams))
            if not table.empty:
                table['recording_group'], table['siteid_a'], table['siteid_b'] = g.group, sa, sb
                scored.append(table)
    pairs = (pd.concat(scored, ignore_index=True) if scored
             else pd.DataFrame(columns=['cellid_a', 'cellid_b', 'p_global', 'p_stim', 'p_phys', 'p_wave']))
    eligible, _ = single_units(units)
    sessions = eligible[['session_id', 'recording_group']].drop_duplicates()
    read = [s for s, group in zip(sessions.session_id, sessions.recording_group) if evidence[str(group)].reads_timing]
    splits = {group: _reference_split(e) for group, e in evidence.items()}
    metadata = dict(provenance or {})
    metadata.update(
        evidence_display_name='Functional likelihood',
        implementation=("project-wide Ensemble on the run's streams from labeled split examples; registration on "
                        'event response + physiology with events, else on physiology + waveform at each offset; '
                        'Mixture on every gated candidate; common mutual-margin best-first tracker'),
        reference_split=(next(iter(splits.values())) if len(set(splits.values())) == 1
                         else '; '.join(f'{group}: {text}' for group, text in splits.items())),
        transductive_prior=('Each session pair fits its mixture prior on every gated candidate, including inference '
                            'observations; not held-out probability calibration'),
        run_evidence={group: e.to_dict() for group, e in evidence.items()},
        minimum_repeats=min_repeats, minimum_spikes=min_spikes, radius_um=radius_um, min_pairs=min_pairs,
        waveform_jitter_um=WAVEFORM_JITTER_UM, n_splits=n_splits, seed=seed,
        null_quantile=NULL_QUANTILE, criteria=crit.to_dict(),
        registration=dict(method=registration, offsets_um=offsets_um, diagnostics=diagnostics),
        reference_counts=reference_counts, mixture_references=mixture_references,
        session_pairs=session_pairs, skipped_groups=skipped,
        timing_provenance={session: timing_by_session[session].provenance for session in read})
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the chunk length is saved only when a group observed on reference windows used chunks; group_observations skips a group of fewer than 2 sessions before building any (event-optional Task 9, final review fix round 1) | date: 2026-10-10]
    n_sessions = sessions.recording_group.astype(str).value_counts()
    observed = [evidence[group] for group, n in n_sessions.items() if n >= 2]
    if any(not e.events or e.spont != 'baseline' for e in observed):
        metadata.update(chunk_seconds=CHUNK_SECONDS)
    # [AGENT EDIT END]
    column = score_column_for(pairs, feature_stream)
    metadata.update(feature_stream=feature_stream, calibration_status='provisional, not held-out calibrated',
                    evidence_columns=list(pairs.columns), used_optional_inputs=['timing'] if read else [])
    return result_from_pairs(SPEC.id, SPEC.version, units, pairs, score_column=column, threshold=threshold,
                             contradiction_threshold=contradiction_threshold, ambiguity_margin=ambiguity_margin,
                             provenance=metadata, run_id=run_id)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the functional_likelihood spec declares day_offsets (drift-view Task 2) | date: 2026-10-08]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: version 3: timing is optional, min_pairs is a run option, feature_stream adds the waveform posterior, help texts name optional events (event-optional Task 9) | date: 2026-10-10]
SPEC = BackendSpec('functional_likelihood', '3',
    requires=frozenset({Input.UNITS, Input.FEATURES, Input.GEOMETRY}), optional=frozenset({Input.TIMING}),
    provides=Provides(frozenset({ScoreKind.UNCALIBRATED}), frozenset({TrackKind.COMMON}), day_offsets=True),
    options=(Option('min_repeats', 'int', 5, 'Minimum condition repetitions for event references (event runs)'),
             Option('min_spikes', 'int', 200, 'Minimum spikes for a reference'),
             Option('n_splits', 'int', 20, 'Splits per day (trials, else 5 s chunks) for the same-unit references'),
             Option('seed', 'int', 0, 'Seed of the random splits and the same-unit waveform jitter'),
             Option('radius_um', 'float', 80.0, 'Same-day neighbours within this depth are different-unit references'),
             Option('min_pairs', 'int', MIN_NEIGHBOUR_PAIRS,
                    'Same-day neighbour pairs within radius_um needed to fit the references'),
             Option('registration', 'choice', 'matches', 'Depth registration before the proximity gate',
                    choices=REGISTRATION_METHODS),
             Option('feature_stream', 'choice', 'combined', 'Posterior to track on (stimulus needs events)',
                    choices=FEATURE_STREAMS),
             *COMMON_TRACKER_OPTIONS),
    builtin=True,
    real_days=True)
# [AGENT EDIT END]
# [AGENT EDIT END]


class FunctionalLikelihoodBackend:
    spec = SPEC

    def available(self):
        return Availability(True, SPEC.version, None)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: timing is optional, so a run without it passes no timing; min_pairs is passed (event-optional Task 9) | date: 2026-10-10]
    def run(self, inp, options, ctx):
        return run_functional_likelihood(
            inp.units, inp.features, inp.geometry, inp.timing or {}, threshold=options['threshold'],
            contradiction_threshold=options['contradiction_threshold'], ambiguity_margin=options['ambiguity_margin'],
            min_repeats=options['min_repeats'], min_spikes=options['min_spikes'], n_splits=options['n_splits'],
            seed=options['seed'], radius_um=options['radius_um'], min_pairs=options['min_pairs'],
            registration=options['registration'], feature_stream=options['feature_stream'],
            provenance=ctx.provenance.to_json(), run_id=ctx.run_id)
    # [AGENT EDIT END]
# [AGENT EDIT END]
