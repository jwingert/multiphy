# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic backend - a calibrated pair classifier trained on its own project (functional-likelihood Task 8) | date: 2026-09-27]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic holds out sessions, not only days (Follow-up Task 4) | date: 2026-09-29]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: module docstring names model_properties(waveform_channels) as the features, peak-threshold pair by default (threshold neighbourhood) | date: 2026-10-04]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names event-response-and-firing registration (launcher-gui Task 10) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the run's evidence, the properties a no-event group drops and the registration each run uses (event-optional Task 10) | date: 2026-10-10]
"""Backend ``pair_logistic``: a logistic pair classifier per project, calibrated on held-out sessions.

Per recording group (one project): ``references.evidence_by_group`` decides once what the
group's days measured (``run_evidence``): events when every day has trial events and a
condition shared by every day, else a no-event run. Every single unit of every day is observed
on its day's resampled splits (``references.group_observations``) of its trials in an event run,
else of 5 s chunks of its window, and gives labeled same/different examples
(``examples.labeled_examples``). Sessions in group order alternate: even positions train a
``CalibratedPairModel`` on the ``model_properties(waveform_channels)`` of their examples (``PROPERTIES``
with one waveform pair: the peak-threshold-neighbourhood pair by default, or the full-array pair), odd positions
calibrate it, so calibration never sees a training session. A no-event group's model leaves out
the event properties ``fingerprint_r`` and ``d_log_rate_shared``, and ``d_log_rate_session``,
which its spontaneous rate already carries. A session is a day, or one temporal half of the day in a
same-day run. The same examples fit an ``ev.Ensemble`` for the depth registration
``functional_likelihood`` uses: an event run registers on event response + physiology, unaligned;
a no-event run on physiology + waveform, with day B shifted by each offset tried. Every cross-session unit pair
inside the proximity gate on registered depths is a candidate; its properties are measured
with day B's footprint aligned onto day A's channels, and the model's calibrated probability is
the score the common tracker reads. Waveform properties need waveform halves from raw voltage,
so a unit without them (templates tier) is refused before any work.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
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
from .calibrated import CalibratedPairModel
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run decides its evidence per group, refuses units without waveform halves and records the jitter, pair minimum and chunk length (event-optional Task 10) | date: 2026-10-10]
from .examples import (MIN_NEIGHBOUR_PAIRS, PROPERTIES, WAVEFORM_JITTER_UM, fit_ensemble, labeled_examples,
                       pair_properties)
from .functional_likelihood import NULL_QUANTILE, REGISTRATION_METHODS, _with_psth
from .references import CHUNK_SECONDS, evidence_by_group, group_observations, require_waveform_halves
# [AGENT EDIT END]
from .registration import functional_offsets
from .tracker import result_from_pairs

SCORE_COLUMN = 'calibrated_match_probability'


def _candidates(obs_a: dict, obs_b: dict, keys, crit, rel: float) -> list:
    """Rows of every unit pair of two days inside the proximity gate on registered depths."""
    rows = []
    for a, x in obs_a.items():
        full_a = _with_psth(x, keys)
        for b, y in obs_b.items():
            evidence = ev.pair_evidence(full_a, _with_psth(y, keys), offset_um=rel)
            if not ev.gates_passed(evidence, crit)['prox']:
                continue
            rows.append(dict(cellid_a=a, cellid_b=b, depth_delta_um=evidence['depth_delta_um'],
                             **pair_properties(x, y, keys, 0, offset_um=rel)))
    return rows


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fitting minimum of CalibratedPairModel, checked per part before fitting (final review fix round 1) | date: 2026-09-28]
MIN_PART_EXAMPLES = 20
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the waveform-property channel choices (peak_threshold, full_array) and the model properties each selects (threshold neighbourhood) | date: 2026-10-04]
WAVEFORM_CHANNELS = ('peak_threshold', 'full_array')


def model_properties(waveform_channels: str) -> tuple:
    """``PROPERTIES`` the classifier scores with: the peak-threshold neighbourhood or the full-array waveform pair."""
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each choice drops the other channel set's waveform and footprint correlations by their footprint names (event-optional Task 1) | date: 2026-10-09]
    if waveform_channels == 'peak_threshold':
        dropped = ('waveform_r', 'footprint_r')
    elif waveform_channels == 'full_array':
        dropped = ('waveform_r_local', 'footprint_r_local')
    else:
        raise ValueError(f'waveform_channels is {waveform_channels!r}; choose one of {WAVEFORM_CHANNELS}')
    # [AGENT EDIT END]
    return tuple(p for p in PROPERTIES if p not in dropped)
# [AGENT EDIT END]


def run_pair_logistic(units, features, geometry, timing_by_session, *, threshold=.9, contradiction_threshold=.1,
                      ambiguity_margin=.02, min_repeats=5, min_spikes=200, n_splits=20, seed=0, radius_um=80.0,
                      # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: min_pairs keyword, default MIN_NEIGHBOUR_PAIRS (event-optional Task 10, D53) | date: 2026-10-10]
                      min_pairs=MIN_NEIGHBOUR_PAIRS,
                      # [AGENT EDIT END]
                      registration='matches', l2=1.0, criteria=None, provenance=None,
                      run_id=None,
                      # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: waveform_channels keyword, default the peak-threshold neighbourhood (threshold neighbourhood) | date: 2026-10-04]
                      waveform_channels='peak_threshold',
                      # [AGENT EDIT END]
                      ) -> MatchResult:  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: declared return type (Task 8 fix round 1) | date: 2026-09-27]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic holds out sessions, not only days (Follow-up Task 4) | date: 2026-09-29]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring says a day without timing has no events (event-optional Task 10) | date: 2026-10-10]
    """Match within each ``recording_group`` by a logistic model calibrated on held-out sessions.

    ``timing_by_session`` maps matcher session ids to verified ``TrialTiming``; a day it lacks
    has no events (``None``: no day has timing). Geometry is measured channel order. Every
    explicit single unit needs waveform halves. Uses no source-cluster truth labels.
    """
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shared waveform-halves refusal names this backend (event-optional Task 10) | date: 2026-10-10]
    require_waveform_halves(units, features, 'pair_logistic')
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: waveform_channels is checked before any observation work (local-neighbourhood Task 3) | date: 2026-10-02]
    properties = model_properties(waveform_channels)
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: min_pairs is checked before any observation work (event-optional Task 10, D53) | date: 2026-10-10]
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 1:
        raise ValueError(f'min_pairs must be an int >= 1, not {min_pairs!r}')
    # [AGENT EDIT END]
    if registration not in REGISTRATION_METHODS:
        raise ValueError(f'registration is {registration!r}; choose one of {REGISTRATION_METHODS}')
    crit = criteria or ev.Criteria(use=('prox',))
    if 'prox' not in crit.use:
        raise ValueError(f"criteria.use is {crit.use}; pair_logistic gates candidates by 'prox'")
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run decides its evidence once per group and observes each group on it (event-optional Task 10) | date: 2026-10-10]
    timing_by_session = timing_by_session or {}
    evidence = evidence_by_group(units, features, timing_by_session, min_repeats=min_repeats)
    groups, skipped = group_observations(units, features, geometry, timing_by_session, min_repeats=min_repeats,
                                         min_spikes=min_spikes, min_reliability=crit.min_reliability,
                                         n_splits=n_splits, seed=seed, evidence=evidence)
    # [AGENT EDIT END]
    if not groups:
        raise ValueError(f'0 recording groups can be matched: {skipped}; pair_logistic trains and calibrates on a '
                         'project with at least 2 days of kept units. Add days or lower the minima.')
    scored, models, references = [], {}, []
    training_days, calibration_days = [], []
    offsets_um, diagnostics, reference_counts, session_pairs = {}, {}, {}, []
    for g in groups:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a no-event group's model drops the event properties and the session rate its spont carries; the examples take min_pairs and the seed (event-optional Task 10, D24) | date: 2026-10-10]
        e = evidence[g.group]
        group_properties = properties if e.events else tuple(
            p for p in properties if p not in ('fingerprint_r', 'd_log_rate_shared', 'd_log_rate_session'))
        examples = labeled_examples(g.observations, g.session_of, g.keys, radius_um=radius_um, min_pairs=min_pairs,
                                    seed=seed)
        # [AGENT EDIT END]
        train_days, held_out = list(g.sessions[0::2]), list(g.sessions[1::2])
        training = examples[examples.session.isin(train_days)]
        calibration = examples[examples.session.isin(held_out)]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a part too thin to fit is refused with its group, days, counts and fix before the classifier sees it (final review fix round 1) | date: 2026-09-28]
        for part, part_days, rows in (('training', train_days, training), ('calibration', held_out, calibration)):
            n_same = int(rows.same_unit.astype(bool).sum())
            n_different = len(rows) - n_same
            if not n_same or not n_different or len(rows) < MIN_PART_EXAMPLES:
                raise ValueError(
                    f'{g.group}: the {part} part (days {part_days}) holds {n_same} same-unit and {n_different} '
                    f'different-unit examples; pair_logistic needs both labels and at least {MIN_PART_EXAMPLES} '
                    'examples in each part. Widen radius_um, add days, or use functional_likelihood.')
        # [AGENT EDIT END]
        reference = f'{g.group}: trained on {train_days}, calibrated on held-out {held_out}'
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the classifier scores with the properties waveform_channels selects (local-neighbourhood Task 3) | date: 2026-10-02]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the classifier scores with the group's measured properties (event-optional Task 10) | date: 2026-10-10]
        model = CalibratedPairModel(group_properties, l2=l2).fit(
            training, calibration, training_groups=training.session, calibration_groups=calibration.session,
            reference=reference)
        # [AGENT EDIT END]
        # [AGENT EDIT END]
        models[g.group] = model.to_dict()
        references.append(reference)
        training_days += train_days
        calibration_days += held_out
        same = examples.same_unit.astype(bool)
        reference_counts[g.group] = dict(
            training_same=int((same & examples.session.isin(train_days)).sum()),
            training_different=int((~same & examples.session.isin(train_days)).sum()),
            calibration_same=int((same & examples.session.isin(held_out)).sum()),
            calibration_different=int((~same & examples.session.isin(held_out)).sum()))

        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registration as functional_likelihood: event response + physiology unaligned with events, physiology + waveform at each offset without (event-optional Task 10, D37) | date: 2026-10-10]
        ensemble, _, diff_ref = fit_ensemble(examples, ev.run_streams(e.events, waveform=not e.events))
        obs_by_session = {s: {c: o for c, o in g.observations.items() if g.session_of[c] == s} for s in g.sessions}
        offsets, diagnostics[g.group] = functional_offsets(obs_by_session, ensemble, diff_ref, g.keys,
                                                           method=registration, null_quantile=NULL_QUANTILE,
                                                           aligned=not e.events)
        # [AGENT EDIT END]
        offsets_um.update({s: float(offsets[s]) for s in g.sessions})

        for sa, sb in itertools.combinations(g.sessions, 2):
            rows = _candidates(obs_by_session[sa], obs_by_session[sb], g.keys, crit, offsets[sb] - offsets[sa])
            session_pairs.append(dict(recording_group=g.group, session_a=sa, session_b=sb, n_candidates=len(rows)))
            if not rows:
                continue
            table = pd.DataFrame(rows)
            table[SCORE_COLUMN] = model.predict_proba(table)
            table['recording_group'], table['siteid_a'], table['siteid_b'] = g.group, sa, sb
            scored.append(table)
    pairs = (pd.concat(scored, ignore_index=True) if scored
             else pd.DataFrame(columns=['cellid_a', 'cellid_b', SCORE_COLUMN]))
    eligible, _ = single_units(units)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the sessions whose timing the run read: those of every group whose evidence reads timing (event-optional Task 10) | date: 2026-10-10]
    sessions = eligible[['session_id', 'recording_group']].drop_duplicates()
    read = [s for s, group in zip(sessions.session_id, sessions.recording_group) if evidence[str(group)].reads_timing]
    # [AGENT EDIT END]
    metadata = dict(provenance or {})
    metadata.update(
        evidence_display_name='Pair logistic',
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic holds out sessions, not only days (Follow-up Task 4) | date: 2026-09-29]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the implementation names split examples of trials or 5 s chunks and the registration each run uses (event-optional Task 10) | date: 2026-10-10]
        implementation=('logistic model on labeled split examples (trials, else 5 s chunks) of alternate sessions, '
                        'calibrated on the held-out sessions; registration on event response + physiology with '
                        'events, else on physiology + waveform at each offset; common mutual-margin best-first tracker'),
        # [AGENT EDIT END]
        # [AGENT EDIT END]
        calibrated_model=models, training_days=training_days, calibration_days=calibration_days,
        registration=dict(method=registration, offsets_um=offsets_um, diagnostics=diagnostics),
        reference_counts=reference_counts, radius_um=radius_um, n_splits=n_splits, seed=seed, l2=l2,
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: provenance records the waveform channels (local-neighbourhood Task 3) | date: 2026-10-02]
        waveform_channels=waveform_channels,
        # [AGENT EDIT END]
        minimum_repeats=min_repeats, minimum_spikes=min_spikes, null_quantile=NULL_QUANTILE,
        criteria=crit.to_dict(), session_pairs=session_pairs, skipped_groups=skipped,
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: provenance saves the run evidence, the pair minimum, the jitter, timing only where it was read, and the chunk length when chunks were used (event-optional Task 10) | date: 2026-10-10]
        run_evidence={group: e.to_dict() for group, e in evidence.items()},
        min_pairs=min_pairs, waveform_jitter_um=WAVEFORM_JITTER_UM,
        timing_provenance={session: timing_by_session[session].provenance for session in read},
        evidence_columns=list(pairs.columns), used_optional_inputs=['timing'] if read else [])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the chunk length is saved only when a group observed on reference windows used chunks; group_observations skips a group of fewer than 2 sessions before building any (event-optional Task 10 fix round 1) | date: 2026-10-10]
    n_sessions = sessions.recording_group.astype(str).value_counts()
    observed = [evidence[group] for group, n in n_sessions.items() if n >= 2]
    if any(not e.events or e.spont != 'baseline' for e in observed):
        metadata.update(chunk_seconds=CHUNK_SECONDS)
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    return result_from_pairs(SPEC.id, SPEC.version, units, pairs, score_column=SCORE_COLUMN,
                             score_kind=ScoreKind.CALIBRATED_PROBABILITY, threshold=threshold,
                             contradiction_threshold=contradiction_threshold, ambiguity_margin=ambiguity_margin,
                             provenance=metadata, run_id=run_id, calibration_reference='; '.join(references))


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: version 3: waveform properties on the peak-threshold neighbourhood by default (threshold neighbourhood) | date: 2026-10-04]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: version 4: timing is optional (event-optional Task 10) | date: 2026-10-10]
SPEC = BackendSpec('pair_logistic', '4',
    requires=frozenset({Input.UNITS, Input.FEATURES, Input.GEOMETRY}), optional=frozenset({Input.TIMING}),
# [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic saves a registration, so its spec declares day_offsets (drift-view post-run fix) | date: 2026-10-08]
    provides=Provides(frozenset({ScoreKind.CALIBRATED_PROBABILITY}), frozenset({TrackKind.COMMON}), day_offsets=True),
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: help texts name optional events and chunks; min_pairs is a run option (event-optional Task 10, D53) | date: 2026-10-10]
    options=(Option('min_repeats', 'int', 5, 'Minimum condition repetitions for event references (event runs)'),
             Option('min_spikes', 'int', 200, 'Minimum spikes for a reference'),
             Option('n_splits', 'int', 20, 'Splits per day (trials, else 5 s chunks) for the labeled examples'),
             Option('seed', 'int', 0, 'Seed of the random splits and the same-unit waveform jitter'),
             Option('radius_um', 'float', 80.0, 'Same-day neighbours within this depth are different-unit examples'),
             Option('min_pairs', 'int', MIN_NEIGHBOUR_PAIRS,
                    'Same-day neighbour pairs within radius_um needed for the labeled examples'),
             # [AGENT EDIT END]
             Option('registration', 'choice', 'matches', 'Depth registration before the proximity gate',
                    choices=REGISTRATION_METHODS),
             Option('l2', 'float', 1.0, 'Ridge penalty of the logistic model and its calibration slope'),
             Option('waveform_channels', 'choice', 'peak_threshold',
                    'Channels of the waveform properties: both units\' peak-threshold neighbourhoods, or the full array',
                    choices=WAVEFORM_CHANNELS),
             *COMMON_TRACKER_OPTIONS),
    builtin=True,
    real_days=True)
# [AGENT EDIT END]


class PairLogisticBackend:
    spec = SPEC

    def available(self):
        return Availability(True, SPEC.version, None)

    def run(self, inp, options, ctx):
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: timing is optional, so a run without it passes no timing; min_pairs is passed (event-optional Task 10) | date: 2026-10-10]
        return run_pair_logistic(
            inp.units, inp.features, inp.geometry, inp.timing or {}, threshold=options['threshold'],
            contradiction_threshold=options['contradiction_threshold'], ambiguity_margin=options['ambiguity_margin'],
            min_repeats=options['min_repeats'], min_spikes=options['min_spikes'], n_splits=options['n_splits'],
            seed=options['seed'], radius_um=options['radius_um'], min_pairs=options['min_pairs'],
            registration=options['registration'], l2=options['l2'],
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the backend passes the waveform_channels option (local-neighbourhood Task 3) | date: 2026-10-02]
            waveform_channels=options['waveform_channels'],
            # [AGENT EDIT END]
            provenance=ctx.provenance.to_json(), run_id=ctx.run_id)
        # [AGENT EDIT END]
# [AGENT EDIT END]
