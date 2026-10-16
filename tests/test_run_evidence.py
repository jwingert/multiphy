"""``run_evidence``: events need a condition with ``min_repeats`` on every day; spont reads baselines only
when every day has them; each fallback is one reason, grouped into tags by ``evidence_tags``."""
import dataclasses
import inspect

import numpy as np
import pytest

from multiphy.contract.inputs import (
    NO_EVENTS_TAG, WHOLE_RECORDING_TAG, RunEvidence, TrialTiming, evidence_tags, run_evidence)

PROVENANCE = {'sample_index_origin': 'zero_based_voltage_row', 'method': 'test'}


def _timing(session, conditions, *, baselines=True):
    """Trial i is [1000i, 1000i + 900) at 1 kHz: baseline [1000i, 1000i + 100), stimulus [1000i + 100, 1000i + 600)."""
    starts = [1000 * i for i in range(len(conditions))]
    return TrialTiming(
        source_session=session, sample_rate=1000.0, analysis_window=(0, 1000 * len(conditions)),
        trial_windows=np.asarray([[t, t + 900] for t in starts], np.int64),
        stimuli=tuple({'stimulus': c, 'start': t + 100, 'stop': t + 600, 'trial': i + 1}
                      for i, (c, t) in enumerate(zip(conditions, starts))),
        baseline_windows=(np.asarray([[t, t + 100] for t in starts], np.int64) if baselines
                          else np.empty((0, 2), np.int64)),
        provenance=dict(PROVENANCE)).validate()


def test_run_evidence_counts_repeats_per_day():
    evidence = run_evidence(('D1', 'D2'), {'D1': _timing('D1', ['A'] * 5), 'D2': _timing('D2', ['A'] * 4)})
    assert evidence.events is False
    assert evidence.shared == {'D1+D2': 0}
    assert evidence.spont == 'baseline'
    assert evidence.reasons == ('no events · shared conditions per day pair: D1+D2 0',)


def test_run_evidence_three_days_need_one_common_condition():
    timing = {'D1': _timing('D1', ['A', 'B'] * 5), 'D2': _timing('D2', ['B', 'C'] * 5),
              'D3': _timing('D3', ['C', 'A'] * 5)}
    evidence = run_evidence(('D1', 'D2', 'D3'), timing)
    assert evidence.events is False
    assert evidence.shared == {'D1+D2': 1, 'D1+D3': 1, 'D2+D3': 1}


def test_run_evidence_day_without_timing():
    evidence = run_evidence(('D1', 'D2'), {'D1': _timing('D1', ['A'] * 5)})
    assert evidence.events is False
    assert evidence.spont == 'whole_recording'
    assert evidence.reasons == ('no events · shared conditions per day pair: D1+D2 0',
                                'spont · whole recording · days without baseline windows: D2')
    assert evidence.reads_timing is False
    assert evidence.to_dict() == {'events': False, 'spont': 'whole_recording', 'shared': {'D1+D2': 0},
                                  'reasons': list(evidence.reasons)}


def test_run_evidence_empty_baseline():
    timing = {day: _timing(day, ['A'] * 5, baselines=False) for day in ('D1', 'D2')}
    evidence = run_evidence(('D1', 'D2'), timing)
    assert evidence.events is True
    assert evidence.spont == 'whole_recording'
    assert evidence.reasons == ('spont · whole recording · days without baseline windows: D1, D2',)
    assert evidence.reads_timing is True


def test_evidence_tags_group_reasons():
    first = 'no events · shared conditions per day pair: D1+D2 0'
    second = 'no events · shared conditions per day pair: D3+D4 0'
    spont = 'spont · whole recording · days without baseline windows: D2'
    assert evidence_tags([spont, first, second]) == [(NO_EVENTS_TAG, f'{first}\n{second}'),
                                                     (WHOLE_RECORDING_TAG, spont)]
    assert evidence_tags([first, first]) == [(NO_EVENTS_TAG, first)]
    with pytest.raises(ValueError, match="unknown evidence reason: 'other'"):
        evidence_tags([first, 'other'])


def test_trial_timing_accepts_empty_baseline():
    for baselines in (np.empty((0, 2), np.int64), [], ()):
        timing = TrialTiming(
            source_session='D1', sample_rate=1000.0, analysis_window=(0, 2000),
            trial_windows=np.asarray([[0, 900], [1000, 1900]], np.int64),
            stimuli=({'stimulus': 'A', 'start': 100, 'stop': 600, 'trial': 1},
                     {'stimulus': 'A', 'start': 1100, 'stop': 1600, 'trial': 2}),
            baseline_windows=baselines, provenance=dict(PROVENANCE))
        assert timing.validate() is timing
        assert run_evidence(('D1',), {'D1': timing}, min_repeats=2).spont == 'whole_recording'


def test_run_evidence_refuses_one_repeat():
    with pytest.raises(ValueError) as caught:
        run_evidence(('D1', 'D2'), {}, min_repeats=1)
    assert str(caught.value) == 'min_repeats must be an int >= 2, not 1'


def test_run_evidence_signatures():
    parameters = inspect.signature(run_evidence).parameters
    assert list(parameters) == ['sessions', 'timing_by_session', 'min_repeats']
    assert parameters['min_repeats'].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters['min_repeats'].default == 5
    assert [f.name for f in dataclasses.fields(RunEvidence)] == ['events', 'spont', 'shared', 'reasons']
