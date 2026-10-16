# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify diagnostic timing, missing-pair identity, and tie-aware reciprocal ranks | date: 2026-09-18]
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multiphy import diagnostic_report as diagnostic
from multiphy.backends import result_from_pairs


def fixture():
    ids = ['a0', 'a1', 'b0', 'b1']
    units = pd.DataFrame(dict(cellid=ids, session_id=['s0', 's1', 's0', 's1'], su_type='SU'))
    labels = pd.DataFrame(dict(cellid=ids, source_unit_id=['a', 'a', 'b', 'b'],
        source_session='original', well='well', partition='test'))
    features = {}
    for i, cid in enumerate(ids):
        wave = np.array([[0, 0], [-1, 0], [-3, -1], [2, 0], [0, 0]], float) * (i // 2 + 1)
        features[cid] = dict(waveform=wave, waveform_halves=np.stack([wave, wave], axis=-1),
            sample_rate=1000., start_sample=i % 2 * 1000, stop_sample=(i % 2 + 1) * 1000, n_waveform_spikes=20,
            half_waveform_spike_samples=[np.arange(10), np.arange(10, 20)],
            fingerprints={'sound': np.array([0., 1., 3.]) if i < 2 else np.array([3., 1., 0.])},
            trial_counts={'sound': 3}, fingerprint_bin_seconds=.05)
    dataset = SimpleNamespace(units=units, labels=labels, features=features,
        metadata={'dataset_id': 'synthetic', 'extraction_options': {'before': 2}}, validate=lambda: None)
    result = result_from_pairs('unitmatch', 'synthetic', units,
        pd.DataFrame(dict(cellid_a=['a0', 'b0'], cellid_b=['a1', 'b1'], match_score=[.99, .1])))
    return dataset, result


def test_tied_best_is_not_unique_and_missing_scores_do_not_rank():
    assert diagnostic.candidate_rank(.8, [.8, .8, .1, np.nan]) == (1, 2, 3)
    assert diagnostic.candidate_rank(.5, [.8, .5, .1]) == (2, 1, 3)
    assert np.isnan(diagnostic.candidate_rank(np.nan, [.8])[0])


def test_diagnostics_count_source_pairs_and_keep_unshifted_sample_origin():
    dataset, result = fixture()
    observations, pairs, evaluation = diagnostic.diagnose(dataset, result)
    assert observations.dominant_offset_ms.eq(0).all()
    assert observations.trough_sample.eq(2).all()
    assert np.allclose(observations.half_r, 1)
    assert len(pairs) == 2 and (~pairs.matched).sum() == 1
    assert pairs.mutual_unique_best.all()
    assert pairs.candidates_a.eq(2).all()
    assert evaluation['false_negative_pairs'] == 1


def test_diagnostics_refuse_mismatched_population():
    dataset, result = fixture()
    dataset.units = dataset.units.iloc[:-1]
    with pytest.raises(ValueError, match='identical'):
        diagnostic.diagnose(dataset, result)


def test_report_writes_only_new_cache_directory_with_lineage(tmp_path, monkeypatch):
    dataset, result = fixture()
    monkeypatch.setattr(diagnostic, 'load_dataset', lambda _: dataset)
    monkeypatch.setattr(diagnostic, 'load_result', lambda _: result)
    monkeypatch.setattr(diagnostic, '_plot_misses', lambda *args: None)
    target = diagnostic.write_report('dataset', 'result', output_dir='diagnostic', root=tmp_path)
    text = (target / 'report.md').read_text()
    assert '1 of 2 available same-source pairs' in text
    assert 'uniquely first in both directions' in text
    assert 'each internal mean uses 10 snippets' in text
    assert '2 candidates per direction' in text and '2–2' not in text
    assert 'first internal waveform half of observation A' in text
    assert 'indices 0 or 4' in text
    assert (target / 'METHODS.md').exists() and (target / 'names.py').exists()
    with pytest.raises(FileExistsError):
        diagnostic.write_report('dataset', 'result', output_dir='diagnostic', root=tmp_path)
# [AGENT EDIT END]
