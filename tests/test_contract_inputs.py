import numpy as np
import pandas as pd
import pytest

from multiphy.contract.inputs import Input, MatchingInput, RawWaveformBundle, CheckpointRef


def units(*ids):
    return pd.DataFrame({'cellid': ids, 'session_id': [i.split('-')[0] for i in ids], 'su_type': ['SU'] * len(ids)})


def bundle(ids, n_chan=4, n_samp=10):
    geometry = {s: np.column_stack([np.zeros(n_chan), np.arange(n_chan) * 20.]) for s in {i.split('-')[0] for i in ids}}
    return RawWaveformBundle(tuple(ids), np.zeros((len(ids), n_samp, n_chan, 2)), tuple(i.split('-')[0] for i in ids),
                             geometry, tuple(np.array([0.1, 0.2]) for _ in ids),
                             {'raw_source': 'test', 'split_method': 'halves'},
                             waveform_spike_counts=np.ones((len(ids), 2)))


def test_units_only_input_is_valid_and_available_is_units():
    inp = MatchingInput(units('A-1', 'B-1'), group='g').validate()
    assert inp.available() == frozenset({Input.UNITS})


def test_from_bundle_reports_every_field_it_filled():
    ids = ('A-1', 'B-1')
    inp = MatchingInput.from_bundle(units(*ids), bundle(ids), group='g',
                                    pairs=pd.DataFrame(columns=['cellid_a', 'cellid_b']),
                                    checkpoint=CheckpointRef('/x/ckpt', 'np1', 'abc'))
    assert inp.available() == {Input.UNITS, Input.CANDIDATE_PAIRS, Input.WAVEFORM_HALVES, Input.GEOMETRY,
                               Input.SPIKE_TIMES, Input.CHECKPOINT}
    assert inp.bundle().cellids == ids and inp.bundle().provenance['raw_source'] == 'test'


def test_label_columns_are_rejected():
    u = units('A-1', 'B-1'); u['partition'] = 'test'
    with pytest.raises(ValueError, match='partition'):
        MatchingInput(u, group='g').validate()


def test_partial_waveforms_name_the_missing_unit():
    ids = ('A-1', 'B-1', 'B-2')
    b = bundle(ids[:2])
    with pytest.raises(ValueError, match='B-2'):
        MatchingInput.from_bundle(units(*ids), b, group='g').validate()


def test_waveform_shape_and_geometry_checks():
    ids = ('A-1', 'B-1')
    b = bundle(ids)
    bad = RawWaveformBundle(b.cellids, b.waveforms[..., :1], b.session_ids, b.channel_positions,
                            b.spike_times_seconds, b.provenance)
    with pytest.raises(ValueError, match='halves'):
        MatchingInput.from_bundle(units(*ids), bad, group='g').validate()
    missing_geom = RawWaveformBundle(b.cellids, b.waveforms, b.session_ids, {'A': b.channel_positions['A']},
                                     b.spike_times_seconds, b.provenance)
    with pytest.raises(ValueError, match='geometry'):
        MatchingInput.from_bundle(units(*ids), missing_geom, group='g').validate()


def test_fingerprint_requires_unique_bin_keys():
    ids = ('A-1', 'B-1')
    b = bundle(ids)
    fp = RawWaveformBundle(b.cellids, b.waveforms, b.session_ids, b.channel_positions, b.spike_times_seconds,
                           b.provenance, peth=np.zeros((2, 2)), peth_bin_keys=('k', 'k'))
    with pytest.raises(ValueError, match='bin keys'):
        MatchingInput.from_bundle(units(*ids), fp, group='g').validate()
def test_fitted_model_input_gone(tmp_path):
    assert ('FITTED_MODEL' in Input.__members__) is False
    import dataclasses
    import inspect
    assert 'fitted_model' not in {f.name for f in dataclasses.fields(MatchingInput)}
    assert 'fitted_model' not in inspect.signature(MatchingInput.from_bundle).parameters


def test_docs_name_no_fixed_model_path(tmp_path):
    from pathlib import Path
    import re
    import multiphy as package
    repo = Path(__file__).resolve().parents[1]
    removed = re.compile(r'score-local|benchmark_local|--model\b|previously fitted|frozen[- ]model|evidence ablations'
                         r'|local combined|stimulus-only and physiology-only|local probability calibration', re.I)
    documents = sorted(Path(package.__file__).parent.rglob('*.md')) + [repo / 'README.md']
    hits = [f'{p.name}:{n}' for p in documents
            for n, line in enumerate(p.read_text().splitlines(), 1)
            if removed.search(line)]
    assert hits == []
