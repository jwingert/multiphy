import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from multiphy import evidence as ev
from multiphy.backends.local import functional_likelihood as fl
from multiphy.backends.local.references import alternating_references
from multiphy.contract.inputs import TrialTiming
from multiphy.features import _isi_acg


def timing():
    return TrialTiming(
        'original', 1000., (0, 4000),
        np.array([[i, i+400] for i in range(0, 4000, 500)]),
        tuple(dict(stimulus='sound:left:60', start=i+100, stop=i+300) for i in range(0, 4000, 500)),
        np.array([[i, i+100] for i in range(0, 4000, 500)]),
        {'sample_index_origin': 'zero_based_voltage_row', 'method': 'known synthetic sample clock'}).validate()


def feature():
    spikes = np.concatenate([np.arange(i+10, i+390, 10 if j % 2 == 0 else 20)
                             for j, i in enumerate(range(0, 4000, 500))])
    return dict(start_sample=0, stop_sample=4000, sample_rate=1000., spike_samples=spikes,
                fingerprint_bin_seconds=.05)


def test_references_alternate_whole_trials_and_preserve_real_intervals():
    f = feature()
    f['fingerprint_halves'] = {'sound:left:60': np.full((4, 2), -999.)}
    r = alternating_references(f, timing(), min_repeats=2)
    assert r['spont_a'] == pytest.approx(90) and r['spont_b'] == pytest.approx(50)
    np.testing.assert_array_equal(r['halves']['sound:left:60'][0], np.full(4, 100.))
    np.testing.assert_array_equal(r['halves']['sound:left:60'][1], [40., 60., 40., 60.])
    expected = np.concatenate([np.arange(i+10, i+390, 10) for i in [0, 1000, 2000, 3000]]) / 1000
    for actual, desired in zip((r['isi_a'], r['acg_a']), _isi_acg(expected)):
        np.testing.assert_array_equal(actual, desired)
    # Gaps and trial identity are retained; no alternating-spike surrogate.
    assert not np.array_equal(r['isi_a'], _isi_acg(f['spike_samples'][::2] / 1000)[0])
    with pytest.raises(ValueError, match='cuts a complete trial'):
        alternating_references(dict(f, stop_sample=3750,
            spike_samples=f['spike_samples'][f['spike_samples'] < 3750]), timing(), min_repeats=2)


def test_one_unit_per_day_is_refused():
    units=pd.DataFrame([dict(cellid=f'{s}-one',session_id=s,siteid=s,recording_group='source',su_type='SU') for s in ['A','B']])
    f=feature()
    # a burst early in every stimulus, in 10 bins, makes both trial halves share a PETH: the unit is reliable
    f['fingerprint_bin_seconds']=.02
    f['spike_samples']=np.concatenate([np.union1d(np.arange(i+10,i+390,10),np.arange(i+101,i+150,4)) for i in range(0,4000,500)])
    waveform=np.array([[0.,0.],[-2.,-1.],[1.,.5]])
    # halves carry the waveform itself: each half's spike width is measured, and a flat half has none
    f.update(waveform=waveform, waveform_halves=np.stack([waveform,waveform],axis=-1),eligible=True)
    features={c:copy.deepcopy(f) for c in units.cellid}
    with pytest.raises(ValueError, match=r'^0 distinct same-day neighbour pairs within 80 µm .*Rerun with a wider radius'):
        fl.run_functional_likelihood(units,features,{'A':np.array([[0,0],[0,20]]),'B':np.array([[0,0],[0,20]])},
            {'A':timing(),'B':timing()},min_repeats=2,n_splits=2)


def test_collision_keeps_one_observation_per_day(monkeypatch):
    units=pd.DataFrame([dict(cellid=f'{s}-{i}',session_id=s,siteid=s,recording_group='source',su_type='SU')
                        for s in ['A','B'] for i in range(6)])
    units.loc[len(units)] = ['A-mua','A','A','source','MUA']
    f=feature()
    f['spike_samples']=np.concatenate([i+np.array([10,30,50,110,120,130,160,210,220,310,350]) for i in range(0,4000,500)])
    f['fingerprint_bin_seconds']=.02
    waveform=np.array([[0.,0.],[-2.,-1.],[1.,.5]])
    f.update(waveform=waveform, waveform_halves=np.stack([waveform,waveform],axis=-1),eligible=True)
    features={c:copy.deepcopy(f) for c in units[units.su_type=='SU'].cellid}
    def collision_scores(a,b,keys,ensemble,crit,rel,same_scores,diff_scores):
        assert len(a)==6 and len(b)==6 and keys==('sound:left:60',)
        return pd.DataFrame([dict(cellid_a='A-0',cellid_b='B-0',p_global=.99),
                             dict(cellid_a='A-1',cellid_b='B-0',p_global=.99)]), {}
    monkeypatch.setattr(fl,'_score_session_pair',collision_scores)
    result=fl.run_functional_likelihood(units,features,{'A':np.array([[0,0],[0,20]]),'B':np.array([[0,0],[0,20]])},
        {'A':timing(),'B':timing()},min_repeats=2,min_spikes=2,n_splits=2)
    assert not any({'A-0','A-1'} <= set(t) for t in result.tracks)
    assert sum(map(len,result.tracks))+len(result.unmatched)==12 and result.excluded_units==('A-mua',)
    assert result.provenance.get('assignment')=='mutual_margin_best_first'
    result=fl.run_functional_likelihood(units,features,{'A':np.array([[0,0],[0,20]]),'B':np.array([[0,0],[0,20]])},
        {'A':timing(),'B':timing()},min_repeats=2,min_spikes=2,ambiguity_margin=0,n_splits=2)
    assert result.tracks==(('A-0','B-0'),)
    decisions={(r.cellid_a,r.cellid_b):r.decision for r in result.pairs.itertuples()}
    assert decisions[('A-0','B-0')]=='selected' and decisions[('A-1','B-0')]=='session_collision'
