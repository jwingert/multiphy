# [AGENT EDIT START | agent: codex | user: wingertj | reason: test original alternating references, verified nominal trial timing, mixture scores and unmatched baseline behavior | date: 2026-09-18]
import copy
import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match import evidence as ev
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the matcher is named by mechanism; baphy timing lives in prepare_baphy (Task 3) | date: 2026-09-27]
from nems_lbhb.unit_match.backends.local import functional_likelihood as fl
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: alternating_references lives in references.py (Task 4) | date: 2026-09-27]
from nems_lbhb.unit_match.backends.local.references import alternating_references
# [AGENT EDIT END]
from nems_lbhb.unit_match.contract.inputs import TrialTiming
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: prepare_baphy moved to nems_lbhb.unit_match_lbhb (Plan C Task 3) | date: 2026-10-01]
from nems_lbhb.unit_match_lbhb.prepare_baphy import recover_baphy_timing
# [AGENT EDIT END]
# [AGENT EDIT END]
from nems_lbhb.unit_match.features import _isi_acg


def timing():
    return TrialTiming(  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: renamed (Task 3) | date: 2026-09-27]
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


def test_references_alternate_whole_trials_and_preserve_real_intervals():  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: named without history (Task 3) | date: 2026-09-27]
    f = feature()
    f['fingerprint_halves'] = {'sound:left:60': np.full((4, 2), -999.)}
    r = alternating_references(f, timing(), min_repeats=2)  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: moved to references.py (Task 4) | date: 2026-09-27]
    assert r['spont_a'] == pytest.approx(90) and r['spont_b'] == pytest.approx(50)
    np.testing.assert_array_equal(r['halves']['sound:left:60'][0], np.full(4, 100.))
    np.testing.assert_array_equal(r['halves']['sound:left:60'][1], [40., 60., 40., 60.])
    expected = np.concatenate([np.arange(i+10, i+390, 10) for i in [0, 1000, 2000, 3000]]) / 1000
    for actual, desired in zip((r['isi_a'], r['acg_a']), _isi_acg(expected)):
        np.testing.assert_array_equal(actual, desired)
    # Gaps and trial identity are retained; no alternating-spike surrogate.
    assert not np.array_equal(r['isi_a'], _isi_acg(f['spike_samples'][::2] / 1000)[0])
    with pytest.raises(ValueError, match='cuts a complete trial'):
        alternating_references(dict(f, stop_sample=3750,  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: moved to references.py (Task 4) | date: 2026-09-27]
            spike_samples=f['spike_samples'][f['spike_samples'] < 3750]), timing(), min_repeats=2)


def baphy_spec(tmp_path, *, stop=0.4):
    parm = tmp_path / 'fixture.m'
    lines = ["globalparams.SiteID = 'TESTa';"]
    index = 0
    for trial in range(1, 5):
        for note, start, end in [('TRIALSTART', 2., 2.), ('PreStimSilence , sound:left:60 , Reference', 2., 2.1),
                                 ('Stim , sound:left:60 , Reference', 2.1, 2.3), ('TRIALSTOP', 2.+stop, 2.+stop+.03)]:
            index += 1
            lines += [f"exptevents({index}).Note = '{note}';", f'exptevents({index}).StartTime = {start};',
                      f'exptevents({index}).StopTime = {end};', f'exptevents({index}).Trial = {trial};']
    parm.write_text('\n'.join(lines))
    np.save(tmp_path / 'samples.npy', np.arange(4)*500 + 1000)
    np.save(tmp_path / 'states.npy', np.ones(4, dtype=int))
    sources = {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in [('parmfile', parm), ('ap_ttl_samples', tmp_path/'samples.npy'), ('ap_ttl_states', tmp_path/'states.npy')]}
    return dict(source_session='TESTa', sample_rate=1000., n_samples=2000, analysis_window=[0, 2000],
        provenance=sources, clock_verification=dict(sample_index_origin='zero_based_voltage_row', job_onsets_verified=True,
            first_sample_number=1000, stimulus_trial_verification={'policy':'strict_trial_count',
                'annotated_trial_count':4, 'analysis_stop_sample':2000}),
        stimuli=[dict(stimulus='sound:left:60', start=i+100, stop=i+300, trial=j+1) for j,i in enumerate(range(0,2000,500))],
        baseline_windows=[[i,i+100] for i in range(0,2000,500)])


def test_timing_uses_trialstop_start_not_stop_or_next_ttl_and_verifies_hashes(tmp_path):
    spec = baphy_spec(tmp_path)
    verified = recover_baphy_timing(spec)
    np.testing.assert_array_equal(verified.trial_windows, [[0,400],[500,900],[1000,1400],[1500,1900]])
    assert verified.source_session == 'TESTa'
    older = copy.deepcopy(spec)
    older['clock_verification'].pop('stimulus_trial_verification')
    older.pop('analysis_window')
    np.testing.assert_array_equal(recover_baphy_timing(older).trial_windows, verified.trial_windows)
    (tmp_path/'fixture.m').write_text((tmp_path/'fixture.m').read_text()+'\n')
    with pytest.raises(ValueError, match='hash changed'):
        recover_baphy_timing(spec)


def test_timing_rejects_ambiguous_overlap_and_altered_prepared_events(tmp_path):
    spec = baphy_spec(tmp_path, stop=.6)
    with pytest.raises(ValueError, match='TRIALSTOP escapes'):
        recover_baphy_timing(spec)
    spec = baphy_spec(tmp_path)
    spec['stimuli'][0]['stop'] += 1
    with pytest.raises(ValueError, match='stimulus/baseline intervals differ'):
        recover_baphy_timing(spec)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: BigNat timing recovery compares the written names the manifest holds (bignat-names Task 2) | date: 2026-10-11]
def test_timing_recovery_reads_bignat_names(tmp_path):
    spec = baphy_spec(tmp_path)
    parm = tmp_path / 'fixture.m'
    lines = parm.read_text().replace('sound:left:60', 'sound.wav').split('\n')
    lines.insert(1, "exptparams(1).TrialObject(1).ReferenceClass = 'BigNat';")
    parm.write_text('\n'.join(lines))
    spec['provenance']['parmfile']['sha256'] = hashlib.sha256(parm.read_bytes()).hexdigest()
    for stim in spec['stimuli']:
        stim['stimulus'] = 'sound.wav:1+NULL:2'
    spec['provenance']['reference_class'] = 'BigNat'
    spec['provenance']['stimulus_names'] = {'sound.wav': 'sound.wav:1+NULL:2'}
    assert [s['stimulus'] for s in recover_baphy_timing(spec).stimuli] == ['sound.wav:1+NULL:2'] * 4
    recorded = copy.deepcopy(spec)
    recorded['provenance'].pop('reference_class')
    with pytest.raises(ValueError, match='stimulus/baseline intervals differ'):
        recover_baphy_timing(recorded)
    other = copy.deepcopy(spec)
    other['provenance']['stimulus_names'] = {'sound.wav': 'sound.wav:2+NULL:1'}
    with pytest.raises(ValueError) as caught:
        recover_baphy_timing(other)
    assert str(caught.value) == 'BigNat stimulus names differ from the prepared manifest'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the original per-day-pair ensemble and per-unit-maxima mixture are gone; one unit per day leaves no same-day neighbours for the project references (functional-likelihood Task 7) | date: 2026-09-27]
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
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refusal names the count and the radius, not only the fix (final review fix round 1) | date: 2026-09-28]
    with pytest.raises(ValueError, match=r'^0 distinct same-day neighbour pairs within 80 µm .*Rerun with a wider radius'):
        fl.run_functional_likelihood(units,features,{'A':np.array([[0,0],[0,20]]),'B':np.array([[0,0],[0,20]])},
            {'A':timing(),'B':timing()},min_repeats=2,n_splits=2)
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: renamed; the common tracker keeps one observation per day (Task 1) | date: 2026-09-27]
def test_collision_keeps_one_observation_per_day(monkeypatch):
# [AGENT EDIT END]
    units=pd.DataFrame([dict(cellid=f'{s}-{i}',session_id=s,siteid=s,recording_group='source',su_type='SU')
                        for s in ['A','B'] for i in range(6)])
    units.loc[len(units)] = ['A-mua','A','A','source','MUA']
    f=feature()
    f['spike_samples']=np.concatenate([i+np.array([10,30,50,110,120,130,160,210,220,310,350]) for i in range(0,4000,500)])
    f['fingerprint_bin_seconds']=.02
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: patch the per-session-pair scoring of v2; halves carry the waveform so each half has a spike width (functional-likelihood Task 7) | date: 2026-09-27]
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
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: common tracker: no track holds two observations of day A (Task 1) | date: 2026-09-27]
    assert not any({'A-0','A-1'} <= set(t) for t in result.tracks)
    assert sum(map(len,result.tracks))+len(result.unmatched)==12 and result.excluded_units==('A-mua',)
    assert result.provenance.get('assignment')=='mutual_margin_best_first'
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1: with no margin the tied links reach the session-uniqueness check itself (Task 1) | date: 2026-09-27]
    result=fl.run_functional_likelihood(units,features,{'A':np.array([[0,0],[0,20]]),'B':np.array([[0,0],[0,20]])},
        {'A':timing(),'B':timing()},min_repeats=2,min_spikes=2,ambiguity_margin=0,n_splits=2)  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: two splits suffice for the patched scores (functional-likelihood Task 7) | date: 2026-09-27]
    assert result.tracks==(('A-0','B-0'),)
    decisions={(r.cellid_a,r.cellid_b):r.decision for r in result.pairs.itertuples()}
    assert decisions[('A-0','B-0')]=='selected' and decisions[('A-1','B-0')]=='session_collision'
    # [AGENT EDIT END]
# [AGENT EDIT END]
