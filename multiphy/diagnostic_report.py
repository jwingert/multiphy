# [AGENT EDIT START | agent: codex | user: wingertj | reason: untuned waveform alignment and missed-pair diagnostics from compact benchmark inputs | date: 2026-09-18]
"""Read-only matching diagnostics; never fits thresholds or changes assignments."""
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd

from .benchmark import load_dataset
from .backends import load_result
from .benchmark_pairs import _aligned_fingerprints
from .evaluation import evaluate_result
from .paths import resolve_cache_path
from .reporting_names import DEFINITIONS, count_range


NAMES = {
    'title': 'Waveform timing and missed identity pairs',
    'table_title': 'Missed pairs: observations of the same original sorted unit',
    'source_unit_id': 'Original sorted unit', 'waveform_r': 'Full-array waveform Pearson correlation',
    'fingerprint_r': 'Stimulus response Pearson correlation', 'rank_a': 'Response rank, first to second',
    'rank_b': 'Response rank, second to first', 'match_score': 'UnitMatch directional-minimum score',
    'half_r_min': 'Lower internal-half waveform Pearson correlation',
    'dominant_delta': 'Dominant-extremum displacement (samples)',
    'plot_title': 'UnitMatch missed pairs: two time segments of each original sorted unit',
    'first': 'First recording segment (mean)', 'second': 'Second recording segment (mean)',
    'time': 'Time from exported spike timestamp (ms)',
    'voltage': 'Mean filtered, referenced voltage (microvolts)',
    'figure_note': 'A missed pair means UnitMatch did not join the two segments carrying the same original sorter unit label.\nEach panel uses the electrode with largest peak-to-peak amplitude of the AVERAGE of the two segment means.\nElectrode numbers are zero-based indices in the saved channel array. Title correlations use all {channels} electrodes × {samples} samples.\n{processing}\n{spikes} spikes per displayed mean; zero marks the exported spike timestamp; no realignment or amplitude normalization.',
    'panel': '{letter}. {unit} — electrode {channel}\nFull-array waveform Pearson correlation {correlation:.3f}',
}


def display(key):
    return NAMES[key]


def correlation(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan


def candidate_rank(score, scores):
    """Rank 1 includes ties; unique best requires exactly one tied candidate."""
    finite = np.asarray(scores, float)
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(score) or not len(finite):
        return np.nan, 0, len(finite)
    ties = np.isclose(finite, score, rtol=0, atol=1e-12)
    return int(1 + np.sum((finite > score) & ~ties)), int(ties.sum()), len(finite)


def diagnose(dataset, result):
    dataset.validate()
    result.validate()
    if set(dataset.units.cellid) != set(result.units.cellid):
        raise ValueError('Diagnostics require the identical dataset and result population')
    if dataset.units.set_index('cellid').session_id.to_dict() != result.units.set_index('cellid').session_id.to_dict():
        raise ValueError('Diagnostic session identities must agree')
    before = dataset.metadata['extraction_options']['before']
    labels = dataset.labels.set_index('cellid')
    observations = []
    for row in dataset.units.itertuples():
        feature = dataset.features[row.cellid]
        wave = np.asarray(feature['waveform'])
        halves = np.asarray(feature['waveform_halves'])
        channel = int(np.argmax(np.ptp(wave, axis=0)))
        trace = wave[:, channel]
        dominant = int(np.argmax(np.abs(trace)))
        half_extrema = [int(np.argmax(np.abs(halves[:, channel, i]))) for i in range(2)]
        observations.append(dict(cellid=row.cellid, source_unit_id=labels.loc[row.cellid, 'source_unit_id'],
            session_id=row.session_id, peak_channel=channel, trough_sample=int(np.argmin(trace)),
            peak_sample=int(np.argmax(trace)), dominant_sample=dominant,
            dominant_polarity='positive' if trace[dominant] > 0 else 'negative',
            dominant_offset_ms=(dominant - before) * 1000 / feature['sample_rate'],
            dominant_at_edge=dominant in (0, len(trace) - 1),
            half_dominant_delta=abs(half_extrema[1] - half_extrema[0]),
            half_r=correlation(halves[..., 0], halves[..., 1]),
            peak_channel_half_r=correlation(halves[:, channel, 0], halves[:, channel, 1]),
            peak_to_peak_uv=float(np.ptp(trace)), n_waveform_spikes=feature['n_waveform_spikes']))
    observed = pd.DataFrame(observations).set_index('cellid', drop=False)
    # Enumerate every opposite-session candidate within each original recording,
    # including candidates excluded by another method's spatial gate.
    scores, candidates = {}, {}
    session = dataset.units.set_index('cellid').session_id.to_dict()
    for _, group in dataset.labels.groupby(['well', 'source_session']):
        for a, b in combinations(group.cellid, 2):
            if session[a] == session[b]:
                continue
            x, y, conditions, _, _, _ = _aligned_fingerprints(dataset.features[a], dataset.features[b], 1, None)
            score = correlation(x, y) if x is not None else np.nan
            scores[frozenset((a, b))] = (score, len(conditions), len(x) if x is not None else 0)
            candidates.setdefault((a, session[b]), []).append(score)
            candidates.setdefault((b, session[a]), []).append(score)
    assigned = {frozenset(pair) for track in result.tracks for pair in combinations(track, 2)}
    probability = {frozenset((p.cellid_a, p.cellid_b)): p.match_score for p in result.pairs.itertuples()}
    rows = []
    for _, group in dataset.labels.groupby(['well', 'source_session', 'source_unit_id']):
        for left, right in combinations(group.itertuples(), 2):
            a, b = left.cellid, right.cellid
            if session[a] == session[b]:
                continue
            if dataset.features[a]['start_sample'] > dataset.features[b]['start_sample']:
                a, b = b, a
            key = frozenset((a, b))
            score, n_conditions, n_bins = scores[key]
            ra, ta, na = candidate_rank(score, candidates[(a, session[b])])
            rb, tb, nb = candidate_rank(score, candidates[(b, session[a])])
            rows.append(dict(source_unit_id=left.source_unit_id, cellid_a=a, cellid_b=b,
                matched=key in assigned, match_score=probability.get(key, np.nan),
                waveform_r=correlation(dataset.features[a]['waveform'], dataset.features[b]['waveform']),
                fingerprint_r=score, rank_a=ra, rank_b=rb, ties_a=ta, ties_b=tb,
                candidates_a=na, candidates_b=nb, mutual_best=ra == rb == 1,
                mutual_unique_best=ra == rb == 1 and ta == tb == 1,
                n_conditions=n_conditions, n_bins=n_bins,
                half_r_min=min(observed.loc[a, 'half_r'], observed.loc[b, 'half_r']),
                peak_channel_half_r_min=min(observed.loc[a, 'peak_channel_half_r'], observed.loc[b, 'peak_channel_half_r']),
                dominant_delta=abs(observed.loc[a, 'dominant_sample'] - observed.loc[b, 'dominant_sample'])))
    return observed.reset_index(drop=True), pd.DataFrame(rows), evaluate_result(result, dataset.labels)


def _preprocessing_text(dataset):
    profiles = [source.get('provenance', {}).get('waveform_preprocessing', {})
                for source in dataset.metadata.get('sources', [])]
    if not profiles or any(p.get('profile') != 'shared_unwhitened_v1' for p in profiles):
        return 'Filtering and reference details: unavailable for this report from saved shared-profile metadata.'
    hz = count_range([p['highpass_hz'] for p in profiles])
    order = count_range([p['filter_order'] for p in profiles])
    references = count_range([len(p['reference_channels']) for p in profiles])
    return (f'{hz}-Hz high-pass Butterworth filter (order {order}, forward/backward); '
            f'at each sample subtract the median across {references} filtered reference channels. No whitening.')


def _plot_misses(dataset, pairs, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({'legend.fontsize': 8, 'figure.figsize': (8, 6), 'axes.labelsize': 20,
        'axes.titlesize': 20, 'axes.spines.right': False, 'axes.spines.top': False,
        'xtick.labelsize': 20, 'ytick.labelsize': 20, 'font.family': ['Calibri', 'Arial', 'DejaVu Sans'],
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.linewidth': 5,
        'xtick.major.size': 18, 'ytick.major.size': 18, 'xtick.major.width': 5,
        'ytick.major.width': 5, 'xtick.minor.width': 3, 'ytick.minor.width': 3,
        'xtick.minor.size': 5, 'ytick.minor.size': 5})
    missed = pairs[~pairs.matched]
    if missed.empty:
        return
    columns = min(3, len(missed))
    rows = int(np.ceil(len(missed) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 5 * rows + 4), squeeze=False)
    for index, (axis, pair) in enumerate(zip(axes.ravel(), missed.itertuples())):
        first, second = dataset.features[pair.cellid_a], dataset.features[pair.cellid_b]
        channel = int(np.argmax(np.ptp((first['waveform'] + second['waveform']) / 2, axis=0)))
        time = (np.arange(len(first['waveform'])) - dataset.metadata['extraction_options']['before']) * 1000 / first['sample_rate']
        axis.plot(time, first['waveform'][:, channel], color='#176ca4', label=display('first'), linewidth=2)
        axis.plot(time, second['waveform'][:, channel], color='#ce4f27', label=display('second'), linewidth=2)
        axis.axvline(0, color='0.6', linewidth=1)
        axis.set_title(display('panel').format(unit=pair.source_unit_id, correlation=pair.waveform_r, letter=chr(65 + index), channel=channel))
    for axis in axes.ravel()[len(missed):]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .956), ncol=2, fontsize=16)
    fig.suptitle(display('plot_title'), fontsize=22)
    fig.supxlabel(display('time'), fontsize=22, y=.15)
    fig.supylabel(display('voltage'), fontsize=22)
    first_wave = next(iter(dataset.features.values()))['waveform']
    spike_counts = [dataset.features[c]['n_waveform_spikes'] for c in set(missed.cellid_a) | set(missed.cellid_b)]
    fig.text(.5, .01, display('figure_note').format(channels=first_wave.shape[1], samples=first_wave.shape[0],
        spikes=count_range(spike_counts), processing=_preprocessing_text(dataset)), ha='center', va='bottom', fontsize=16)
    fig.tight_layout(rect=(.025, .18, 1, .92))
    fig.savefig(target / 'missed_waveforms.png', dpi=130)
    plt.close(fig)


def write_report(dataset_path, result_path, *, output_dir=None, root=None):
    dataset, result = load_dataset(dataset_path), load_result(result_path)
    observed, pairs, evaluation = diagnose(dataset, result)
    target = resolve_cache_path(output_dir or Path('benchmarks') / dataset.metadata['dataset_id'] /
        'diagnostics' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex), root=root)
    target.mkdir(parents=True, exist_ok=False)
    observed.to_csv(target / 'waveform_quality.csv', index=False)
    pairs.to_csv(target / 'identity_pairs.csv', index=False)
    (target / 'names.py').write_text('"""Display labels for this diagnostic artifact."""\nNAMES = ' + repr(NAMES) +
        '\n\ndef display(key):\n    return NAMES[key]\n')
    missed = pairs[~pairs.matched]
    columns = ['source_unit_id', 'waveform_r', 'fingerprint_r', 'rank_a', 'rank_b', 'match_score', 'half_r_min', 'dominant_delta']
    table = '| ' + ' | '.join(display(c) for c in columns) + ' |\n|' + '|'.join(['---'] * len(columns)) + '|\n'
    for _, row in missed.iterrows():
        table += '| ' + ' | '.join(str(row[c]) if c == 'source_unit_id' else ('Unavailable' if not np.isfinite(row[c]) else str(int(row[c])) if c in ('rank_a', 'rank_b', 'dominant_delta') else f'{row[c]:.3f}') for c in columns) + ' |\n'
    features = list(dataset.features.values())
    rates = count_range([f['sample_rate'] for f in features])
    samples = count_range([len(f['waveform']) for f in features])
    channels = count_range([f['waveform'].shape[1] for f in features])
    waveform_counts = count_range([f['n_waveform_spikes'] for f in features])
    half_counts = count_range([len(h) for f in features for h in f.get('half_waveform_spike_samples', [])])
    trial_counts = count_range([n for f in features for n in f['trial_counts'].values()])
    bins_per_condition = count_range([len(v) for f in features for v in f['fingerprints'].values()])
    bin_ms = count_range([1000 * f['fingerprint_bin_seconds'] for f in features])
    sound_durations = count_range([len(v) * f['fingerprint_bin_seconds'] for f in features for v in f['fingerprints'].values()], decimals=3)
    windows = sorted({(f['start_sample'], f.get('stop_sample'), f['sample_rate']) for f in features})
    window_text = '; '.join(f'[{a / fs:.3f}, {b / fs:.3f}) seconds' if b is not None else 'stop time unavailable' for a, b, fs in windows)
    split_rules = sorted({source.get('provenance', {}).get('split_method', 'Split-construction rule unavailable in saved provenance')
                          for source in dataset.metadata.get('sources', [])})
    split_rules = [rule.replace('whole BAPHY trials', 'whole experimental trials from the BAPHY experiment control system') + '; trial indices are divided into approximately equal contiguous groups, with cuts at trial starts' if 'whole BAPHY trials' in rule else rule for rule in split_rules]
    source_count = len(dataset.labels[['well', 'source_session', 'source_unit_id']].drop_duplicates())
    before = dataset.metadata['extraction_options']['before']
    shape = features[0]['waveform'].shape
    thresholds = result.provenance.get('threshold', 'unavailable')
    report = f'''# {display('title')}

{DEFINITIONS['reference']}

This report concerns {len(observed)} chronological observations from {source_count} original sorted units in {dataset.labels.source_session.nunique()} recording(s): {', '.join(sorted(dataset.labels.source_session.unique()))}. Dataset identifier: `{dataset.metadata['dataset_id']}`. Each original unit here has two outer observations; a source-identity pair is the pair of those observations. These are time segments of the same recording, not separate days.

Outer observation windows from recording start are {window_text}; [start, stop) includes the start and excludes the stop. Split rule: {'; '.join(split_rules) or 'unavailable in saved provenance'}. Each outer observation is divided again at its sample-window midpoint into early and late internal halves for independent waveform means. Thus an internal half is shorter than either observation being matched. Spikes are sampled without replacement separately for the full observation and its two internal halves; the full mean can share spikes with an internal mean, while the two internal halves cannot share spikes. Each full waveform mean uses {waveform_counts} snippets and each internal mean uses {half_counts} snippets. Each sound condition uses {trial_counts} complete trials per observation; trials crossing an outer boundary are excluded.

UnitMatch formed {evaluation['true_positive_pairs']} correct source-identity pairs and {evaluation['false_positive_pairs']} incorrect source-identity pairs, missing {evaluation['false_negative_pairs']} of {evaluation['n_known_positive_pairs']} available same-source pairs. {DEFINITIONS['directions']} Recorded tracking threshold: {thresholds}. {DEFINITIONS['intermediate']} This inspection fits no thresholds and changes no decisions.

## Timing and waveform quality

Saved preprocessing: {_preprocessing_text(dataset)} The reference subtraction removes the instantaneous voltage component shared across channels; it is not a temporal baseline subtraction.

A snippet contains {samples} samples on {channels} electrodes, sampled at {rates} Hz. Sample indices are zero-based: time zero is index {before}, the exported spike timestamp. For this archive the snippet spans {(0 - before) * 1000 / features[0]['sample_rate']:.3f} through {(shape[0] - 1 - before) * 1000 / features[0]['sample_rate']:.3f} ms, inclusive at the recorded sample centers. An edge extremum means the first or last sample, indices 0 or {shape[0] - 1}.

For timing statistics, the electrode with greatest peak-to-peak amplitude is chosen independently for each observation. Across {len(observed)} observations, its negative trough lies at sample {count_range(observed.trough_sample)}; its positive peak lies at {count_range(observed.peak_sample)}. The larger absolute extremum lies at {count_range(observed.dominant_sample)} ({count_range(observed.dominant_offset_ms, decimals=3)} ms from the exported timestamp); {int(observed.dominant_at_edge.sum())}/{len(observed)} such extrema touch a snippet edge. It is positive in {int(observed.dominant_polarity.eq('positive').sum())}/{len(observed)} observations, so negative-trough position alone is not a uniform timing marker.

Each observation contributes one internal-half waveform Pearson correlation: flatten all sample-by-electrode entries of each internal mean, subtract each vector's mean, and correlate the two vectors. Across these {len(observed)} correlations, the median is {observed.half_r.median():.3f}, range {count_range(observed.half_r, decimals=3)}. Repeating the calculation using only each observation's selected electrode gives median {observed.peak_channel_half_r.median():.3f}, range {count_range(observed.peak_channel_half_r, decimals=3)}; full-array correlations include weak/noisy electrodes. The absolute difference between the two internal extrema indices, measured on that same observation-selected electrode, is at most {observed.half_dominant_delta.max()} samples.

Across the {len(pairs)} source-identity pairs, the dominant-extremum displacement is at most {pairs.dominant_delta.max()} samples; for the {len(missed)} missed pairs it is at most {missed.dominant_delta.max()} samples. This is the absolute, unsigned difference of extrema sample indices on each observation's independently selected electrode; the two selected electrodes may differ. It measures timing displacement, not spatial distance. The figures use a separate, jointly selected electrode and preserve exported timestamps and amplitudes, with no realignment or amplitude normalization. Stable non-edge extrema and reproducible shapes argue against a gross, observation-dependent snippet timing failure; they cannot independently validate a common clock offset, original sorting quality, or a one-sample indexing convention. Biphasic waveforms can have a delayed positive peak even when their negative trough is aligned.

## {display('table_title')}

{table}

Full-array waveform Pearson correlation compares all sample-by-electrode entries of the two outer-observation means. Lower internal-half waveform Pearson correlation is the smaller of the two observations' internal early/late correlations described above. All correlations are dimensionless, with −1 indicating opposite patterns and +1 identical patterns up to offset and positive scale.

Stimulus response Pearson correlation uses mean firing rates (spikes/second) from complete trials of identical sound conditions, with channel/ear, silence and attenuation kept distinct. Each condition contributes {bins_per_condition} consecutive {bin_ms}-ms bins, starting at its nominal sound onset and ending at {sound_durations} seconds after onset (no prestimulus baseline subtraction or acoustic-latency correction). Vectors concatenate matching conditions and bin positions before Pearson correlation; only conditions with at least one complete trial in both observations enter this descriptive diagnostic. This differs from any stricter repeat-count gate used by a matcher.

Ranks are descending correlation ranks of the true counterpart against every measured opposite-segment candidate in the same original recording, without a spatial-distance restriction: {count_range(pairs.candidates_a)} candidates per direction. Rank 1 is highest and allows ties within absolute correlation tolerance 1e-12. “First to second” ranks the second observation among candidates for the first; “second to first” reverses that candidate pool. Among the {len(missed)} misses, {int(missed.mutual_best.sum())} are mutual rank 1, of which {int(missed.mutual_unique_best.sum())} are uniquely first in both directions. Each source-identity pair has {count_range(pairs.n_conditions)} common sound conditions and {count_range(pairs.n_bins)} concatenated time bins. These rankings describe evidence only; they do not demonstrate that any combined matching algorithm would recover these pairs without additional errors.

![Missed-pair waveform overlays](missed_waveforms.png)

The figure selects one shared electrode per pair by the largest peak-to-peak amplitude of the average of both observation means. Electrode indices are zero-based channel indices in the saved array. Its title correlations use the entire multichannel arrays, not just the displayed electrode traces. Blue and orange show the first and second recording segments respectively.

Input dataset: `{Path(dataset_path).resolve()}`. Matching result: `{Path(result_path).resolve()}`. Detailed observation and pair values are in `waveform_quality.csv` and `identity_pairs.csv`.
'''
    (target / 'report.md').write_text(report)
    (target / 'METHODS.md').write_text(f'''# Waveform diagnostics — figure & methods notes

- **Output folder:** `{target}`
- **Artifacts:** `report.md`, `waveform_quality.csv`, `identity_pairs.csv`, `missed_waveforms.png`, `names.py`
- **Generated by:** `nems_lbhb/unit_match/diagnostic_report.py:write_report()` · **Updated:** `2026-09-18`

## Goal
Check gross timing and reproducibility of extracted temporal waveforms, and describe UnitMatch's missed source-identity pairs. The report does not tune thresholds or modify assignments.

## Data lineage
- `features.npz` + `features.json` ← `extract_benchmark.py:extract_manifest()` and `features.py:temporal_split_unit_features()`; acquisition scaling, shared high-pass and median reference, independent temporal means and response fingerprints (source settings in dataset manifest).
- `reference_labels.csv` ← `benchmark.py:build_dataset()`; original sorted source-unit labels, used only for diagnostic grouping.
- `result.json` ← `runners.py:run_backend_dataset()`; official UnitMatch scoring and native assignment, with recorded source revision.
- Diagnostic CSV files ← `diagnostic_report.py:diagnose()`; extrema and full-waveform correlations, exhaustive within-recording opposite-segment fingerprint correlations and tie-aware reciprocal ranks.

## Panels
### Panels A onward: missed source-identity pair overlays
- **Made by:** `_plot_misses()` in `nems_lbhb/unit_match/diagnostic_report.py`.
- **Input:** compact waveform means for source-identity pairs absent from result tracks.
- **Steps:** select one shared electrode by largest peak-to-peak amplitude of the two-observation mean; overlay each original mean without realignment or scaling; place zero at the exported timestamp using the measured sample rate. Plot titles show full multichannel Pearson correlations, not the displayed electrode alone. Internal reliability halves split each outer sample interval at its midpoint in features.py:temporal_split_unit_features(); the full-observation mean is sampled separately. UnitMatch directional comparisons and native tracking are defined in external_backends.py:_matrix_pairs() and _um_result(), using the pinned upstream metric_functions.py and assign_unique_id.py.
- **Shows:** absolute-time morphology and amplitude consistency for the missed pairs.

## Caveats
Source identity assumes the original sorter was correct. Two time segments from one recording do not reproduce probe reinsertion. A common timestamp offset cannot be inferred from waveform averages alone. Full multichannel correlations include low-amplitude electrodes. Response rankings use observed test features descriptively and are not a fitted matching method.
''')
    (target / 'missed_waveforms_legend.md').write_text('**Waveform overlays reveal whether missed source-identity pairs retain consistent timing and shape.**\n' + '\n'.join(f'**({chr(65 + i)})** Original sorted unit {pair.source_unit_id}: first (blue) and second (orange) recording-segment means on the shared electrode selected by largest peak-to-peak amplitude of their average.' for i, pair in enumerate(missed.itertuples())) + '\nTime is relative to the exported spike timestamp; voltage is filtered, referenced microvolts. Panel titles report Pearson correlation over all electrodes and samples, not the single displayed trace. Each line averages ' + waveform_counts + ' spike snippets. No temporal realignment or amplitude normalization is applied.\n')
    _plot_misses(dataset, pairs, target)
    return target
# [AGENT EDIT END]
