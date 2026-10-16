"""Convert complete two-part temporal summaries into one real-day observation.

This reads only archives, source manifests, small spike/geometry files, and raw
file metadata. It never opens raw voltage. Source cluster labels are used only
for within-recording assembly and never establish cross-day identity.
"""
import copy
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from .benchmark import load_dataset
from .extract.full_day import _pool_waveforms, _spike_features, pool_day_row
from .extract.unit_features import _checked_sample_spikes, waveform_sample_selection
from .waveform_preprocessing import SharedWaveformPreprocessor


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value, manifest):
    path = Path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def reuse_day_summary(archive_path, source_manifest_path, spec, *, session_date,
                      analysis_window, extraction_options):
    """Return a validated day summary, pooled by `extract/full_day.py`.

    The source manifest must be the original, unmodified preparation input whose
    hash was recorded during extraction. Complete ordinary archives only: no
    omission views, collections, missing source units, or partially covered days.
    """
    date.fromisoformat(session_date)
    archive, manifest_path = Path(archive_path).resolve(), Path(source_manifest_path).resolve()
    manifest = json.loads((archive / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('kind') is not None:
        raise ValueError('Reuse requires a complete ordinary archive, not a collection or omission view')
    dataset = load_dataset(archive)
    if dataset.exclusions or len(dataset.metadata['sources']) != 1:
        raise ValueError('Reuse requires one complete source with no excluded or omitted observations')
    source_manifest_hash = _hash(manifest_path)
    if dataset.metadata.get('extraction_manifest_sha256') != source_manifest_hash:
        raise ValueError('Source preparation manifest hash changed or is unavailable')
    original = json.loads(manifest_path.read_text())
    if len(original.get('recordings', [])) != 1 or original['recordings'][0] != spec:
        raise ValueError('Reuse requires the unchanged original single-source manifest specification')
    source = dataset.metadata['sources'][0]
    session, well = spec['source_session'], spec['well']
    if source['source_session'] != session or source['well'] != well or source['preprocessing'] != spec['preprocessing']:
        raise ValueError('Original recording identity or preprocessing differs')
    opts = dict(original.get('extraction_options', {}))
    opts.update(before=opts.get('before', 41), after=opts.get('after', 41))
    requested = dict(extraction_options)
    requested.update(before=requested.get('before', 41), after=requested.get('after', 41))
    if opts != dataset.metadata['extraction_options'] or requested != opts:
        raise ValueError('Reuse extraction settings must match the archived raw-derived summaries')
    provenance = source['provenance']
    clock = spec.get('clock_verification', {})
    if (clock.get('sample_index_origin') != 'zero_based_voltage_row' or not clock.get('method')
            or provenance.get('clock_verification') != clock or provenance.get('n_samples') != spec['n_samples']):
        raise ValueError('Reuse requires the original verified acquisition sample clock')
    if isinstance(spec['raw_voltage'], (list, tuple)):
        raise ValueError('multiday reuse reads one raw file; this day has raw segments')
    raw = _path(spec['raw_voltage'], manifest_path)
    stat = raw.stat()  # Deliberately no raw open, decompression, or voltage read.
    if (str(raw) != provenance.get('raw_source') or stat.st_size != provenance.get('raw_size_bytes')
            or stat.st_mtime_ns != provenance.get('raw_mtime_ns')):
        raise ValueError('Raw-source path, size, or modification time changed since extraction')
    for key in ('voltage_scale', 'baseline_samples'):
        if provenance.get(key) != spec.get(key, 1. if key == 'voltage_scale' else 0):
            raise ValueError('Voltage scaling or baseline preprocessing changed')
    paths = {key: _path(spec[key + '_file'], manifest_path) for key in ('spikes', 'geometry')}
    for key, path in paths.items():
        if str(path) != provenance.get(key + '_source') or _hash(path) != provenance.get(key + '_sha256'):
            raise ValueError(f'Original {key} source changed since extraction')
    geometry = np.load(paths['geometry'], allow_pickle=False)
    shared = spec.get('waveform_preprocessing')
    if shared:
        if (shared.get('profile') != 'shared_unwhitened_v1' or spec.get('common_median', False)
                or spec.get('baseline_samples', 0)):
            raise ValueError('Invalid shared waveform preprocessing declaration')
        processing = SharedWaveformPreprocessor(spec['sample_rate'], len(geometry),
            **{k: shared[k] for k in ('highpass_hz', 'filter_order', 'padding_seconds', 'reference_channels') if k in shared}).metadata()
        processing.update(chunk_seconds=1., output_dtype='float32')
        archived_processing = dict(provenance.get('waveform_preprocessing') or {})
        # Disk versus in-memory intermediate storage is operational metadata.
        # Keep every other field, including unknown future transforms, checked.
        archived_processing.pop('storage', None)
        if processing != archived_processing:
            raise ValueError('Shared voltage preprocessing differs from archived raw means')
    elif provenance.get('common_median') != spec.get('common_median', False):
        raise ValueError('Median reference preprocessing changed')
    units = tuple(spec['single_unit_ids'])
    labels = dataset.labels
    if (len(units) != len(set(units)) or set(labels.source_unit_id) != set(units)
            or set(labels.source_session) != {session} or set(labels.well) != {well}
            or not labels.groupby('source_unit_id').size().eq(2).all()):
        raise ValueError('Reuse needs exactly two eligible parts for every declared original single unit')
    windows = np.asarray(spec['windows'])
    if (windows.shape != (2, 2) or not np.isfinite(windows).all()
            or not np.equal(windows, np.floor(windows)).all() or (windows[:, 1] <= windows[:, 0]).any()
            or windows[0, 1] != windows[1, 0] or [int(windows[0, 0]), int(windows[1, 1])] != list(analysis_window)):
        raise ValueError('Reuse needs exactly two contiguous parts covering the complete analysis window')
    trials = spec.get('stimuli', [])
    if any(t['start'] < windows[0, 1] < t['stop'] for t in trials):
        raise ValueError('A stimulus trial crosses the archived boundary; regenerate compatible trial-aligned summaries')
    table, features = [], {}
    archive_hash = _hash(archive / 'manifest.json')
    with np.load(paths['spikes'], allow_pickle=False) as saved:
        for unit in units:
            selected_labels = labels[labels.source_unit_id == unit]
            rows = sorted([dataset.features[c] for c in selected_labels.cellid], key=lambda row: row['start_sample'])
            spikes = _checked_sample_spikes(saved[unit])
            if len(spikes) and spikes[-1] >= spec['n_samples']:
                raise ValueError('Source spike samples exceed physical recording bounds')
            full = spikes[(spikes >= analysis_window[0]) & (spikes < analysis_window[1])]
            if not np.array_equal(np.concatenate([r['spike_samples'] for r in rows]), full):
                raise ValueError('Archived parts do not preserve the complete original spike train')
            for index, (row, bounds) in enumerate(zip(rows, windows)):
                if ([row['start_sample'], row['stop_sample']] != bounds.tolist()
                        or row['sample_rate'] != spec['sample_rate'] or row['preprocessing'] != spec['preprocessing']):
                    raise ValueError('Archived part windows, sample rate or preprocessing differ')
                selection, eligible = waveform_sample_selection(spikes, bounds, opts['before'], opts['after'],
                    opts.get('max_spikes', 1000), opts.get('seed', 0) + index * 3)
                if (not np.array_equal(selection, row['waveform_spike_samples'])
                        or row['n_waveform_spikes'] != len(selection) or row['n_eligible_waveform_spikes'] != eligible):
                    raise ValueError('Archived waveform selections or counts do not match the original raw extraction plan')
                if np.asarray(row['waveform']).shape != (opts['before'] + opts['after'], len(geometry)):
                    raise ValueError('Archived waveform shape differs from source geometry or sample crop')
            selections = [np.asarray(r['waveform_spike_samples']).copy() for r in rows]
            if np.intersect1d(*selections).size:
                raise ValueError('Independent waveform halves cannot reuse selected spike indices')
            for cid in selected_labels.cellid:
                pseudo_session = dataset.units.set_index('cellid').loc[cid, 'session_id']
                if not np.array_equal(dataset.geometry[pseudo_session], geometry):
                    raise ValueError('Archived geometry differs from measured source channel order')
            feature = pool_day_row(rows, spikes, spec, analysis_window=analysis_window, windows=windows,
                                   max_spikes=opts.get('max_spikes', 1000))
            cid = unit if unit.startswith(session + '-') else session + '-' + unit
            table.append(dict(cellid=cid, session_id=session, siteid=session, well=well,
                              session_date=session_date, recording_group=well, su_type='SU'))
            features[cid] = feature
    clean_provenance = copy.deepcopy(provenance)
    clean_provenance.update(extraction_manifest=str(manifest_path), extraction_manifest_sha256=source_manifest_hash,
        analysis_window=list(analysis_window), summary_reuse=dict(archive=str(archive), archive_manifest_sha256=archive_hash,
            archive_file_sha256=dict(manifest['sha256']), half_windows=windows.astype(int).tolist(),
            fingerprint_half_definition='Each condition chronological trials split at floor(n_trials/2), matching fresh extraction; independent of waveform outer windows',
            voltage_read=False, within_recording_assembly='Original source cluster IDs, discarded after real-day assembly',
            sampling='Independent outer means; full mean/SEM pooled over their selected snippets, not a fresh full-day sample'))
    info = dict(source_session=session, well=well, session_date=session_date,
        analysis_window=list(analysis_window), physical_n_samples=spec['n_samples'],
        preprocessing=spec['preprocessing'], provenance=clean_provenance)
    return dict(table=table, features=features, geometry={session: geometry.copy()}, sources=[info], exclusions=[])
