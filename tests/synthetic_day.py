"""A synthetic phy day on disk, and the schema 1 inputs the old extraction path reads.

``write_day(folder, day)`` writes ``folder / day`` in the layout ``init --from`` reads:
a phy ``sort/`` folder, ``events.csv``, a flat int16 ``raw.bin`` and ``date.txt``.
Eight channels, 30 kHz, 600000 samples; clusters 0-2 are ``good``, cluster 3 is
``mua``, and ``weak_unit=True`` adds a ``good`` cluster 4 with too few late spikes.
The raw file is Gaussian noise plus each cluster's unwhitened template at its spikes.
``write_legacy_inputs`` writes ``spikes.npz``, ``geometry.npy`` and a schema 1
``extraction_manifest.json`` so ``extract_benchmark.extract_manifest`` reads the same day.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import shutil
from typing import Optional

import numpy as np

from multiphy.waveform_preprocessing import SharedWaveformPreprocessor

OPTIONS = {'before': 41, 'after': 41, 'max_spikes': 100, 'min_spikes': 20, 'seed': 0}

N_CHANNELS = 8
SAMPLE_RATE = 30000.0
N_SAMPLES = 600000
UV_PER_BIT = 0.195
GEOMETRY = np.column_stack(([0, 32] * 4, [0, 0, 20, 20, 40, 40, 60, 60])).astype(float)
TEMPLATE_CHANNELS = (1, 3, 5, 6, 2)     # template k peaks on channel TEMPLATE_CHANNELS[k]
TEMPLATE_SAMPLES = 82
N_TRIALS = 10


@dataclass(frozen=True)
class SyntheticDay:
    """Paths and ground truth of one synthetic day; samples are zero-based raw rows."""
    day: str
    root: Path
    sort_dir: Path
    raw_path: Path
    gz_path: Optional[Path]
    events_path: Optional[Path]
    date: str
    units: list
    spikes: dict
    templates: dict
    geometry: np.ndarray
    stimuli: list
    baselines: np.ndarray
    n_samples: int
    uv_per_bit: float


def _unique_samples(rng, n, low, high):
    """``n`` distinct sorted int64 samples, uniform in ``[low, high)``."""
    return np.sort(rng.choice(high - low, size=n, replace=False) + low).astype(np.int64)


def _unwhitened_templates():
    t = np.arange(TEMPLATE_SAMPLES)
    templates = np.empty((len(TEMPLATE_CHANNELS), TEMPLATE_SAMPLES, N_CHANNELS))
    for k, channel in enumerate(TEMPLATE_CHANNELS):
        d_k = np.linalg.norm(GEOMETRY - GEOMETRY[channel], axis=1)
        templates[k] = (-(80 + 20 * k) * np.exp(-((t - 41) / 3) ** 2)[:, None]
                        * np.exp(-d_k ** 2 / 800)[None, :])
    return templates


def _write_tsv(path, column, labels):
    rows = [f'cluster_id\t{column}'] + [f'{cluster}\t{label}' for cluster, label in labels.items()]
    path.write_text('\n'.join(rows) + '\n')


def write_day(folder: Path, day: str, *, seed: int = 0, date: str = '2026-01-01',
              weak_unit: bool = False, gzip_raw: bool = False,
              with_templates: bool = True, conditions: tuple = ('A', 'B'), events: bool = True) -> SyntheticDay:
    """Write ``folder / day`` and return its paths and ground truth; trial ``i`` plays ``conditions[i % len(conditions)]``.

    ``events=False`` writes no ``events.csv`` and returns ``events_path=None``, no stimuli and
    no baselines; the spikes, raw file and every random draw are the same as with events.
    """
    folder = Path(folder).resolve()
    root = folder / day
    sort_dir = root / 'sort'
    sort_dir.mkdir(parents=True)
    raw_path, events_path = root / 'raw.bin', root / 'events.csv'

    rng = np.random.default_rng(seed)
    cluster_spikes = {cluster: _unique_samples(rng, n, 100, 599900)
                      for cluster, n in ((0, 200), (1, 200), (2, 200), (3, 150))}
    labels = {0: 'good', 1: 'good', 2: 'good', 3: 'mua'}
    if weak_unit:
        cluster_spikes[4] = np.sort(np.concatenate((_unique_samples(rng, 30, 100, 300000),
                                                    _unique_samples(rng, 5, 300000, 596900))))
        labels[4] = 'good'

    times = np.concatenate(list(cluster_spikes.values()))
    clusters = np.concatenate([np.full(len(s), c) for c, s in cluster_spikes.items()])
    order = np.lexsort((clusters, times))
    np.save(sort_dir / 'spike_times.npy', times[order].astype(np.uint64))
    np.save(sort_dir / 'spike_clusters.npy', clusters[order].astype(np.int32))
    np.save(sort_dir / 'channel_positions.npy', GEOMETRY)
    _write_tsv(sort_dir / 'cluster_group.tsv', 'group', labels)
    _write_tsv(sort_dir / 'cluster_KSLabel.tsv', 'KSLabel', labels)

    unwhitened = _unwhitened_templates()
    if with_templates:
        whitening = np.diag(1 / (1 + 0.1 * np.arange(N_CHANNELS)))
        np.save(sort_dir / 'templates.npy', (unwhitened @ whitening).astype(np.float32))
        np.save(sort_dir / 'whitening_mat_inv.npy', np.linalg.inv(whitening))
        np.save(sort_dir / 'spike_templates.npy', clusters[order].astype(np.uint32))

    (sort_dir / 'params.py').write_text(
        f'dat_path = {str(raw_path)!r}\n'
        f'n_channels_dat = {N_CHANNELS}\n'
        "dtype = 'int16'\n"
        'offset = 0\n'
        f'sample_rate = {SAMPLE_RATE!r}\n'
        'hp_filtered = False\n')

    voltage = rng.normal(0, 5, (N_SAMPLES, N_CHANNELS))
    half = TEMPLATE_SAMPLES // 2
    for cluster, samples in cluster_spikes.items():
        for s in samples:
            voltage[s - half:s + half] += unwhitened[cluster]
    raw = np.ascontiguousarray(np.round(voltage / UV_PER_BIT).astype('<i2'))
    raw_path.write_bytes(raw.tobytes(order='C'))
    gz_path = None
    if gzip_raw:
        gz_path = folder / f'{day}_raw.bin.gz'
        with open(raw_path, 'rb') as source, gzip.open(gz_path, 'wb') as target:
            shutil.copyfileobj(source, target)

    lines, stimuli, baselines = ['kind,start,stop,trial,condition'], [], []
    for i in range(N_TRIALS):
        start = 60000 * i
        condition = conditions[i % len(conditions)]
        lines.append(f'trial,{start},{start + 57000},{i + 1},')
        lines.append(f'baseline,{start},{start + 6000},,')
        lines.append(f'stimulus,{start + 6000},{start + 21000},{i + 1},{condition}')
        stimuli.append({'stimulus': condition, 'start': start + 6000, 'stop': start + 21000, 'trial': i + 1})
        baselines.append([start, start + 6000])
    if events:
        events_path.write_text('\n'.join(lines) + '\n')
    else:
        events_path, stimuli, baselines = None, [], np.empty((0, 2), np.int64)
    (root / 'date.txt').write_text(date + '\n')

    spikes = {f'{day}:{c}': s for c, s in cluster_spikes.items() if labels[c] == 'good'}
    return SyntheticDay(day=day, root=root, sort_dir=sort_dir, raw_path=raw_path, gz_path=gz_path,
                        events_path=events_path, date=date, units=sorted(spikes), spikes=spikes,
                        templates={k: unwhitened[k].copy() for k in range(len(unwhitened))},
                        geometry=GEOMETRY.copy(), stimuli=stimuli,
                        baselines=np.asarray(baselines, dtype=np.int64),
                        n_samples=N_SAMPLES, uv_per_bit=UV_PER_BIT)


def extracted_project(folder: Path, days: tuple = ('D1', 'D2'), *, project: str = 'P',
                      conditions: Optional[dict] = None, tiers: Optional[dict] = None,
                      weak_units: tuple = (), no_event_days: tuple = ()) -> Path:
    """Write, add and extract each of ``days`` into project ``project`` under ``folder / 'lab'``; return the project folder.

    Day ``i`` has seed ``i`` and date ``2026-01-<i + 1>``; ``conditions`` and ``tiers`` map a day to
    its trial conditions and extraction tier, and a day in ``weak_units`` gets the weak unit. A day
    in ``no_event_days`` is written without events and added with ``events=None``.
    """
    from multiphy.launch import commands

    folder = Path(folder)
    root = folder / 'lab'
    for i, day in enumerate(days):
        sd = write_day(folder / 'in', day, seed=i, date=f'2026-01-{i + 1:02d}', weak_unit=day in weak_units,
                       conditions=(conditions or {}).get(day, ('A', 'B')), events=day not in no_event_days)
        commands.init_day(root, project, day=day, sort=str(sd.sort_dir), date=sd.date,
                          events=None if sd.events_path is None else str(sd.events_path),
                          raw=str(sd.raw_path), uv_per_bit=sd.uv_per_bit)
        commands.extract(root, project, days=[day], tier=(tiers or {}).get(day), scratch_root=folder / 'scratch')
    return root / project


def write_legacy_inputs(day: SyntheticDay, out_dir: Path, *, dataset_id: str, well: str,
                        partition: str = 'test') -> Path:
    """Write the schema 1 extraction inputs for ``day`` into ``out_dir``; return the manifest path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / 'spikes.npz', **day.spikes)
    np.save(out_dir / 'geometry.npy', day.geometry)
    recording = {
        'source_session': day.day,
        'well': well,
        'partition': partition,
        'raw_voltage': str(day.raw_path),
        'spikes_file': 'spikes.npz',
        'geometry_file': 'geometry.npy',
        'single_unit_ids': list(day.units),
        'sample_rate': SAMPLE_RATE,
        'n_samples': day.n_samples,
        'windows': [[0, 300000], [300000, 597000]],
        'analysis_window': [0, 597000],
        'dtype': '<i2',
        'voltage_scale': day.uv_per_bit,
        'waveform_preprocessing': SharedWaveformPreprocessor(SAMPLE_RATE, N_CHANNELS).metadata(),
        'preprocessing': 'synthetic legacy day',
        'clock_verification': {'sample_index_origin': 'zero_based_voltage_row',
                               'method': 'synthetic sample clock'},
        'provenance': {},
        'stimuli': [dict(s) for s in day.stimuli],
        'baseline_windows': day.baselines.tolist(),
    }
    manifest = {'schema_version': 1, 'dataset_id': dataset_id,
                'extraction_options': dict(OPTIONS), 'recordings': [recording]}
    path = out_dir / 'extraction_manifest.json'
    path.write_text(json.dumps(manifest, indent=2))
    return path
