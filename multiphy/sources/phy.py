# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the public phy reader: single units, spikes, geometry, channel map and unwhitened templates from a KiloSort or phy folder (multiphy project folder Task 7; fix round 1: channel_map.npy) | date: 2026-09-25]
"""The ``phy`` reader: a KiloSort or phy output folder serves the ``sort`` role.

``options['sort_dir']`` holds ``params.py``, ``spike_times.npy``, ``spike_clusters.npy``,
``channel_positions.npy`` and ``cluster_group.tsv`` (else ``cluster_KSLabel.tsv``). A cluster
labelled ``good`` is a single unit with id ``<day>:<cluster>``. When ``templates.npy``,
``whitening_mat_inv.npy`` and ``spike_templates.npy`` are all present, a unit's template is the
spike-count-weighted mean of the unwhitened templates its spikes carry. ``channel_map.npy``,
when present, gives the raw-file column of each ``channel_positions.npy`` row; it reaches
``DaySource.channel_map`` for the ``binary`` reader to check. ``params.py`` is parsed, never
executed. The sample count is the last spike + 1 until the raw or events reader
sets it. ``raw_proposal`` offers ``params.py``'s ``dat_path`` to the ``binary`` reader, and
``raw_layout`` gives any raw file the dtype, offset and channel count ``params.py`` declares.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Optional

import numpy as np

from .base import DaySource

REQUIRED_FILES = {'params': 'params.py', 'spike_times': 'spike_times.npy',
                  'spike_clusters': 'spike_clusters.npy', 'channel_positions': 'channel_positions.npy'}
LABEL_FILES = ('cluster_group.tsv', 'cluster_KSLabel.tsv')  # the first present one decides
LABEL_COLUMNS = {'group': 'cluster_group.tsv', 'KSLabel': 'cluster_KSLabel.tsv'}
CHANNEL_MAP_FILE = 'channel_map.npy'  # geometry row -> raw-file column; optional
TEMPLATE_FILES = {'templates': 'templates.npy', 'whitening_mat_inv': 'whitening_mat_inv.npy',
                  'spike_templates': 'spike_templates.npy'}
CLOCK_METHOD = 'phy spike_times.npy sample indices'


def _missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(f'phy: {path} is missing.')


def _not_literal() -> ValueError:
    return ValueError('phy: params.py may hold only literal assignments.')


def parse_params(path: Path) -> dict:
    """``params.py`` as name -> value; every statement must assign a literal to one name. Nothing runs."""
    path = Path(path)
    if not path.is_file():
        raise _missing(path)
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except SyntaxError:
        raise _not_literal() from None
    params: dict = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            raise _not_literal()
        try:
            params[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            raise _not_literal() from None
    return params


def _param(params: dict, key: str):
    if key not in params:
        raise ValueError(f'phy: params.py has no {key}.')
    return params[key]


def _labels(path: Path) -> tuple:
    """(cluster -> label, the file its label column names); column ``group`` wins over ``KSLabel``."""
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        columns = [c.strip() for c in (reader.fieldnames or [])]
        reader.fieldnames = columns
        rows = list(reader)
    column = next((c for c in LABEL_COLUMNS if c in columns), None)
    if column is None or 'cluster_id' not in columns:
        raise ValueError(f'phy: {path} needs a cluster_id column and a group or KSLabel column.')
    labels = {int(row['cluster_id']): (row[column] or '').strip() for row in rows}
    return labels, LABEL_COLUMNS[column]


def _channel_map(path: Path, n_channels: int) -> np.ndarray:
    """``channel_map.npy`` as int64, one raw-file column per geometry row; KiloSort's ``(n, 1)`` reads as ``(n,)``."""
    channel_map = np.load(path).ravel()
    if not np.issubdtype(channel_map.dtype, np.integer):
        raise ValueError('phy: channel_map.npy must hold integer channel indices.')
    if len(channel_map) != n_channels:
        raise ValueError(f'phy: channel_map.npy holds {len(channel_map)} channels; '
                         f'channel_positions.npy has {n_channels}.')
    return channel_map.astype(np.int64)


def _unit_templates(files: dict, index_of: dict, n_spikes: int, n_channels: int) -> dict:
    """Unit id -> float32 ``(n_template_samples, n_channels)``: spike-count-weighted mean of unwhitened templates."""
    templates = np.load(files['templates'])
    whitening_inv = np.load(files['whitening_mat_inv'])
    spike_templates = np.load(files['spike_templates']).ravel()
    if len(spike_templates) != n_spikes:
        raise ValueError(f'phy: spike_templates.npy holds {len(spike_templates)} spikes; '
                         f'spike_times.npy holds {n_spikes}.')
    if templates.ndim != 3 or templates.shape[2] != n_channels:
        raise ValueError(f'phy: templates.npy has shape {templates.shape}; '
                         f'channel_positions.npy has {n_channels} channels.')
    out = {}
    for unit, index in index_of.items():
        ids, counts = np.unique(spike_templates[index], return_counts=True)
        unwhitened = templates[ids].astype(np.float64) @ whitening_inv  # templates[i] @ whitening_mat_inv
        out[unit] = (np.tensordot(counts, unwhitened, axes=1) / counts.sum()).astype(np.float32)
    return out


class PhyReader:
    """Reads the ``sort`` role from the phy folder ``options['sort_dir']``."""
    id = 'phy'
    version = 1
    roles = frozenset({'sort'})

    def files(self, options: dict) -> dict:
        sort_dir = Path(options['sort_dir'])
        files = {}
        for key, name in REQUIRED_FILES.items():
            path = sort_dir / name
            if not path.is_file():
                raise _missing(path)
            files[key] = path
        labels = [sort_dir / name for name in LABEL_FILES]
        present = [path for path in labels if path.is_file()]
        if not present:
            raise _missing(labels[0])
        files['cluster_labels'] = present[0]
        if (sort_dir / CHANNEL_MAP_FILE).is_file():
            files['channel_map'] = sort_dir / CHANNEL_MAP_FILE
        template_paths = {key: sort_dir / name for key, name in TEMPLATE_FILES.items()}
        if all(path.is_file() for path in template_paths.values()):
            files.update(template_paths)
        return files

    def read(self, files: dict, options: dict, day: Optional[DaySource] = None) -> DaySource:
        params = parse_params(files['params'])
        sample_rate = float(_param(params, 'sample_rate'))
        times = np.load(files['spike_times']).ravel()
        clusters = np.load(files['spike_clusters']).ravel()
        if len(times) != len(clusters):
            raise ValueError(f'phy: spike_times.npy holds {len(times)} spikes; '
                             f'spike_clusters.npy holds {len(clusters)}.')
        if len(times) == 0:
            raise ValueError('phy: spike_times.npy holds no spikes.')
        geometry = np.asarray(np.load(files['channel_positions']), dtype=float)
        if geometry.ndim != 2 or geometry.shape[1] != 2:
            raise ValueError(f'phy: channel_positions.npy must be (n_channels, 2), not {geometry.shape}.')
        channel_map = _channel_map(files['channel_map'], geometry.shape[0]) if 'channel_map' in files else None
        labels, labels_from = _labels(Path(files['cluster_labels']))

        good = sorted(cluster for cluster, label in labels.items() if label == 'good')
        order = np.argsort(clusters, kind='stable')
        ordered = clusters[order]
        starts = np.searchsorted(ordered, good, side='left')
        stops = np.searchsorted(ordered, good, side='right')
        spikes, index_of = {}, {}
        for cluster, start, stop in zip(good, starts, stops):
            if start == stop:  # a label left behind by a merge or split; the cluster has no spikes
                continue
            unit = f"{options['day']}:{cluster}"
            index_of[unit] = order[start:stop]
            spikes[unit] = np.sort(times[index_of[unit]].astype(np.int64))

        templates = None
        if all(key in files for key in TEMPLATE_FILES):
            templates = _unit_templates(files, index_of, len(times), geometry.shape[0])

        n_samples = int(times.max()) + 1
        return DaySource(
            sample_rate=sample_rate, n_samples=n_samples, analysis_window=(0, n_samples),
            spikes=spikes, su_type={unit: 'SU' for unit in spikes}, geometry=geometry,
            clock={'sample_index_origin': 'zero_based_voltage_row', 'method': CLOCK_METHOD,
                   'n_samples_from': 'last_spike'},
            provenance={'reader': f'{self.id}@{self.version}', 'su_labels_from': labels_from, 'params': params},
            templates=templates, channel_map=channel_map,
        )

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: final review fix round 1: a raw file's layout comes from params.py whether or not it names a dat_path; an undeclared channel count is refused, never taken from the geometry | date: 2026-09-25]
    def raw_layout(self, options: dict) -> dict:
        """The raw file's layout as ``params.py`` declares it: ``{'dtype', 'offset', 'n_channels', 'sample_rate'}``.

        Read whether or not ``params.py`` names a ``dat_path``. A flat raw file has no header to
        count its channels, so ``n_channels_dat`` is required: without it, ``ValueError``.
        ``dtype`` defaults to ``int16`` and ``offset`` to 0, as KiloSort writes them.
        """
        params = parse_params(Path(options['sort_dir']) / 'params.py')
        if 'n_channels_dat' not in params:
            raise ValueError('phy: params.py has no n_channels_dat; a raw file needs its declared channel count.')
        return {'dtype': np.dtype(params.get('dtype', 'int16')).str, 'offset': int(params.get('offset', 0)),
                'n_channels': int(params['n_channels_dat']), 'sample_rate': float(_param(params, 'sample_rate'))}
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: post-run fix (final review 2): finding params.py's dat_path reads no layout, so a day without a raw file needs no n_channels_dat | date: 2026-09-26]
    def dat_path(self, options: dict) -> Optional[str]:
        """The file ``params.py`` names as ``dat_path`` (a relative one from ``sort_dir``), or ``None``.

        Reads no layout: a day that uses no raw file needs no ``n_channels_dat``.
        """
        sort_dir = Path(options['sort_dir'])
        dat_path = parse_params(sort_dir / 'params.py').get('dat_path')
        if isinstance(dat_path, (list, tuple)):
            dat_path = dat_path[0] if dat_path else None
        if not dat_path:
            return None
        path = Path(dat_path)
        return str(path if path.is_absolute() else sort_dir / path)

    def raw_proposal(self, options: dict) -> Optional[dict]:
        """``binary`` parameters for ``params.py``'s ``dat_path``, or ``None`` when it names no file."""
        path = self.dat_path(options)
        return None if path is None else {'path': path, **self.raw_layout(options)}
    # [AGENT EDIT END]


READER = PhyReader()
# [AGENT EDIT END]
