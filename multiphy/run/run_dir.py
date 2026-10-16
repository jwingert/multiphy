"""The files a run leaves behind, and how a result is read back.

Reading and writing use format 2 only: ``score``, ``track_kind`` and ``tracking_rule``
beside the run's units, pairs, tracks and provenance. Any other format or schema
version is refused on read.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from ..contract.provenance import Provenance, payload_digest
from ..contract.results import MatchResult, ScoreBlock, TrackKind

FORMAT = 'nems_unit_match_result'


def write_json(path, value, *, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read_json_object(path):
    if path is None:
        return None
    with Path(path).expanduser().open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def save_result(result, path, *, overwrite=False):
    """Atomic publish of a validated result; never overwrite by default."""
    result.validate()
    payload = result.payload()
    document = {'format': FORMAT, 'schema_version': 2, 'sha256': payload_digest(payload), 'result': payload}
    serialized = json.dumps(document, indent=2, allow_nan=False) + '\n'
    path = Path(path).expanduser().absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _tables(payload, path):
    payload = dict(payload)
    for key in ('units', 'pairs'):
        table = payload[key]
        if not isinstance(table, dict) or set(table) != {'columns', 'records'}:
            raise ValueError(f'{path}: invalid {key} table structure')
        columns, rows = table['columns'], table['records']
        if (not isinstance(columns, list) or len(columns) != len(set(columns))
                or not isinstance(rows, list) or any(not isinstance(r, dict) or set(r) != set(columns) for r in rows)):
            raise ValueError(f'{path}: invalid {key} table')
        payload[key] = pd.DataFrame(rows, columns=columns)
    if 'match_score' in payload['pairs']:
        payload['pairs']['match_score'] = pd.to_numeric(payload['pairs'].match_score, errors='raise')
    payload['tracks'] = tuple(tuple(t) for t in payload['tracks'])
    payload['unmatched'] = tuple(payload['unmatched'])
    payload['excluded_units'] = tuple(payload['excluded_units'])
    return payload


def load_result(path):
    """Read a format 2 result and return a validated MatchResult."""
    def reject_nonfinite(token):
        raise ValueError(f'{path}: nonstandard JSON number {token!r}; use null for missing data')

    path = Path(path).expanduser()
    with path.open(encoding='utf-8') as stream:
        document = json.load(stream, parse_constant=reject_nonfinite)
    if not isinstance(document, dict) or document.get('format') != FORMAT or document.get('schema_version') != 2:
        raise ValueError(f'{path}: unsupported matching result format or schema version')
    payload = document.get('result')
    if not isinstance(payload, dict) or document.get('sha256') != payload_digest(payload):
        raise ValueError(f'{path}: result content digest is missing or does not match')
    base = {'backend_id', 'version', 'run_id', 'units', 'pairs', 'tracks', 'unmatched', 'excluded_units', 'provenance'}
    if set(payload) != base | {'score', 'track_kind', 'tracking_rule'}:
        raise ValueError(f'{path}: format 2 fields do not match the schema')
    payload = _tables(payload, path)
    return MatchResult(payload['backend_id'], payload['version'], payload['run_id'], payload['units'], payload['pairs'],
                       payload['tracks'], payload['unmatched'], payload['excluded_units'],
                       ScoreBlock.from_json(payload['score']), TrackKind(payload['track_kind']),
                       payload['tracking_rule'], Provenance.from_json(payload['provenance'])).validate()
