"""Typed provenance carried by inputs and results, plus JSON normalization.

Known fields are attributes; anything else lives in ``extra``. ``to_json`` flattens
``extra`` into the mapping so a format-2 file reads like the format-1 files did.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, replace
import hashlib
import json
from typing import Optional

import numpy as np
import pandas as pd


def json_safe(value):
    """Normalize numerical metadata without permitting executable serialization."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError('JSON mappings require string keys')
        return {k: json_safe(v) for k, v in value.items()}
    raise ValueError(f'Unsupported metadata type: {type(value).__name__}')


def payload_digest(payload) -> str:
    return hashlib.sha256(json.dumps(json_safe(payload), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Provenance:
    dataset_id: Optional[str] = None
    input_manifest_sha256: Optional[str] = None
    group: Optional[str] = None
    partition: Optional[str] = None
    tier: Optional[str] = None
    implementation: Optional[dict] = None
    upstream_version: Optional[str] = None
    verified_api_revision: Optional[str] = None
    parameters: Optional[dict] = None
    calibration_status: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'extra' and getattr(self, f.name) is not None}
        clash = set(out) & set(self.extra)
        if clash:
            raise ValueError(f'extra provenance keys shadow typed fields: {sorted(clash)}')
        out.update(self.extra)
        return json_safe(out)

    @classmethod
    def from_json(cls, d: Optional[dict]) -> 'Provenance':
        d = dict(d or {})
        if not isinstance(d, dict):
            raise ValueError('Provenance must be a mapping')
        known = {f.name for f in fields(cls)} - {'extra'}
        typed = {k: d.pop(k) for k in list(d) if k in known}
        return cls(extra=d, **typed)

    def merged(self, **changes) -> 'Provenance':
        extra = dict(self.extra, **changes.pop('extra', {}))
        return replace(self, extra=extra, **changes)

    def get(self, key, default=None):
        """Read a typed field or an extra key by name; keeps old ``provenance.get`` call sites working."""
        if key in {f.name for f in fields(self)} and key != 'extra':
            value = getattr(self, key)
            return default if value is None else value
        return self.extra.get(key, default)
