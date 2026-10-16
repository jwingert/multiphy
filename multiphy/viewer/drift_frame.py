"""The Qt-free drift frame: one rule that turns a run's saved registration into drawn depths.

``drift_frame`` reads ``provenance['group_runs'][group]['provenance']['registration']`` and
returns one ``Column`` per day in day order, each mapped into the anchor day's depth frame;
``unit_depth`` and ``electrode_depths`` apply it. A day with no offsets is drawn raw
(``offsets_um == ()``), with the reason in ``Column.reason`` or ``DriftFrame.no_offsets``.
A registration this module cannot read raises ``ValueError``. ``registered_positions`` reads every
unit's ``x_um`` and drawn depth the same way, and ``unit_distance`` is the 2D distance between two
units of one recording group in that frame. Imports no Qt and no matplotlib.
"""
import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Column:
    """One day's probe column: ``depth_scale`` and offsets into the run's frame, ``()`` when raw."""
    session: str
    scale: float
    offsets_um: tuple
    reason: str


@dataclass(frozen=True)
class DriftFrame:
    """Every day of one recording group mapped into the anchor day's depth frame."""
    group: str
    anchor: Optional[str]
    method: str
    columns: tuple
    no_offsets: str
    anchor_scale: float
    anchor_offset_um: float
    unit_offsets_um: dict
    unit_shanks: dict
    shank_width_um: Optional[float]


def group_label(group: str) -> str:
    """A group key saved as a JSON list reads as its items joined by `` · ``; any other key unchanged."""
    try:
        items = json.loads(group)
    except (TypeError, ValueError):
        return group
    if isinstance(items, list):
        return ' · '.join(str(item) for item in items)
    return group


def _unit_groups(units: pd.DataFrame) -> pd.Series:
    if 'recording_group' in units.columns:
        return units['recording_group'].astype(str)
    return pd.Series([''] * len(units), index=units.index, dtype=object)


def group_units(units: pd.DataFrame, group: str) -> pd.DataFrame:
    """The units whose ``recording_group`` (as ``str``; ``''`` when the column is absent) equals ``group``."""
    return units[_unit_groups(units) == group]


def group_sessions(units: pd.DataFrame, group: str) -> list:
    """The group's sessions, by ``(session_date, session)`` when dated, else by session."""
    rows = group_units(units, group)
    column = 'session_id' if 'session_id' in rows.columns else 'siteid'
    rows = rows.drop_duplicates(column)
    if 'session_date' in rows.columns:
        return rows.sort_values(['session_date', column], kind='mergesort')[column].tolist()
    return sorted(rows[column].tolist())


def drawn_group(units: pd.DataFrame, selection: Sequence[str]) -> Optional[str]:
    """The group of the first selected unit in ``units``; none selected → the first group by name."""
    if units.empty:
        return None
    groups = _unit_groups(units)
    by_cell = dict(zip(units['cellid'], groups))
    for cellid in selection:
        if cellid in by_cell:
            return by_cell[cellid]
    return sorted(set(groups))[0]


def _real(value) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and math.isfinite(value))


def _count(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0


def _shank_list(value) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(_real(v) for v in value)


def _check(reg, group: str) -> None:
    """Raise ``ValueError`` for a registration this module cannot read."""
    def fail(problem):
        raise ValueError(f'registration of {group}: {problem}')

    if not isinstance(reg, dict):
        fail('not a dict')
    if not isinstance(reg.get('method'), str):
        fail('method missing or not a str')
    for key in ('offsets_um', 'depth_scale', 'unit_offsets_um'):
        value = reg.get(key, {})
        if not isinstance(value, dict) or not all(_real(v) for v in value.values()):
            fail(f'{key} is not a dict of finite numbers')
    if any(v == 0 for v in reg.get('depth_scale', {}).values()):
        fail('a depth_scale of 0')
    shank_offsets = reg.get('shank_offsets_um', {})
    if not isinstance(shank_offsets, dict) or not all(_shank_list(v) for v in shank_offsets.values()):
        fail('shank_offsets_um is not a dict of non-empty lists of finite numbers')
    if len({len(v) for v in shank_offsets.values()}) > 1:
        fail('shank_offsets_um lists differ in length')
    unit_shanks = reg.get('unit_shanks', {})
    if not isinstance(unit_shanks, dict) or not all(_count(v) for v in unit_shanks.values()):
        fail('unit_shanks is not a dict of shank numbers >= 0')
    both = sorted(set(reg.get('offsets_um', {})) & set(shank_offsets))
    if both:
        fail(f'{", ".join(both)} in both offsets_um and shank_offsets_um')
    diagnostics = reg.get('diagnostics') or {}
    if not isinstance(diagnostics, dict) or not isinstance(diagnostics.get(group) or {}, dict):
        fail('diagnostics is not a dict of dicts')
    diag = diagnostics.get(group) or {}
    groups = diag.get('groups')
    if 'groups' in diag and not (isinstance(groups, list) and all(
            isinstance(g, list) and all(isinstance(s, str) for s in g) for g in groups)):
        fail(f'diagnostics[{group}][groups] is not a list of lists of str')
    shank_lists = bool(shank_offsets)
    pairs = reg.get('pair_drifts', [])
    if not isinstance(pairs, list):
        fail('pair_drifts is not a list')
    for pair in pairs:
        if not (isinstance(pair, dict) and isinstance(pair.get('session_a'), str)
                and isinstance(pair.get('session_b'), str) and _count(pair.get('n_matches'))):
            fail('a pair_drifts entry lacks str session_a, session_b or an int n_matches >= 0')
        shank_lists = shank_lists or isinstance(pair.get('shank_depth_um'), list)
        if pair['n_matches'] > 0:
            depth, shanks = pair.get('depth_um'), pair.get('shank_depth_um')
            if (depth is None) == (shanks is None):
                fail(f'pair {pair["session_a"]}–{pair["session_b"]} needs exactly one of '
                     'depth_um or shank_depth_um')
            if depth is not None and not _real(depth):
                fail(f'pair {pair["session_a"]}–{pair["session_b"]} depth_um is not a finite number')
            if shanks is not None and not _shank_list(shanks):
                fail(f'pair {pair["session_a"]}–{pair["session_b"]} shank_depth_um is not a '
                     'non-empty list of finite numbers')
    width = reg.get('shank_width_um')
    if shank_lists and not (_real(width) and width > 0):
        fail('per-shank offsets without a finite shank_width_um > 0')


def _ordered_groups(sessions: Sequence[str], lists) -> list:
    """Each list's members in day order, a day in no list its own group; groups by first day."""
    position = {s: i for i, s in enumerate(sessions)}
    seen, groups = set(), []
    for members in lists:
        kept = sorted({s for s in members if s in position and s not in seen}, key=position.get)
        seen.update(kept)
        if kept:
            groups.append(kept)
    groups += [[s] for s in sessions if s not in seen]
    return sorted(groups, key=lambda g: position[g[0]])


def _anchor(sessions: Sequence[str], groups: list):
    """``sessions[0]`` when its group has 2+ days, else the first day of the largest such group."""
    position = {s: i for i, s in enumerate(sessions)}
    group_of = {s: g for g in groups for s in g}
    first = group_of[sessions[0]]
    multi = [g for g in groups if len(g) >= 2]
    if len(first) >= 2 or not multi:
        return sessions[0], first
    best = min(multi, key=lambda g: (-len(g), position[g[0]]))
    return best[0], best


def _raw(group: str, reg, sessions: Sequence[str], no_offsets: str) -> DriftFrame:
    return DriftFrame(group, None, reg['method'] if reg else '',
                      tuple(Column(s, 1.0, (), '') for s in sessions), no_offsets, 1.0, 0.0, {}, {}, None)


def _components(sessions: Sequence[str], kept) -> list:
    parent = {s: s for s in sessions}

    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for pair in kept:
        parent[root(pair['session_a'])] = root(pair['session_b'])
    members = {}
    for s in sessions:
        members.setdefault(root(s), []).append(s)
    return list(members.values())


def drift_frame(provenance, group: str, sessions: Sequence[str], *, saves_offsets: bool) -> DriftFrame:
    """Read the group run's saved registration into one ``Column`` per day of ``sessions`` (day order).

    ``provenance`` is anything with ``.get``. A day outside the anchor's linked group is drawn raw
    with ``no matches to <anchor>``; a group with no usable offsets is drawn raw with
    ``DriftFrame.no_offsets`` saying why. A registration that cannot be read raises ``ValueError``.
    """
    sessions = list(sessions)
    if not sessions:
        raise ValueError('no sessions to draw')
    entry = (provenance.get('group_runs') or {}).get(group)
    reg = entry['provenance'].get('registration') if entry else None
    if reg is None:
        return _raw(group, reg, sessions,
                    'run predates offsets' if saves_offsets else 'backend saves no offsets')
    _check(reg, group)
    if reg['method'] == 'none':
        return _raw(group, reg, sessions, 'registration off')
    unit_shanks = {k: int(v) for k, v in (reg.get('unit_shanks') or {}).items()}
    width = reg.get('shank_width_um')
    width = float(width) if _real(width) else width
    if 'pair_drifts' in reg:
        return _pair_frame(reg, group, sessions, unit_shanks, width)

    offsets = reg.get('offsets_um') or {}
    shank_offsets = reg.get('shank_offsets_um') or {}
    scales = reg.get('depth_scale') or {}
    registered = set(offsets) | set(shank_offsets)
    diag = (reg.get('diagnostics') or {}).get(group) or {}
    lists = [[s for s in g if s in registered and s in sessions] for g in diag.get('groups') or []]
    lists = [g for g in lists if g]
    if not lists:
        return _raw(group, reg, sessions, diag.get('note') or 'group not registered')
    anchor, linked = _anchor(sessions, _ordered_groups(sessions, lists))
    if anchor in shank_offsets:
        raise ValueError(f'registration of {group}: anchor {anchor} has per-shank offsets only')
    anchor_scale = float(scales.get(anchor, 1.0))
    anchor_offset = float(offsets.get(anchor, 0.0))
    columns = []
    for day in sessions:
        if day == anchor:
            columns.append(Column(anchor, anchor_scale, (anchor_offset,), ''))
        elif day in linked:
            shift = ((float(offsets[day]),) if day in offsets
                     else tuple(float(v) for v in shank_offsets[day]))
            columns.append(Column(day, float(scales.get(day, 1.0)), shift, ''))
        else:
            columns.append(Column(day, 1.0, (), f'no matches to {anchor}'))
    unit_offsets = {k: float(v) for k, v in (reg.get('unit_offsets_um') or {}).items()}
    return DriftFrame(group, anchor, reg['method'], tuple(columns), '', anchor_scale, anchor_offset,
                      unit_offsets, unit_shanks, width)


def _pair_frame(reg, group: str, sessions: list, unit_shanks: dict, width) -> DriftFrame:
    """Each day shifted by its own pair with the anchor; drift never chains through a third day."""
    drawn = set(sessions)
    kept = [p for p in reg['pair_drifts']
            if p['n_matches'] > 0 and p['session_a'] in drawn and p['session_b'] in drawn]
    anchor, _ = _anchor(sessions, _ordered_groups(sessions, _components(sessions, kept)))
    direct = {}
    for pair in kept:
        if pair['session_b'] == anchor:
            raise ValueError(f'registration of {group}: pair {pair["session_a"]}–{anchor} ends at '
                             f'the anchor {anchor}')
        if pair['session_a'] == anchor:
            direct.setdefault(pair['session_b'], pair)
    columns = []
    for day in sessions:
        if day == anchor:
            columns.append(Column(anchor, 1.0, (0.0,), ''))
        elif day in direct:
            pair = direct[day]
            shift = ((float(pair['depth_um']),) if pair.get('depth_um') is not None
                     else tuple(float(v) for v in pair['shank_depth_um']))
            columns.append(Column(day, 1.0, shift, ''))
        else:
            columns.append(Column(day, 1.0, (), f'no matches to {anchor}'))
    return DriftFrame(group, anchor, reg['method'], tuple(columns), '', 1.0, 0.0, {}, unit_shanks, width)


def shank_of(x_um: float, width_um: float, n_shanks: int) -> int:
    """The shank ``i`` with ``i * width_um < x_um < (i + 1) * width_um``; else ``0`` (the upstream rule)."""
    for i in range(n_shanks):
        if i * width_um < x_um < (i + 1) * width_um:
            return i
    return 0


def unit_depth(frame: DriftFrame, column: Column, cellid: str, depth: float, x_um: float) -> Optional[float]:
    """A unit's drawn depth in the anchor's frame; ``None`` when its shank cannot be known."""
    offsets = column.offsets_um
    if offsets == ():
        return depth
    if cellid in frame.unit_offsets_um:
        corrected = depth + frame.unit_offsets_um[cellid]
    else:
        if len(offsets) == 1:
            shift = offsets[0]
        else:
            if cellid in frame.unit_shanks:
                shank = frame.unit_shanks[cellid]
            elif x_um is not None and math.isfinite(x_um):
                shank = shank_of(x_um, frame.shank_width_um, len(offsets))
            else:
                return None
            if shank >= len(offsets):
                raise ValueError(f'{cellid} is on shank {shank}; {column.session} has '
                                 f'{len(offsets)} shank offsets')
            shift = offsets[shank]
        corrected = column.scale * depth + shift
    return (corrected - frame.anchor_offset_um) / frame.anchor_scale


def electrode_depths(frame: DriftFrame, column: Column, geometry: np.ndarray) -> np.ndarray:
    """Each electrode's drawn depth (``geometry[:, 1]``) in the anchor's frame; raw when no offsets."""
    geometry = np.asarray(geometry, dtype=float)
    y = geometry[:, 1].copy()
    offsets = column.offsets_um
    if offsets == ():
        return y
    if len(offsets) == 1:
        shift = offsets[0]
    else:
        shift = np.array([offsets[shank_of(x, frame.shank_width_um, len(offsets))]
                          for x in geometry[:, 0]])
    return (column.scale * y + shift - frame.anchor_offset_um) / frame.anchor_scale
def _measured(units: pd.DataFrame, name: str) -> np.ndarray:
    """``units[name]`` as floats, as the drift view reads it; missing column or non-finite → ``nan``."""
    if name not in units.columns:
        return np.full(len(units), np.nan)
    values = pd.to_numeric(units[name], errors='coerce').to_numpy(float, na_value=np.nan)
    return np.where(np.isfinite(values), values, np.nan)


def registered_positions(units: pd.DataFrame, provenance) -> pd.DataFrame:
    """One row per unit (index ``cellid``): ``group``, ``x_um`` and its drawn ``depth`` as the drift view draws it.

    A group whose registration cannot be read, or whose unit lies beyond its day's shank offsets,
    keeps its ``x_um`` with every ``depth`` ``nan``; a unit on an unlinked day keeps its measured depth.
    """
    cellids = units['cellid'].astype(str).to_numpy()
    groups = _unit_groups(units).to_numpy()
    x_um = _measured(units, 'x_um')
    measured = _measured(units, 'depth')
    depth = np.full(len(units), np.nan)
    session_column = 'session_id' if 'session_id' in units.columns else 'siteid'
    sessions = units[session_column].to_numpy() if session_column in units.columns else [None] * len(units)
    for group in sorted(set(groups)):
        rows = np.flatnonzero(groups == group)
        try:
            frame = drift_frame(provenance, group, group_sessions(units, group), saves_offsets=False)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        columns = {column.session: column for column in frame.columns}
        drawn = {}
        try:
            for i in rows:
                column = columns.get(sessions[i])
                if column is None or not np.isfinite(measured[i]):
                    continue
                y = unit_depth(frame, column, cellids[i], float(measured[i]), float(x_um[i]))
                if y is not None:
                    drawn[i] = float(y)
        except ValueError:
            continue
        for i, y in drawn.items():
            depth[i] = y
    return pd.DataFrame({'group': groups.astype(str), 'x_um': x_um, 'depth': depth},
                        index=pd.Index(cellids, name='cellid'))


def unit_distance(positions: pd.DataFrame, a: str, b: str) -> float:
    """``hypot(dx, ddepth)`` of two units of one group in ``registered_positions``; else ``nan``."""
    if a not in positions.index or b not in positions.index:
        return math.nan
    row_a, row_b = positions.loc[a], positions.loc[b]
    if row_a['group'] != row_b['group']:
        return math.nan
    numbers = [float(row_a['x_um']), float(row_b['x_um']), float(row_a['depth']), float(row_b['depth'])]
    if not all(math.isfinite(v) for v in numbers):
        return math.nan
    return math.hypot(numbers[0] - numbers[1], numbers[2] - numbers[3])
