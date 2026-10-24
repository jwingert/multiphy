"""Drift validation: inject a known rigid shift into one day and measure what the matchers recover.

A ``DriftDataset`` is one project's units, features, geometry and trial timing with its reference
labels (``ground_truth``) and the per-session depth correction that registers each day
(``truth_offsets``, µm, the ``registration`` sign convention: ``offsets[b] - offsets[a]`` is added
to day ``b``'s depths to put them in day ``a``'s frame). ``inject_shift`` moves one session's
channel depths by a whole number of rows and updates its truth offset. ``validate_datasets`` runs
every backend and registration on every shift (applied to the last session in session order) and
writes, once, into a new folder: per-run results, ``summary.csv``, ``calibration.csv``, two
figures with legends, ``METHODS.md`` and ``manifest.json``. A backend's ``ValueError`` becomes a
``refused`` row with its reason; the loop continues. Datasets are never pooled.

CLI: ``python -m multiphy.drift_validation [--root DIR] --project NAME --day DAY [--day DAY ...]
--output DIR [--radius-um 80] [--seed 0]``.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .project.folder import Project
from .project.selection import Selection
from .run.config import cache_root

SUMMARY_COLUMNS = ('dataset', 'backend', 'registration', 'shift_um', 'status', 'reason', 'offset_error_um',
                   'n_offset_pairs',
                   'precision', 'recall', 'n_true_pairs', 'brier', 'n_scored_pairs')
CALIBRATION_COLUMNS = ('dataset', 'backend', 'registration', 'shift_um', 'bin_lower', 'bin_upper', 'n',
                       'mean_score', 'observed_same_fraction')
N_BINS = 10
FIGURE_PARAMS = {
    'legend.fontsize': 8, 'axes.labelsize': 20, 'axes.titlesize': 20, 'axes.spines.right': False,
    'axes.spines.top': False, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'Arial', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'axes.linewidth': 5, 'xtick.major.size': 18, 'ytick.major.size': 18,
    'xtick.major.width': 5, 'ytick.major.width': 5, 'xtick.minor.width': 3, 'ytick.minor.width': 3,
    'xtick.minor.size': 5, 'ytick.minor.size': 5}


@dataclass(frozen=True)
class DriftDataset:
    """One project's matcher inputs, its reference labels and the true per-session depth corrections."""
    name: str
    units: pd.DataFrame
    features: dict
    geometry: dict
    timing: dict
    ground_truth: pd.DataFrame
    truth_offsets: dict
    source: Optional[str] = None  # where the dataset was read from; None = in memory


def load_project_day(project: Project, day: str) -> DriftDataset:
    """One extracted project day: its halves archive with the day's own trial events.

    The two halves are the day's sessions and windows of one recording, so every truth offset is
    ``0.0``. A day extracted without events has ``timing == {}`` and is validated on waveforms and
    physiology alone.
    """
    sel = Selection(project, (day,))
    ds = sel.dataset()
    return DriftDataset(sel.dataset_id, ds.units, ds.features, ds.geometry, sel.timing(),
                        ds.labels.drop(columns=['partition'], errors='ignore'),
                        {s: 0.0 for s in sorted(ds.units.session_id.unique())},
                        source=str(project.day_dir(day)))


def _row_pitch(geometry: np.ndarray) -> float:
    y = np.unique(np.asarray(geometry, dtype=float)[:, 1])
    steps = np.diff(y)
    steps = steps[steps > 0]
    if not len(steps):
        raise ValueError('the geometry has 1 row; a shift needs at least 2 rows to define a row pitch')
    return float(steps.min())


def inject_shift(dataset: DriftDataset, session: str, shift_um: float) -> DriftDataset:
    """A copy with ``session``'s channel depths increased by ``shift_um`` and its truth offset decreased by it.

    ``shift_um`` must be a whole multiple of the session's row pitch (the smallest positive
    difference of its sorted unique channel y).
    """
    if session not in dataset.geometry:
        raise ValueError(f'session {session!r} has no geometry; sessions: {sorted(dataset.geometry)}')
    pitch = _row_pitch(dataset.geometry[session])
    if not np.isfinite(float(shift_um)) or not (np.isfinite(pitch) and pitch > 0):
        raise ValueError(f'shift {shift_um} µm is not a whole number of rows of {session}; its row pitch is '
                         f'{pitch:g} µm. Use a finite multiple of {pitch:g}')
    rows = float(shift_um) / pitch
    if not np.isclose(rows, np.round(rows), rtol=0, atol=1e-9):
        raise ValueError(f'shift {shift_um} µm is not a whole number of rows of {session}; its row pitch is '
                         f'{pitch:g} µm. Use a multiple of {pitch:g}')
    geometry = {s: np.array(g, dtype=float, copy=True) for s, g in dataset.geometry.items()}
    geometry[session][:, 1] += float(shift_um)
    truth_offsets = dict(dataset.truth_offsets)
    truth_offsets[session] = float(truth_offsets[session]) - float(shift_um)
    return dataclasses.replace(dataset, geometry=geometry, truth_offsets=truth_offsets)


def _sessions(dataset: DriftDataset) -> list:
    """Session ids in the matcher's day order (``references._session_order``: session_date, else start_sample)."""
    from .backends.local.references import _session_order
    units = dataset.units.assign(session_id=dataset.units.session_id.astype(str))
    if 'session_date' not in units:
        units = units[units.cellid.isin(set(dataset.features))]
    return [str(s) for s in _session_order(units, dataset.features)]


def _session_pairs(dataset: DriftDataset) -> list:
    """Session pairs ``(a, b)``, in session order, whose sessions share a ``recording_group``.

    Both backends register each group separately and centre its offsets on zero, so an offset
    difference across groups compares two unrelated frames and is never scored.
    """
    units = dataset.units.assign(session_id=dataset.units.session_id.astype(str))
    group_of = (units.drop_duplicates('session_id').set_index('session_id').recording_group.to_dict()
                if 'recording_group' in units else {})
    return [(a, b) for a, b in itertools.combinations(_sessions(dataset), 2)
            if group_of.get(a) == group_of.get(b) and pd.notna(group_of.get(a, ''))]


def _offset_error(offsets: dict, truth: dict, pairs: list, shifted: str) -> tuple:
    """``(error, reason)``: the largest absolute error of relative offsets over ``pairs`` (``_session_pairs``).

    Undefined (NaN with a reason) unless the ``shifted`` session is in a compared pair, so a run
    never reads as recovered from sessions the injected shift did not touch.
    """
    if not pairs:
        return np.nan, 'no two sessions share a recording_group, so no relative offset is defined'
    if not any(shifted in pair for pair in pairs):
        return np.nan, (f'shifted session {shifted} shares a recording_group with no other session, '
                        'so its injected shift is not measured')
    sessions = list(dict.fromkeys(s for pair in pairs for s in pair))
    missing = [s for s in sessions if s not in offsets or offsets[s] is None or not np.isfinite(offsets[s])]
    if missing:
        return np.nan, f'no registered offset for sessions {missing}'
    return float(max(abs((offsets[b] - offsets[a]) - (truth[b] - truth[a])) for a, b in pairs)), ''


def _registered_pairs(result, pairs: list, registration: str) -> tuple:
    """``(kept, unmeasured)``: the ``pairs`` whose sessions the run registered to each other, and the rest.

    With ``registration='none'`` every offset is 0 in one shared frame, so every pair is kept (the
    error is then the uncorrected drift). Otherwise ``ev.register_depths`` centres each connected
    component of measured sessions on its own mean (``registration.diagnostics[group]['groups']``),
    so a pair across components, or with a session in no reported component, has no measured
    relative offset.
    """
    if registration == 'none':
        return list(pairs), []
    diagnostics = (result.provenance.get('registration') or {}).get('diagnostics') or {}
    component = {}
    for group, diag in diagnostics.items():
        for i, members in enumerate((diag or {}).get('groups') or []):
            for session in members:
                component[str(session)] = (str(group), i)
    kept = [(a, b) for a, b in pairs if a in component and component.get(a) == component.get(b)]
    return kept, [pair for pair in pairs if pair not in kept]


def _registration_error(result, registration: str, truth: dict, pairs: list, shifted: str) -> tuple:
    """``(error, reason)`` of ``_offset_error`` over the pairs ``_registered_pairs`` keeps.

    NaN with a reason when the ``shifted`` session is in a recording_group pair but registered to
    no other session; a finite error over the kept pairs names any unmeasured pairs in ``reason``.
    """
    offsets = (result.provenance.get('registration') or {}).get('offsets_um') or {}
    if not pairs or not any(shifted in pair for pair in pairs):
        return _offset_error(offsets, truth, pairs, shifted)
    kept, unmeasured = _registered_pairs(result, pairs, registration)
    if not any(shifted in pair for pair in kept):
        return np.nan, (f'shifted session {shifted} was registered to no other session (no measured offset '
                        f'connects it; unmeasured session pairs {unmeasured}), so its injected shift is not measured')
    error, reason = _offset_error(offsets, truth, kept, shifted)
    if unmeasured and not reason:
        reason = (f'unmeasured session pairs {unmeasured} (different registration components) are not scored; '
                  f'offset error is over {kept}')
    return error, reason


def _known_pairs(result, ground_truth: pd.DataFrame) -> dict:
    """(cellid, cellid) -> same unit, for cross-session pairs of one source recording."""
    labels = ground_truth.assign(cellid=ground_truth.cellid.astype(str))
    labels = labels[labels.cellid.isin(set(result.units.cellid))]
    session_of = result.units.set_index('cellid').session_id.to_dict()
    truth = {}
    for _, group in labels.groupby(['well', 'source_session']):
        for a, z in itertools.combinations(group.itertuples(), 2):
            if session_of[a.cellid] != session_of[z.cellid]:
                truth[tuple(sorted((a.cellid, z.cellid)))] = bool(a.source_unit_id == z.source_unit_id)
    return truth


def _brier(result, truth: dict) -> tuple:
    errors = []
    for r in result.pairs.itertuples():
        key = tuple(sorted((r.cellid_a, r.cellid_b)))
        score = pd.to_numeric(r.match_score, errors='coerce')
        if key in truth and np.isfinite(score):
            errors.append((float(score) - float(truth[key])) ** 2)
    return (float(np.mean(errors)) if errors else np.nan), len(errors)


def _json_shift(shift):
    """A requested shift for manifest.json: a finite float, else its text (a refused run's shift, e.g. 'inf')."""
    value = float(shift)
    return value if np.isfinite(value) else str(shift)


def _git_head() -> str:
    try:
        out = subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else 'unknown'


def _style(pairs: list) -> dict:
    """(dataset, backend) -> colour; one colour per line family."""
    import matplotlib
    colours = matplotlib.colormaps['tab10']
    return {key: colours(i % 10) for i, key in enumerate(pairs)}


def _drift_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    from matplotlib.figure import Figure

    ok = summary[summary.status == 'ok']
    families = list(dict.fromkeys(zip(summary.dataset, summary.backend)))
    colours = _style(families)
    with matplotlib.rc_context(FIGURE_PARAMS):
        fig = Figure(figsize=(16, 6))
        left, right = fig.subplots(1, 2)
        for (dataset, backend), colour in colours.items():
            for registration, style in (('matches', '-'), ('none', '--')):
                rows = ok[(ok.dataset == dataset) & (ok.backend == backend) & (ok.registration == registration)]
                rows = rows.sort_values('shift_um')
                label = f'{dataset} · {backend} · {registration}'
                left.plot(rows.shift_um, rows.offset_error_um, style, marker='o', color=colour, lw=3, label=label)
                right.plot(rows.shift_um, rows.recall, style, marker='o', color=colour, lw=3, label=label)
        left.set_xlabel('injected shift (µm)')
        left.set_ylabel('offset error (µm)')
        right.set_xlabel('injected shift (µm)')
        right.set_ylabel('pair recall')
        right.set_ylim(-0.05, 1.05)
        left.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=150)


def _calibration_figure(calibration: pd.DataFrame, path: Path) -> None:
    import matplotlib
    from matplotlib.figure import Figure

    shown = calibration[(calibration.shift_um == 0) & (calibration.registration == 'matches') & (calibration.n > 0)]
    families = list(dict.fromkeys(zip(calibration.dataset, calibration.backend)))
    colours = _style(families)
    with matplotlib.rc_context(FIGURE_PARAMS):
        fig = Figure(figsize=(8, 6))
        ax = fig.subplots()
        ax.plot([0, 1], [0, 1], ':', color='0.5', lw=2, label='identity')
        for (dataset, backend), colour in colours.items():
            rows = shown[(shown.dataset == dataset) & (shown.backend == backend)]
            if len(rows):
                ax.plot(rows.mean_score, rows.observed_same_fraction, '-', marker='o', color=colour, lw=3,
                        label=f'{dataset} · {backend}')
        if shown.empty:
            ax.text(0.5, 0.5, 'no scored runs · shift 0 · matches', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('match score (bin mean)')
        ax.set_ylabel('observed same fraction')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=150)


def _n_text(summary: pd.DataFrame, column: str) -> str:
    lines = []
    for (dataset, backend), rows in summary.groupby(['dataset', 'backend'], sort=False):
        values = ', '.join(f'{r.registration} shift {r.shift_um:g}: '
                           + (f'{r[column]:g}' if r.status == 'ok' and pd.notna(r[column]) else r.status)
                           for _, r in rows.iterrows())
        lines.append(f'- {dataset} · {backend}: {values}')
    return '\n'.join(lines)


def _drift_legend(summary: pd.DataFrame) -> str:
    return f'''# drift_recovery.png

**Drift recovery against an injected rigid shift.** One session per dataset (the last in session
order) had its channel depths moved by the injected shift; each line is one dataset and one
matching backend, solid with match-driven registration (`matches`), dashed without registration
(`none`). **Left:** offset error, the largest absolute error of the estimated relative depth offset
over session pairs that share a `recording_group`, the frame each backend registers in; with
`matches`, only pairs within one registration component (sessions linked by measured offsets)
count, and a point with unmeasured pairs names them in summary.csv `reason` (`n_offset_pairs` per
point below); a shifted session registered to no other session has no point (0 pairs). **Right:** pair recall,
true-positive track links over known same-unit cross-session pairs (`n_true_pairs` per point below).
Refused runs are not drawn. Each dataset is shown separately; nothing is pooled across datasets.

Session pairs compared per offset-error point:
{_n_text(summary, 'n_offset_pairs')}

Known same-unit pairs per point:
{_n_text(summary, 'n_true_pairs')}
'''


def _calibration_legend(calibration: pd.DataFrame) -> str:
    shown = calibration[(calibration.shift_um == 0) & (calibration.registration == 'matches')]
    lines = []
    for (dataset, backend), rows in shown.groupby(['dataset', 'backend'], sort=False):
        lines.append(f'- {dataset} · {backend}: pairs per bin (low to high score) '
                     + ', '.join(str(int(n)) for n in rows.n))
    counts = '\n'.join(lines) or '- no scored run at shift 0 with matches'
    return f'''# calibration.png

**Score calibration without injected shift.** For each dataset and backend, run at shift 0 with
match-driven registration (`matches`), the known cross-session pairs with a finite match score are
put in {N_BINS} equal score bins on [0, 1]; each point is one non-empty bin, at its mean score
(x) and the fraction of its pairs that are the same unit (y). The dotted diagonal is perfect
calibration. `functional_likelihood` scores are uncalibrated posteriors; `pair_logistic` scores are
probabilities calibrated on held-out sessions. Pairs of one unit are dependent, so no intervals are
drawn. Each dataset is shown separately; nothing is pooled across datasets.

Pairs per point:
{counts}
'''


def _methods(datasets, shifted: dict, options: dict, shifts_um, radius_um: float, seed: int,
             commit: str) -> str:
    rows = '\n'.join(f'| {d.name} | {d.source or "in memory (no path)"} | {len(d.units)} | '
                     f'{len(_sessions(d))} | {shifted[d.name]} |' for d in datasets)
    backend_options = '\n'.join(f'- `{b}`: `{json.dumps(o, sort_keys=True)}`' for b, o in options.items())
    return f'''# Drift validation · methods

**Goal.** Measure how well each matcher recovers a known rigid probe drift (registration) and how
its match scores relate to the observed same-unit fraction (calibration), one dataset at a time.

## Datasets

| dataset | source | units | sessions | shifted session |
|---|---|---|---|---|
{rows}

Each project day is read by `drift_validation.load_project_day` from its hash-verified halves
archive (`Selection(project, (day,)).dataset()`) and the day's own events (`Selection.timing()`,
the same trial timing for both halves); a day extracted without events has no timing and is
validated on waveforms and physiology alone. Its sessions are the day's two temporal halves, time
windows of one recording, so every true offset starts at 0 µm.

## Shift rule

For each shift in {list(shifts_um)} µm, `inject_shift` adds the shift to the channel y of the last
session in session order and subtracts it from that session's true offset. A shift must be a whole
multiple of that session's row pitch (smallest positive difference of its unique channel y).

## Runs

Every dataset × shift × backend × registration runs `registry.get(backend).run(MatchingInput(...),
options, RunContext(run_id, work_dir=runs/<run_id>, seed={seed}))`; the result is saved to
`runs/<run_id>/result.json`. Options are each backend's defaults with `registration`,
`radius_um = {radius_um:g}` and `seed = {seed}` set:

{backend_options}

A `ValueError` from a run is recorded as `status = refused` with its message in `reason`.

## Metrics (summary.csv)

- `offset_error_um`: max over session pairs (a, b) that share a `recording_group` of
  |(o_b − o_a) − (t_b − t_a)|, o = the result's `registration.offsets_um` provenance, t = true
  offsets (µm). Each backend registers a group in its own zero-mean frame, so pairs across groups
  are not compared. With `registration = matches`, only pairs in one registration component
  (`registration.diagnostics[group].groups`: sessions linked by measured offsets, each centred on
  its own mean) are compared; the other pairs are named in `reason` as unmeasured. With
  `registration = none` all offsets are 0, so the error is the uncorrected drift. Undefined (NaN,
  with a `reason`) when the shifted session shares its group with no other session, or is
  registered to no other session, since its shift is then never measured.
- `n_offset_pairs`: the session pairs `offset_error_um` is the maximum over; 0 when it is undefined.
- `precision`, `recall`: `pair_precision`, `pair_recall` of `evaluation.evaluate_result(result,
  ground_truth, expected_partition=None)`; every within-track pair is a predicted link.
- `n_true_pairs`: known same-unit cross-session pairs.
- `brier`, `n_scored_pairs`: mean of (match_score − same)² over known cross-session pairs with a
  finite score, and their number.
- calibration.csv: per run, {N_BINS} equal score bins on [0, 1] (`evaluation.score_reliability`):
  `n`, mean score, observed same fraction; a refused run has its {N_BINS} bins with `n = 0`.

## Figures

- `drift_recovery.png` ← summary.csv (see `drift_recovery_legend.md`).
- `calibration.png` ← calibration.csv at shift 0 with `matches` (see `calibration_legend.md`).

Code: `multiphy/drift_validation.py` at commit `{commit}`.
'''


def validate_datasets(datasets: Sequence[DriftDataset], output: Path, *, shifts_um: tuple = (0, 20, 40, 80, 160, 320),
                      backends: tuple = ('functional_likelihood', 'pair_logistic'),
                      registrations: tuple = ('matches', 'none'), radius_um: float = 80.0, seed: int = 0) -> pd.DataFrame:
    """Run every backend and registration on every injected shift of every dataset; write the outputs.

    ``datasets`` must be at least 1, each with a unique name (``ValueError``), and ``output`` must
    not exist (``FileExistsError``); both are checked before ``output`` is created, so a refused call
    writes nothing, and ``output`` is created before any work. Each dataset's ``source`` is listed in
    ``METHODS.md`` and ``manifest.json``.
    Returns the summary table.
    """
    from .contract import registry
    from .contract.backend import RunContext
    from .contract.inputs import MatchingInput
    from .evaluation import evaluate_result
    from .run.run_dir import save_result, write_json

    output = Path(output)
    datasets = list(datasets)
    names = [d.name for d in datasets]
    if not datasets or len(set(names)) != len(names):
        raise ValueError(f'{len(datasets)} datasets named {names}; give at least 1, each with a unique name')
    output.mkdir(parents=True, exist_ok=False)
    options_by_backend = {}
    for backend in backends:
        options = registry.get(backend).spec.defaults()
        options.update(radius_um=float(radius_um), seed=int(seed))
        options_by_backend[backend] = options

    summary, calibration, shifted = [], [], {}
    for dataset in datasets:
        sessions = _sessions(dataset)
        pairs = _session_pairs(dataset)
        shifted[dataset.name] = sessions[-1]
        for shift in shifts_um:
            for backend in backends:
                for registration in registrations:
                    run_id = f'{dataset.name}__{backend}__{registration}__shift{float(shift):g}'.replace('/', '_')
                    work_dir = output / 'runs' / run_id
                    work_dir.mkdir(parents=True)
                    row = dict(dataset=dataset.name, backend=backend, registration=registration, shift_um=shift,
                               status='ok', reason='', offset_error_um=np.nan, n_offset_pairs=np.nan,
                               precision=np.nan, recall=np.nan,
                               n_true_pairs=np.nan, brier=np.nan, n_scored_pairs=np.nan)
                    try:
                        data = inject_shift(dataset, sessions[-1], float(shift))
                        options = dict(options_by_backend[backend], registration=registration)
                        inp = MatchingInput(data.units, group=data.name, features=data.features,
                                            geometry=data.geometry, timing=data.timing).validate()
                        result = registry.get(backend).run(inp, options, RunContext(run_id, work_dir=work_dir,
                                                                                    seed=int(seed)))
                        offset_error, offset_reason = _registration_error(result, registration, data.truth_offsets,
                                                                          pairs, sessions[-1])
                        n_offset_pairs = (len(_registered_pairs(result, pairs, registration)[0])
                                          if np.isfinite(offset_error) else 0)
                        scores = evaluate_result(result, data.ground_truth, expected_partition=None, n_bins=N_BINS)
                        brier = _brier(result, _known_pairs(result, data.ground_truth))
                        bins = [(b['lower'], b['upper'], b['n_pairs'], b['mean_score'], b['observed_same_fraction'])
                                for b in scores['score_reliability']['bins']]
                        save_result(result, work_dir / 'result.json')
                    except ValueError as exc:
                        # a refused run keeps its N_BINS calibration rows, empty, so it stays identifiable
                        row.update(status='refused', reason=f'{type(exc).__name__}: {exc}')
                        edges = np.linspace(0.0, 1.0, N_BINS + 1)
                        bins = [(float(lo), float(hi), 0, np.nan, np.nan) for lo, hi in zip(edges[:-1], edges[1:])]
                    else:
                        row.update(offset_error_um=offset_error, n_offset_pairs=n_offset_pairs, reason=offset_reason,
                                   brier=brier[0],
                                   n_scored_pairs=brier[1], precision=scores['pair_precision'],
                                   recall=scores['pair_recall'], n_true_pairs=scores['n_known_positive_pairs'])
                    summary.append(row)
                    for lo, hi, n, mean_score, same in bins:
                        calibration.append(dict(dataset=dataset.name, backend=backend, registration=registration,
                                                shift_um=shift, bin_lower=lo, bin_upper=hi, n=n,
                                                mean_score=mean_score, observed_same_fraction=same))

    summary_table = pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS))
    calibration_table = pd.DataFrame(calibration, columns=list(CALIBRATION_COLUMNS))
    summary_table.to_csv(output / 'summary.csv', index=False)
    calibration_table.to_csv(output / 'calibration.csv', index=False)
    _drift_figure(summary_table, output / 'drift_recovery.png')
    _calibration_figure(calibration_table, output / 'calibration.png')
    (output / 'drift_recovery_legend.md').write_text(_drift_legend(summary_table), encoding='utf-8')
    (output / 'calibration_legend.md').write_text(_calibration_legend(calibration_table), encoding='utf-8')
    commit = _git_head()
    (output / 'METHODS.md').write_text(_methods(datasets, shifted, options_by_backend, shifts_um,
                                                radius_um, seed, commit), encoding='utf-8')
    files = ['summary.csv', 'calibration.csv', 'drift_recovery.png', 'calibration.png', 'drift_recovery_legend.md',
             'calibration_legend.md', 'METHODS.md', 'manifest.json']
    runs = sorted(str(p.relative_to(output)) for p in (output / 'runs').glob('*/result.json'))
    manifest = dict(
        inputs=[dict(name=d.name, source=d.source, n_units=int(len(d.units)), sessions=_sessions(d),
                     shifted_session=shifted[d.name],
                     truth_offsets_um={str(k): float(v) for k, v in d.truth_offsets.items()}) for d in datasets],
        options=dict(shifts_um=[_json_shift(s) for s in shifts_um], backends=list(backends),
                     registrations=list(registrations), radius_um=float(radius_um), seed=int(seed),
                     backend_options=options_by_backend),
        files=files + runs, git_commit=commit)
    write_json(output / 'manifest.json', manifest, exclusive=True)
    return summary_table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m multiphy.drift_validation',
                                     description='Inject known shifts into extracted project days and measure '
                                                 'drift registration and score calibration.')
    parser.add_argument('--root', metavar='DIR', help='folder of projects; default MULTIPHY_CACHE_ROOT')
    parser.add_argument('--project', required=True, help='project name under the root')
    parser.add_argument('--day', action='append', required=True, help='extracted day of the project (repeat for several)')
    parser.add_argument('--output', required=True, type=Path, help='new output folder; an existing one is refused')
    parser.add_argument('--radius-um', type=float, default=80.0, help='radius_um passed to every backend')
    parser.add_argument('--seed', type=int, default=0, help='seed passed to every backend')
    args = parser.parse_args(argv)
    project = Project.open(cache_root(args.root), args.project)
    validate_datasets([load_project_day(project, d) for d in args.day], args.output,
                      radius_um=args.radius_um, seed=args.seed)
    print(f'Drift validation written: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
