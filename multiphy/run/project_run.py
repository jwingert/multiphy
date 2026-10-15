# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project run records its selection before the backend starts (Plan B Task 2) | date: 2026-09-28]
"""Run one registered backend on a selection of extracted project days.

By default a project run lives in its selection's ``runs/`` folder (the day's for one day,
the project's for several). An explicit ``output_dir`` is the caller's choice and goes to
``runner.run`` as given (kept under the project root); ``selection_runs`` searches only the
selection's ``runs/`` folder, so a run written elsewhere is not listed there. ``runner.run`` writes the selection's ``selection.json`` and the
dataset's ``pairs.csv`` into the new folder before its manifest, so a failed run still
shows what it was given. Every refusal here comes before the folder exists.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Optional
import uuid

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the single-unit check reads the units frame (event-optional Task 14) | date: 2026-10-10]
import pandas as pd
# [AGENT EDIT END]

from ..contract import registry
from ..contract.backend import BackendSpec
from ..contract.inputs import CheckpointRef, Input
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the backends' own single-unit filter and the halves' session ids (event-optional Task 14) | date: 2026-10-10]
from ..contract.results import single_units
from ..project.selection import HALVES_SESSIONS
# [AGENT EDIT END]
from ..project.selection import Selection
from . import catalog, options as opt, runner


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: days without a shared condition run without events; only a benchmark baseline is refused (event-optional Task 4) | date: 2026-10-09]
def check_real_day_run(selection: Selection, spec: BackendSpec) -> None:
    """Refuse a benchmark baseline across days."""
    if selection.same_day:
        return None
    if not spec.real_days:
        raise ValueError(f'{spec.id} is a benchmark baseline (real_days False); it cannot run across days')
    return None
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day, or one day's half, without a single unit is refused before the run folder exists (event-optional Task 14) | date: 2026-10-10]
def check_single_unit_days(selection: Selection, units: pd.DataFrame) -> None:
    """Refuse a selection with a day, or for one day a half, that holds no single unit of ``units``.

    Single units are the ones ``single_units`` keeps, the backends' own filter; one day's sessions
    are its two halves (``HALVES_SESSIONS``), several days' sessions are the days.
    """
    held = set(single_units(units)[0].session_id)
    sessions = HALVES_SESSIONS if selection.same_day else selection.days
    missing = [session for session in sessions if session not in held]
    if not missing:
        return None
    if not selection.same_day:
        raise ValueError(f'days without a single unit: {", ".join(missing)}; run without them')
    day = selection.days[0]
    if len(missing) == len(HALVES_SESSIONS):
        raise ValueError(f'day {day} has no single unit')
    raise ValueError(f'day {day} half {missing[0]} has no single unit')
# [AGENT EDIT END]


def project_run_path(selection: Selection, backend_id: str) -> Path:
    """A new run folder name under the selection's ``runs/``: UTC stamp, backend id and a random suffix."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return selection.runs_dir() / f'{stamp}_{backend_id}_{uuid.uuid4().hex[:8]}'


def run_selection(selection: Selection, backend_id: str, options: Optional[dict] = None, *,
                  output_dir: Optional[Path] = None) -> Path:
    """Run ``backend_id`` on ``selection``; one day is scored against its halves' source labels."""
    spec = registry.get(backend_id).spec
    check_real_day_run(selection, spec)
    options = opt.validate(spec, options or {})
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a backend that reads timing gets the timed days only; none when no day has trial events (event-optional Task 4) | date: 2026-10-09]
    timing = selection.timing() if Input.TIMING in spec.requires | spec.optional else {}
    extra = {'timing': timing} if Input.TIMING in spec.requires | spec.optional and timing else {}
    # [AGENT EDIT END]
    if options.get('checkpoint'):
        extra['checkpoint'] = CheckpointRef(options['checkpoint'], options.get('checkpoint_probe'))
    dataset = selection.dataset()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: days without single units are refused before the run folder exists (event-optional Task 14) | date: 2026-10-10]
    check_single_unit_days(selection, dataset.units)
    # [AGENT EDIT END]
    record = selection.record_bytes()
    # dataset() and record_bytes() each read the project: a change between the two reads would
    # leave a selection.json that is not the input the manifest names.
    if hashlib.sha256(record).hexdigest() != dataset.metadata['input_manifest_sha256']:
        raise ValueError(f'project {selection.project.name} changed while this run was prepared; run it again')
    return runner.run(dataset, backend_id, options, group_column='recording_group',
                      partition='all' if selection.same_day else None,
                      output_dir=output_dir or project_run_path(selection, backend_id),
                      root=selection.project.root, evaluate=selection.same_day, extra_inputs=extra,
                      files={'selection.json': record,
                             'pairs.csv': dataset.pairs.to_csv(index=False).encode()})


def selection_runs(selection: Selection) -> list:
    """The completed runs of exactly this selection record, newest first."""
    return catalog.find_runs(None, dataset_id=selection.dataset_id,
                             identity={'input_manifest_sha256': selection.sha256()},
                             extra_dirs=[selection.runs_dir()])
# [AGENT EDIT END]
