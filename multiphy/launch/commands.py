"""The launcher commands: ``init_day``, ``init_from``, ``extract``, ``status``, ``status_lines`` and ``run``.

``init_day`` adds one day to a project, creating the project when it is new. Its sort is a phy
folder; ``events`` is an events CSV; the raw file is ``raw``, or else the ``dat_path`` that the
sort's ``params.py`` names, used only when ``uv_per_bit`` is given. A raw file's dtype, offset
and channel count are the ones ``params.py`` declares, whether or not it names a ``dat_path``;
a ``params.py`` without ``n_channels_dat`` refuses the raw file. ``init_from`` adds every day
folder of a ``--from`` folder (``sort/``, ``date.txt``, and optionally ``events.csv`` and one
``raw.*`` file; a folder named ``raw.*`` is not one). It checks every folder, and opens every
file it will hash, before it writes anything, and records each path as it lies inside that
folder: a link is recorded as given, with the file it resolves to now.
``extract`` runs ``extract_day`` on the named days, by default on every day not yet extracted.
``status`` checks each day's recorded files and features without writing, except that
``verify=True`` re-hashes every file and records the first sha256 of a raw file that has none;
a held ``project.lock`` refuses that record at once (``ProjectLocked``). ``status_lines``
renders a status report as one line per day. ``run`` runs one registered backend on a selection
of extracted days, in this process when the backend is available here and no interpreter is
named, else in another Python interpreter. ``init_day`` and ``init_from`` record a ``probe``
type in the project identity when it has none, and refuse one that differs. No module here
imports a lab module.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import re
import sys
from typing import Optional

from ..extract.pipeline import extract_day
from ..contract import registry
from ..contract.inputs import Input
from ..contract.results import Decision
from ..dependency_setup import verified_interpreter
from ..project.selection import Selection, default_days
from ..run import options as opt
from ..run.project_run import check_real_day_run, check_single_unit_days, project_run_path, run_selection
from ..run.subprocess import build_selection_command
from ..run.run_dir import load_result
from ..project.days import add_day
from ..project.folder import NAME_PATTERN, RUNS_DIR, Project
from ..project.identity import FileIdentity, SourceChanged, SourceMissing, retargeted, sha256_file
from ..project.identity import verify as verify_identity
from ..sources.base import file_role, get_reader
from ..sources.binary import _uv_per_bit as binary_uv_per_bit
from ..sources.phy import READER as PHY_READER
from .control import run_captured

RAW_FOUND_NOTE = 'raw file found at dat_path; pass --uv-per-bit to use it'
DATE_PATTERN = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')
FEATURE_LEVELS = ('halves', 'full')
FEATURES_DIR = 'features'


def _plan_day(*, day: str, sort: str, events: Optional[str], raw: Optional[str],
              uv_per_bit: Optional[float]) -> tuple:
    """``(readers, reader_options, notes)`` for one day; reads the sort's ``params.py`` and writes nothing."""
    if raw is not None and uv_per_bit is None:
        raise ValueError('--uv-per-bit is required with a raw file.')
    readers = {'sort': 'phy'}
    options = {'day': day, 'sort_dir': os.path.abspath(sort)}
    if events is not None:
        readers['events'] = 'events_csv'
        options['events_path'] = os.path.abspath(events)
    notes = []
    dat_path = PHY_READER.dat_path(options)
    if raw is None and dat_path is not None and os.path.isfile(dat_path):
        if uv_per_bit is None:
            notes.append(RAW_FOUND_NOTE)
        else:
            raw = dat_path
    if raw is not None:
        readers['raw'] = 'binary'
        options['raw_path'] = os.path.abspath(raw)
        # The binary reader's own rule: a NaN, infinite or non-positive scale is refused here, in
        # init_from's pre-check, rather than when a later day's manifest cannot be written.
        options['uv_per_bit'] = binary_uv_per_bit({'uv_per_bit': uv_per_bit})
        layout = PHY_READER.raw_layout(options)
        options.update(raw_dtype=layout['dtype'], raw_offset=layout['offset'], raw_n_channels=layout['n_channels'])
    return readers, options, notes


def _check_inputs(readers: dict, options: dict) -> None:
    """Refuse a day whose readers name a file that is not there, or one ``add_day`` would hash but cannot open.

    The sort reader checks its own folder. Every file but the raw voltage is opened for reading
    (``PermissionError``), as ``add_day`` hashes it; nothing is read.
    """
    for reader_id in dict.fromkeys(readers.values()):
        reader = get_reader(reader_id)
        for key, path in reader.files(options).items():
            if not os.path.isfile(path):
                raise SourceMissing(f'{file_role(reader, key)} source missing: {os.path.realpath(path)}')
            if file_role(reader, key) != 'raw':  # add_day hashes every file but the raw voltage
                with open(path, 'rb'):
                    pass


def _iso_date(text: str) -> Optional[datetime.date]:
    """``text`` as a date when it is exactly ``YYYY-MM-DD``, else ``None``."""
    if not DATE_PATTERN.match(text):
        return None
    try:
        return datetime.date.fromisoformat(text)
    except ValueError:
        return None


def _check_probe(root: Path, project: str, probe: Optional[str]) -> None:
    """Refuse ``probe`` when ``project`` exists with a different probe recorded; writes nothing."""
    if probe is None:
        return None
    try:
        old = Project.open(root, project).read()['identity']['probe']
    except FileNotFoundError:  # a new project
        return None
    if old is not None and old != probe:
        raise ValueError(f'project {project} probe is {old}; --probe says {probe}')
    return None


def _claim_probe(opened: Project, probe: Optional[str]) -> None:
    """Record ``probe`` in the project identity when it has none; refuse a different one under ``project.lock``.

    A refusal writes nothing. The check is repeated inside ``Project.update``, so a probe
    another process recorded after the first read is refused too.
    """
    if probe is None:
        return None
    if opened.read()['identity']['probe'] == probe:
        return None

    def change(manifest: dict) -> None:
        old = manifest['identity']['probe']
        if old is None:
            manifest['identity']['probe'] = probe
        elif old != probe:
            raise ValueError(f'project {opened.name} probe is {old}; --probe says {probe}')

    opened.update(change)
    return None


def init_day(root: Path, project: str, *, day: str, sort: str, date: str, events: Optional[str] = None,
             raw: Optional[str] = None, uv_per_bit: Optional[float] = None, probe: Optional[str] = None) -> dict:
    """Add ``day`` to ``project``, creating the project when it is new; return ``{'project', 'day', 'files', 'notes'}``.

    ``files`` is the sorted list of recorded file keys. Without ``raw``, the file that the sort's
    ``params.py`` names as ``dat_path`` becomes the raw file when it exists and ``uv_per_bit`` is
    given; when ``uv_per_bit`` is missing, a note says so instead. A raw file needs
    ``uv_per_bit``. The raw file's dtype, offset and channel count come from ``params.py``, whether
    or not it names a ``dat_path``; one without ``n_channels_dat`` refuses the raw file, since a
    flat file has no header to count its channels. The raw rules are applied and the sort's
    ``params.py`` is read before the project is created; ``add_day`` checks the day name, the
    date and the input files after that, so its refusal can leave a new project empty.
    ``probe`` is claimed before the day in one locked check-and-set: a project with a different
    probe refuses it and nothing is written; one with none records it. A day ``add_day`` then
    refuses leaves the probe recorded, so rerunning the same command completes the day.
    """
    readers, options, notes = _plan_day(day=day, sort=sort, events=events, raw=raw, uv_per_bit=uv_per_bit)
    opened = Project.ensure(root, project)
    _claim_probe(opened, probe)
    record = add_day(opened, day, readers=readers, reader_options=options, date=date, date_source='user')
    return {'project': opened.name, 'day': day, 'files': sorted(record['files']), 'notes': notes}


def init_from(root: Path, project: str, from_dir: Path, *, uv_per_bit: Optional[float] = None,
              probe: Optional[str] = None) -> dict:
    """Add every day folder under ``from_dir`` to ``project``; return ``{'project', 'days', 'notes'}``.

    A day folder holds ``sort/``, ``date.txt`` (``YYYY-MM-DD``) and optionally ``events.csv`` and
    one ``raw.*`` file; a folder named ``raw.*`` is not a raw file. Every folder under
    ``from_dir`` is a day folder, a hidden one too, and each is checked, in sorted order, before
    anything is written: a missing ``sort/`` or ``date.txt``, a malformed date, more than one
    ``raw.*`` file, a raw file without ``uv_per_bit`` or with one that is not a positive finite
    number, a raw file whose sort's ``params.py`` declares no ``n_channels_dat``, a day already
    in the project, two days
    sharing a date, a day name the project cannot hold (a hidden folder's name among them), a file
    a reader needs that is not there, and a file ``add_day`` would hash that cannot be opened
    (``PermissionError``) are all refused. Each day is then added by ``init_day`` with its paths
    inside ``from_dir``, never resolved here. Each note is prefixed with its day. A ``probe`` that
    differs from the project's recorded probe is refused with the other checks.
    """
    from_dir = Path(from_dir)
    if not from_dir.is_dir():
        raise FileNotFoundError(f'{from_dir} is not a folder.')
    names = sorted(entry.name for entry in from_dir.iterdir() if entry.is_dir())
    if not names:
        raise ValueError(f'{from_dir} holds no day folders.')
    _check_probe(root, project, probe)
    try:
        recorded = Project.open(root, project).read()['days']
    except FileNotFoundError:  # a new project
        recorded = {}
    dated = {datetime.date.fromisoformat(record['date']): day for day, record in sorted(recorded.items())}

    plans = []
    for name in names:
        folder = from_dir / name
        if not NAME_PATTERN.match(name):
            raise ValueError(f'Day name {name} must match [A-Za-z0-9][A-Za-z0-9_.-]*.')
        if not (folder / 'sort').is_dir():
            raise ValueError(f'{name}: missing sort/')
        if not (folder / 'date.txt').is_file():
            raise ValueError(f'{name}: missing date.txt')
        date = (folder / 'date.txt').read_text(encoding='utf-8').strip()
        parsed = _iso_date(date)
        if parsed is None:
            raise ValueError(f'{name}: date.txt is not YYYY-MM-DD')
        # A folder (or a link to one) is not a raw file; a dangling link counts, and is refused as missing below.
        raws = sorted(path for path in folder.glob('raw.*') if not path.is_dir())
        if len(raws) > 1:
            raise ValueError(f'{name}: more than one raw.* file')
        if raws and uv_per_bit is None:
            raise ValueError('--uv-per-bit is required when a day has a raw file.')
        if name in recorded:
            raise ValueError(f'Day {name} already exists; a day is added once. Put a changed day in a new project.')
        if parsed in dated:
            raise ValueError(f'Days {dated[parsed]} and {name} share the date {parsed.isoformat()}.')
        dated[parsed] = name
        events = folder / 'events.csv'
        plan = {'day': name, 'sort': str(folder / 'sort'), 'date': date,
                'events': str(events) if os.path.lexists(events) else None,
                'raw': str(raws[0]) if raws else None}
        readers, options, _ = _plan_day(day=name, sort=plan['sort'], events=plan['events'], raw=plan['raw'],
                                        uv_per_bit=uv_per_bit)
        _check_inputs(readers, options)
        plans.append(plan)

    notes = []
    for plan in plans:
        result = init_day(root, project, uv_per_bit=uv_per_bit, probe=probe, **plan)
        notes.extend(f"{plan['day']}: {note}" for note in result['notes'])
    return {'project': result['project'], 'days': names, 'notes': notes}


def extract(root: Path, project: str, *, days: Optional[list] = None, tier: Optional[str] = None,
            scratch_root: Optional[Path] = None) -> dict:
    """Run ``extract_day`` on each of ``days``; return ``{'project', 'days': {day: result}}``.

    ``days`` defaults to every day whose ``features['full']`` is ``None``, sorted. A named day
    already extracted is refused by ``extract_day`` (``FileExistsError``); the first refusal
    stops the command, and the days extracted before it stay recorded.
    """
    opened = Project.open(root, project)
    if days is None:
        records = opened.read()['days']
        days = sorted(day for day, record in records.items() if record['features']['full'] is None)
    results = {}
    for day in days:
        results[day] = extract_day(opened, day, tier=tier, scratch_root=scratch_root)
    return {'project': opened.name, 'days': results}


def _check_sources(files: dict, verify: bool) -> dict:
    """``{'missing', 'changed', 'retargeted', 'digests'}`` for one day's recorded files.

    Size and ``mtime_ns`` are always checked; ``verify`` re-hashes every file with a recorded
    sha256. With ``verify``, a file without one (the raw file) is hashed, and its digest is kept
    in ``digests`` when the file is unchanged after the read.
    """
    state = {'missing': False, 'changed': False, 'retargeted': False, 'digests': {}}
    for key in sorted(files):
        identity = FileIdentity.from_dict(files[key])
        try:
            verify_identity(identity, rehash=verify)
            if verify and identity.sha256 is None:
                digest = sha256_file(identity.resolved)
                verify_identity(identity, rehash=False)  # still the recorded file once it was read
                state['digests'][key] = digest
        except SourceChanged:
            state['changed'] = True
        except (FileNotFoundError, NotADirectoryError):  # SourceMissing, or a file removed while it was read
            state['missing'] = True
        if retargeted(identity):
            state['retargeted'] = True
    return state


def _record_digests(manifest: dict, digests: dict) -> None:
    """Record each ``(day, key) -> digest`` with ``sha256_from`` ``'verify'`` where no sha256 is recorded yet."""
    for (day, key), digest in digests.items():
        file = manifest['days'][day]['files'][key]
        if file['sha256'] is None:
            file.update(sha256=digest, sha256_from='verify')


def _sources(state: dict) -> str:
    for name in ('missing', 'changed', 'retargeted'):
        if state[name]:
            return name
    return 'ok'


def _features(day_dir: Path, recorded: dict, sources: str) -> str:
    """``none`` before extraction; ``stale`` when a features manifest hashes to another id, or the sources changed."""
    if recorded['full'] is None:
        return 'none'
    for level in FEATURE_LEVELS:
        manifest = day_dir / FEATURES_DIR / level / 'manifest.json'
        found = sha256_file(manifest) if manifest.is_file() else None
        if found != recorded[level]:
            return 'stale'
    return 'stale' if sources == 'changed' else 'ok'


def _folders(path: Path) -> int:
    """The number of folders in ``path``; 0 when there is no ``path``."""
    return sum(1 for entry in path.iterdir() if entry.is_dir()) if path.is_dir() else 0


def status(root: Path, project: str, *, verify: bool = False) -> dict:
    """Each day's sources, features and runs; return ``{'project', 'days': [row], 'across_days_runs'}``.

    A row is ``{'day', 'date', 'tier', 'sources', 'features', 'runs'}``. ``sources`` is
    ``missing`` when a recorded file is gone, else ``changed`` when one differs from its
    identity, else ``retargeted`` when a recorded path now resolves to another file, else ``ok``.
    ``features`` is ``none`` before extraction; ``stale`` when a manifest under ``features/``
    hashes to an id other than the recorded one, or the sources changed; else ``ok``, also when
    the sources are missing. ``runs`` counts the folders in the day's ``runs/``;
    ``across_days_runs`` counts those in the project's ``runs/``. Without ``verify`` nothing is
    written. ``verify=True`` re-hashes every file, raw included, and records the sha256 of a raw
    file that has none, with ``sha256_from`` ``'verify'``; a sha256 recorded meanwhile that
    differs marks the day's sources ``changed``. A held ``project.lock`` makes that record raise
    ``ProjectLocked`` at once, and nothing is recorded.
    """
    opened = Project.open(root, project)
    manifest = opened.read()
    checks = {day: _check_sources(manifest['days'][day]['files'], verify) for day in sorted(manifest['days'])}
    digests = {(day, key): digest for day, state in checks.items() for key, digest in state['digests'].items()}
    if digests:
        manifest = opened.update(lambda current: _record_digests(current, digests))
        for (day, key), digest in digests.items():
            if manifest['days'][day]['files'][key]['sha256'] != digest:
                checks[day]['changed'] = True
    rows = []
    for day, state in checks.items():
        record = manifest['days'][day]
        sources = _sources(state)
        rows.append({'day': day, 'date': record['date'], 'tier': record['tier'], 'sources': sources,
                     'features': _features(opened.day_dir(day), record['features'], sources),
                     'runs': _folders(opened.day_dir(day) / RUNS_DIR)})
    return {'project': opened.name, 'days': rows, 'across_days_runs': _folders(opened.path / RUNS_DIR)}


def status_lines(report: dict) -> list:
    """One line per day of a ``status`` report, then the across-days line."""
    lines = [f"{row['day']} · {row['date']} · {row['tier'] or '—'} · sources {row['sources']} · "
             f"features {row['features']} · runs {row['runs']}" for row in report['days']]
    lines.append(f"across days · runs {report['across_days_runs']}")
    return lines


def _pythonpath(repo: Path) -> str:
    """``repo`` prepended to the current ``PYTHONPATH``."""
    existing = os.environ.get('PYTHONPATH')
    return os.pathsep.join([str(repo), existing]) if existing else str(repo)


def run(root: Path, project: str, *, backend: str, days: Optional[list] = None, options: Optional[dict] = None,
        python: Optional[str] = None) -> dict:
    """Run ``backend`` on ``days`` of ``project``; return ``{'project', 'days', 'backend', 'run', 'excluded'}``.

    ``days`` defaults to every extracted day, sorted by date, and needs 2 or more of them; one
    named day matches its two temporal halves. Every refusal (days, backend, options, a day or
    one day's half without a single unit) comes before the run folder exists and before any
    warning. A backend that reads timing gets one ``warning · <reason>`` line on stderr per
    evidence fallback (``Selection.evidence``) before it starts. An available
    backend runs in this process unless ``python`` names an interpreter; otherwise it runs in
    ``python`` or the verified interpreter under ``root``, and a failure there raises ``RuntimeError``
    naming the run folder it left. A saved ``result.json`` holding ``proposed_link`` pairs of two
    units of one session prints one ``warning · <n> proposed same-day link(s) · review in the
    viewer`` line on stderr.
    """
    opened = Project.open(root, project)
    if days:
        days = tuple(days)
    else:
        days = default_days(opened)
        if len(days) < 2:
            raise ValueError(f'launch run needs 2 or more extracted days by default; {project} has {len(days)}; '
                             'pass --day for one day')
    selection = Selection(opened, days)
    backend_obj = registry.get(backend)
    check_real_day_run(selection, backend_obj.spec)
    options = opt.validate(backend_obj.spec, options or {})
    check_single_unit_days(selection, selection.dataset().units)
    spec = backend_obj.spec
    if Input.TIMING in spec.requires | spec.optional:
        # validate filled in the spec's default; a spec without the option takes run_evidence's
        evidence = selection.evidence(options['min_repeats']) if 'min_repeats' in options else selection.evidence()
        for reason in evidence.reasons:
            print(f'warning · {reason}', file=sys.stderr)
    output = project_run_path(selection, backend)
    availability = backend_obj.available()
    if availability.ok and python is None:
        run_selection(selection, backend, options, output_dir=output)
    else:
        interpreter = python or verified_interpreter(root=root)
        if interpreter is None:
            raise ValueError(f'{backend} is unavailable here ({availability.reason}) and no verified interpreter '
                             'exists; pass --python EXE')
        program, argv = build_selection_command(opened.path, days, backend, options, output=output,
                                                interpreter=interpreter)
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=_pythonpath(repo), PYTHONDONTWRITEBYTECODE='1')
        proc = run_captured([program, *argv], cwd=repo, env=env)
        if proc.returncode != 0:
            lines = proc.stderr.strip().splitlines()
            raise RuntimeError(f'{backend} failed: {lines[-1] if lines else ""} · run folder {output}')
    if (output / 'result.json').is_file():
        result = load_result(output / 'result.json')
        session_of = dict(zip(result.units.cellid, result.units.session_id))
        proposed = result.pairs[result.pairs.decision == Decision.PROPOSED_LINK.value]
        count = sum(session_of[a] == session_of[b] for a, b in zip(proposed.cellid_a, proposed.cellid_b))
        if count == 1:
            print('warning · 1 proposed same-day link · review in the viewer', file=sys.stderr)
        elif count >= 2:
            print(f'warning · {count} proposed same-day links · review in the viewer', file=sys.stderr)
    return {'project': opened.name, 'days': list(days), 'backend': backend, 'run': output,
            'excluded': len(selection.record()['excluded'])}


def export(root: Path, project: str, *, run: str, out: Optional[Path] = None) -> dict:
    """Export the accepted tracks of across-day run ``run``; return ``{'path', 'tracks', 'units'}``.

    ``tracks`` counts the distinct ``track_id``s of the written CSV and ``units`` its data rows.
    """
    import pandas as pd

    from ..curation import export_run

    opened = Project.open(root, project)
    path = export_run(opened.path, run, out)
    table = pd.read_csv(path, dtype=str, keep_default_na=False, compression=None)
    return {'path': path, 'tracks': table['track_id'].nunique(), 'units': len(table)}
