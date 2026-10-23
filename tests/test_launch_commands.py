"""The launcher commands ``init_day``, ``init_from``, ``extract``, ``status`` and ``status_lines``.

``init_from`` reads a ``--from`` folder of day folders (``sort/``, ``events.csv``, ``raw.*``,
``date.txt``), checks all of them before writing anything and records each day's paths as
given, links unresolved. ``extract`` runs ``extract_day`` on every day not yet extracted.
``status`` reports each day's sources (``missing``, ``changed``, ``retargeted``, ``ok``), its
features (``none``, ``stale``, ``ok``) and its runs; ``verify=True`` re-hashes every file and
records a raw file's first sha256.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import pytest

from multiphy.launch.commands import extract, init_day, init_from, status, status_lines
from multiphy.project.folder import Project, ProjectLocked
from multiphy.project.identity import sha256_file
from multiphy.sources.base import clear_readers_for_tests, get_reader, read_day

from .lab_blocker import run_without_lab
from .synthetic_day import N_CHANNELS, N_SAMPLES, write_day

REPO = Path(__file__).resolve().parents[1]
UV_PER_BIT = 0.195
# The brief's values, written out rather than imported from the code under test.
D1_LINE = 'D1 · 2026-01-01 · raw · sources ok · features ok · runs 0'
ACROSS_LINE = 'across days · runs 0'
RAW_FOUND = 'raw file found at dat_path; pass --uv-per-bit to use it'


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


def _link_day(from_dir: Path, sd, *, date: str = '2026-01-01') -> Path:
    """``from_dir / sd.day``: ``sort`` and ``raw.bin`` link to the synthetic day, ``events.csv`` is a copy."""
    folder = from_dir / sd.day
    folder.mkdir(parents=True)
    (folder / 'sort').symlink_to(sd.sort_dir)
    (folder / 'raw.bin').symlink_to(sd.raw_path)
    shutil.copyfile(sd.events_path, folder / 'events.csv')
    (folder / 'date.txt').write_text(date + '\n')
    return folder


@pytest.fixture
def day(tmp_path):
    """``(DIR, sd)``: the ``--from`` folder ``DIR`` holding D1, and D1's synthetic day under ``tmp_path / 'in'``."""
    sd = write_day(tmp_path / 'in', 'D1')
    from_dir = tmp_path / 'from'
    _link_day(from_dir, sd)
    return from_dir, sd


def _root(tmp_path) -> Path:
    return tmp_path / 'lab'


def _record(root: Path, day: str = 'D1') -> dict:
    return Project.open(root, 'P').read()['days'][day]


def _project_json(root: Path) -> bytes:
    return (root / 'P' / 'project.json').read_bytes()


def _flip_byte(path: Path, offset: int) -> None:
    """Flip one byte of ``path`` and restore its ``mtime_ns``, so only a re-hash can see the change."""
    before = os.stat(path)
    with open(path, 'r+b') as stream:
        stream.seek(offset)
        value = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([value[0] ^ 0xFF]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert os.stat(path).st_mtime_ns == before.st_mtime_ns and os.stat(path).st_size == before.st_size


def _units(root: Path, day: str = 'D1') -> list:
    table = pd.read_csv(root / 'P' / 'days' / day / 'features' / 'full' / 'units.csv', dtype=str,
                        keep_default_na=False)
    return table['cellid'].tolist()


def test_init_from_links(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    (from_dir / 'README.txt').write_text('not a day')  # a file is not a day folder
    result = init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    assert result == {'project': 'P', 'days': ['D1'], 'notes': []}

    record = _record(root)
    files = record['files']
    given = files['spike_times']['given']
    assert Path(given).parts[-3:] == ('D1', 'sort', 'spike_times.npy')
    assert given == str(from_dir / 'D1' / 'sort' / 'spike_times.npy')  # the link as given, never resolved
    assert files['spike_times']['resolved'] == os.path.realpath(sd.sort_dir / 'spike_times.npy')
    assert files['raw']['given'] == str(from_dir / 'D1' / 'raw.bin')
    assert files['raw']['resolved'] == os.path.realpath(sd.raw_path)
    assert files['events']['given'] == str(from_dir / 'D1' / 'events.csv')
    assert files['events']['resolved'] == str(from_dir / 'D1' / 'events.csv')  # a copy, not a link

    assert record['date'] == '2026-01-01' and record['date_source'] == 'user'
    assert record['readers'] == {'sort': 'phy@1', 'events': 'events_csv@1', 'raw': 'binary@1'}
    assert record['reader_options'] == {
        'day': 'D1', 'sort_dir': str(from_dir / 'D1' / 'sort'), 'events_path': str(from_dir / 'D1' / 'events.csv'),
        'raw_path': str(from_dir / 'D1' / 'raw.bin'), 'uv_per_bit': UV_PER_BIT,
        'raw_dtype': '<i2', 'raw_offset': 0, 'raw_n_channels': 8}
    assert record['features'] == {'halves': None, 'full': None}

    # init_day on its own: params.py's dat_path becomes the raw file once uv_per_bit is given.
    one = init_day(root, 'Q', day='D1', sort=str(from_dir / 'D1' / 'sort'), date='2026-01-01',
                   uv_per_bit=UV_PER_BIT)
    assert one['project'] == 'Q' and one['day'] == 'D1' and one['notes'] == []
    assert one['files'] == sorted(files.keys() - {'events'})
    options = Project.open(root, 'Q').read()['days']['D1']['reader_options']
    assert options['raw_path'] == str(sd.raw_path)
    assert (options['raw_dtype'], options['raw_offset'], options['raw_n_channels']) == ('<i2', 0, 8)

    # A folder named raw.* is not a raw file: beside raw.bin it is not a second raw file, and on its
    # own it neither asks for uv_per_bit nor becomes the raw file.
    both = tmp_path / 'both'
    _link_day(both, sd)
    (both / 'D1' / 'raw.backup').mkdir()
    assert init_from(root, 'B', both, uv_per_bit=UV_PER_BIT)['days'] == ['D1']
    assert Project.open(root, 'B').read()['days']['D1']['files']['raw']['given'] == str(both / 'D1' / 'raw.bin')
    alone = tmp_path / 'alone'
    _link_day(alone, sd)
    (alone / 'D1' / 'raw.bin').unlink()
    (alone / 'D1' / 'raw.backup').mkdir()
    result = init_from(root, 'N', alone)
    assert result == {'project': 'N', 'days': ['D1'], 'notes': [f'D1: {RAW_FOUND}']}
    assert 'raw' not in Project.open(root, 'N').read()['days']['D1']['files']


def test_init_from_refuses_incomplete(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    sd2 = write_day(tmp_path / 'in', 'D2', seed=1)
    d2 = _link_day(from_dir, sd2, date='2026-01-02')
    (d2 / 'date.txt').unlink()
    with pytest.raises(ValueError, match=re.escape('D2: missing date.txt')):
        init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    assert not (root / 'P' / 'project.json').exists()
    assert not (root / 'P').exists()

    # Every other refusal also comes before anything is written: each case breaks D2 one way.
    def repair() -> None:
        (d2 / 'date.txt').write_text('2026-01-02\n')
        if os.path.lexists(d2 / 'raw.dat'):
            (d2 / 'raw.dat').unlink()
        if not os.path.lexists(d2 / 'sort'):
            (d2 / 'sort').symlink_to(sd2.sort_dir)

    cases = [
        (ValueError, 'D2: missing sort/', lambda: (d2 / 'sort').unlink(), UV_PER_BIT),
        (ValueError, 'D2: date.txt is not YYYY-MM-DD', lambda: (d2 / 'date.txt').write_text('2026-1-2\n'),
         UV_PER_BIT),
        (ValueError, 'D2: more than one raw.* file', lambda: (d2 / 'raw.dat').symlink_to(sd2.raw_path), UV_PER_BIT),
        (ValueError, '--uv-per-bit is required when a day has a raw file.', lambda: None, None),
        (ValueError, 'Days D1 and D2 share the date 2026-01-01.',
         lambda: (d2 / 'date.txt').write_text('2026-01-01\n'), UV_PER_BIT),
        *[(ValueError, f'binary: uv_per_bit must be a positive number, not {scale!r}.', lambda: None, scale)
          for scale in (float('nan'), float('inf'), 0.0, -0.195)],
        (FileNotFoundError, f"phy: {d2 / 'sort' / 'spike_clusters.npy'} is missing.",
         lambda: (sd2.sort_dir / 'spike_clusters.npy').rename(tmp_path / 'spike_clusters.npy'), UV_PER_BIT),
    ]
    for error, message, breaks, uv_per_bit in cases:
        repair()
        breaks()
        with pytest.raises(error, match=re.escape(message)):
            init_from(root, 'P', from_dir, uv_per_bit=uv_per_bit)
        assert not (root / 'P').exists(), message
    (tmp_path / 'spike_clusters.npy').rename(sd2.sort_dir / 'spike_clusters.npy')
    repair()

    # D1 is sound; a file of D2 that add_day would hash but cannot read is refused before D1 is
    # written, so a rerun after the fix is not refused with 'Day D1 already exists'.
    for unreadable in (d2 / 'events.csv', sd2.sort_dir / 'cluster_group.tsv'):
        mode = unreadable.stat().st_mode
        unreadable.chmod(0)
        try:
            if not os.access(unreadable, os.R_OK):  # permissions bind every user but root
                with pytest.raises(PermissionError, match=re.escape(unreadable.name)):
                    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
                assert not (root / 'P').exists(), unreadable
        finally:
            unreadable.chmod(mode)

    # A day name the project cannot hold, an empty folder and a missing folder are refused as well.
    bad = tmp_path / 'bad'
    _link_day(bad, sd).rename(bad / '_D1')
    with pytest.raises(ValueError, match=re.escape('Day name _D1 must match')):
        init_from(root, 'P', bad, uv_per_bit=UV_PER_BIT)
    hidden = tmp_path / 'hidden'
    _link_day(hidden, sd)
    (hidden / '.D2').mkdir()
    with pytest.raises(ValueError, match=re.escape('Day name .D2 must match')):
        init_from(root, 'P', hidden, uv_per_bit=UV_PER_BIT)
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'empty' / 'README.txt').write_text('not a day')
    with pytest.raises(ValueError, match='holds no day folders'):
        init_from(root, 'P', tmp_path / 'empty', uv_per_bit=UV_PER_BIT)
    with pytest.raises(FileNotFoundError, match='is not a folder'):
        init_from(root, 'P', tmp_path / 'nowhere', uv_per_bit=UV_PER_BIT)
    assert not (root / 'P').exists()

    # Repaired, both days are added.
    assert init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)['days'] == ['D1', 'D2']


def test_status_line_format(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    assert status_lines(status(root, 'P')) == ['D1 · 2026-01-01 · — · sources ok · features none · runs 0',
                                               ACROSS_LINE]
    extract(root, 'P', scratch_root=tmp_path / 'scratch')

    report = status(root, 'P')
    assert report == {'project': 'P', 'across_days_runs': 0, 'days': [
        {'day': 'D1', 'date': '2026-01-01', 'tier': 'raw', 'sources': 'ok', 'features': 'ok', 'runs': 0}]}
    assert status_lines(report) == [D1_LINE, ACROSS_LINE]

    # runs counts folders, never files.
    day_dir = root / 'P' / 'days' / 'D1'
    (day_dir / 'runs' / 'r1').mkdir(parents=True)
    (day_dir / 'runs' / 'note.txt').write_text('x')
    for name in ('a', 'b'):
        (root / 'P' / 'runs' / name).mkdir(parents=True)
    assert status_lines(status(root, 'P')) == [D1_LINE.replace('runs 0', 'runs 1'), 'across days · runs 2']

    # A features manifest that hashes to another id is stale; so are features of a changed source.
    manifest = day_dir / 'features' / 'full' / 'manifest.json'
    original = manifest.read_bytes()
    manifest.write_bytes(original + b'\n')
    assert status(root, 'P')['days'][0]['features'] == 'stale'
    manifest.write_bytes(original)
    assert status(root, 'P')['days'][0]['features'] == 'ok'
    with open(from_dir / 'D1' / 'events.csv', 'a') as stream:
        stream.write('\n')
    assert status_lines(status(root, 'P'))[0] == 'D1 · 2026-01-01 · raw · sources changed · features stale · runs 1'


def test_retargeted_link_reads_recorded_file(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)

    copy = tmp_path / 'copy' / 'sort'
    shutil.copytree(sd.sort_dir, copy)
    for name, column in (('cluster_group.tsv', 'group'), ('cluster_KSLabel.tsv', 'KSLabel')):
        (copy / name).write_text(f'cluster_id\t{column}\n0\tgood\n1\tgood\n2\tmua\n3\tmua\n')
    (from_dir / 'D1' / 'sort').unlink()
    (from_dir / 'D1' / 'sort').symlink_to(copy)

    before = _project_json(root)
    assert status(root, 'P')['days'][0]['sources'] == 'retargeted'
    assert _project_json(root) == before  # status without verify writes nothing
    extract(root, 'P')
    assert _units(root) == ['D1:0', 'D1:1', 'D1:2']
    assert status_lines(status(root, 'P'))[0] == 'D1 · 2026-01-01 · raw · sources retargeted · features ok · runs 0'


def test_verify_hashes_dat_raw(tmp_path, day, monkeypatch):
    from_dir, sd = day
    root = _root(tmp_path)
    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    assert _record(root)['files']['raw']['sha256'] is None
    assert status(root, 'P')['days'][0]['sources'] == 'ok'
    assert _record(root)['files']['raw']['sha256'] is None  # only verify hashes the raw file

    # A sha256 that another process records between the hash and the record is kept; this day's
    # sources then read changed, because the two hashes differ.
    unchanged = _project_json(root)
    other = 'f' * 64
    real_update = Project.update

    def recorded_meanwhile(project, change):
        def other_record(manifest):
            manifest['days']['D1']['files']['raw'].update(sha256=other, sha256_from='extract')
        real_update(project, other_record)
        return real_update(project, change)

    monkeypatch.setattr(Project, 'update', recorded_meanwhile)
    assert status(root, 'P', verify=True)['days'][0]['sources'] == 'changed'
    raw = _record(root)['files']['raw']
    assert (raw['sha256'], raw['sha256_from']) == (other, 'extract')
    monkeypatch.undo()
    (root / 'P' / 'project.json').write_bytes(unchanged)

    # A held project.lock: status without verify writes nothing and still reports; verify refuses at
    # once, without waiting for the lock, and records nothing.
    lock = root / 'P' / 'project.lock'
    lock.write_text('otherhost 1')
    assert status(root, 'P')['days'][0]['sources'] == 'ok'
    started = time.monotonic()
    with pytest.raises(ProjectLocked, match='locked by otherhost 1'):
        status(root, 'P', verify=True)
    assert time.monotonic() - started < 10
    assert _project_json(root) == unchanged
    lock.unlink()

    report = status(root, 'P', verify=True)
    assert report['days'][0]['sources'] == 'ok'
    raw = _record(root)['files']['raw']
    digest = sha256_file(sd.raw_path)
    assert raw['sha256'] == digest
    assert raw['sha256_from'] == 'verify'
    assert _record(root)['files']['spike_times']['sha256_from'] == 'init'

    _flip_byte(sd.raw_path, 1234)
    assert status(root, 'P')['days'][0]['sources'] == 'ok'  # size and mtime_ns cannot see it
    assert status_lines(status(root, 'P', verify=True))[0] == \
        'D1 · 2026-01-01 · — · sources changed · features none · runs 0'
    raw = _record(root)['files']['raw']
    assert (raw['sha256'], raw['sha256_from']) == (digest, 'verify')  # the recorded hash is kept

    # verify re-hashes every other file as well.
    _flip_byte(sd.raw_path, 1234)
    assert status(root, 'P', verify=True)['days'][0]['sources'] == 'ok'
    _flip_byte(sd.sort_dir / 'cluster_group.tsv', 0)
    assert status(root, 'P')['days'][0]['sources'] == 'ok'
    assert status(root, 'P', verify=True)['days'][0]['sources'] == 'changed'


def test_status_after_sources_deleted(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    extract(root, 'P')
    shutil.rmtree(from_dir)
    shutil.rmtree(tmp_path / 'in')

    before = _project_json(root)
    line = status_lines(status(root, 'P'))[0]
    assert 'sources missing · features ok' in line
    assert line == 'D1 · 2026-01-01 · raw · sources missing · features ok · runs 0'
    assert status_lines(status(root, 'P', verify=True))[0] == line
    assert _project_json(root) == before  # nothing to hash, nothing recorded


def test_extracted_day_refused(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    result = extract(root, 'P')
    assert list(result['days']) == ['D1'] and result['project'] == 'P'
    assert result['days']['D1']['tier'] == 'raw'

    before = _project_json(root)
    with pytest.raises(FileExistsError, match='already extracted'):
        extract(root, 'P', days=['D1'])
    with pytest.raises(ValueError, match='already exists'):
        init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    assert _project_json(root) == before
    assert status(root, 'P')['days'][0]['features'] == 'ok'
    assert extract(root, 'P') == {'project': 'P', 'days': {}}  # nothing left to extract


def test_extract_default_days(tmp_path, day):
    from_dir, sd = day
    root = _root(tmp_path)
    init_from(root, 'P', from_dir, uv_per_bit=UV_PER_BIT)
    extract(root, 'P', days=['D1'])

    sd2 = write_day(tmp_path / 'in', 'D2', seed=1)
    with pytest.raises(ValueError, match=re.escape('--uv-per-bit is required with a raw file.')):
        init_day(root, 'P', day='D2', sort=str(sd2.sort_dir), date='2026-01-02', raw=str(sd2.raw_path))
    assert 'D2' not in Project.open(root, 'P').read()['days']
    with pytest.raises(ValueError, match=re.escape('--uv-per-bit is required with a raw file.')):
        init_day(root, 'R', day='D2', sort=str(sd2.sort_dir), date='2026-01-02', raw=str(sd2.raw_path))
    assert not (root / 'R').exists()  # refused before the project is created
    with pytest.raises(ValueError, match=re.escape('Day name _D2 must match')):
        init_day(root, 'S', day='_D2', sort=str(sd2.sort_dir), date='2026-01-02')
    assert Project.open(root, 'S').read()['days'] == {}  # add_day's refusal leaves the new project empty
    added = init_day(root, 'P', day='D2', sort=str(sd2.sort_dir), date='2026-01-02', events=str(sd2.events_path))
    assert added['notes'] == [RAW_FOUND]
    assert 'raw' not in added['files']

    result = extract(root, 'P')
    assert list(result['days']) == ['D2']
    assert result['days']['D2']['tier'] == 'templates'
    assert _units(root, 'D2') == sd2.units
    assert status_lines(status(root, 'P')) == [
        D1_LINE, 'D2 · 2026-01-02 · templates · sources ok · features ok · runs 0', ACROSS_LINE]


def test_flow_without_lab_modules(tmp_path):
    code = f'''
import os
import shutil
from pathlib import Path

from tests.synthetic_day import write_day
from multiphy.launch.commands import extract, init_from, status, status_lines

tmp = Path({str(tmp_path)!r})
sd = write_day(tmp / 'in', 'D1')
folder = tmp / 'from' / 'D1'
folder.mkdir(parents=True)
(folder / 'sort').symlink_to(sd.sort_dir)
(folder / 'raw.bin').symlink_to(sd.raw_path)
shutil.copyfile(sd.events_path, folder / 'events.csv')
(folder / 'date.txt').write_text('2026-01-01\\n')
init_from(tmp / 'lab', 'P', tmp / 'from', uv_per_bit=0.195)
extract(tmp / 'lab', 'P')
for line in status_lines(status(tmp / 'lab', 'P')):
    print(line)
print('lab modules', sorted(m for m in sys.modules if m.startswith(BLOCKED)))
'''
    proc = run_without_lab(code, REPO)
    assert proc.returncode == 0, proc.stderr
    assert D1_LINE in proc.stdout
    assert 'lab modules []' in proc.stdout


@pytest.mark.parametrize('declared', ['dat_path', 'count_only', 'offset_only', 'nothing'])
def test_raw_layout_comes_from_params(tmp_path, declared):
    """project_folder.md, section 5: ``init`` takes a raw file's layout from ``params.py``.

    The raw file holds one channel more than the 8-row geometry, as a sync channel would. Its
    size is a whole number of 8-channel rows, so a reader left to the geometry's row count
    would read 9/8 as many samples of 8 channels. ``n_channels_dat`` reaches ``binary``, which
    refuses the mismatch by name, with the file named as ``dat_path`` (``dat_path``) or not
    (``count_only``). A declared byte offset reaches it too (``offset_only``: 8 channels
    declared, no ``dat_path``), and is refused. With no ``n_channels_dat`` (``nothing``),
    ``init`` refuses the raw file and writes nothing.
    """
    sd = write_day(tmp_path / 'incoming', 'D1')
    extra = sd.root / 'extra_channel.bin'
    np.zeros((N_SAMPLES, N_CHANNELS + 1), dtype='<i2').tofile(extra)
    params = ["dtype = 'int16'", 'sample_rate = 30000.0', 'hp_filtered = False']
    params += {'dat_path': [f'dat_path = {str(extra)!r}', f'n_channels_dat = {N_CHANNELS + 1}', 'offset = 0'],
               'count_only': [f'n_channels_dat = {N_CHANNELS + 1}', 'offset = 0'],
               'offset_only': [f'n_channels_dat = {N_CHANNELS}', 'offset = 16'],
               'nothing': ['offset = 0']}[declared]
    (sd.sort_dir / 'params.py').write_text('\n'.join(params) + '\n')
    lab = tmp_path / 'lab'

    def init():
        return init_day(lab, 'M1', day='D1', sort=str(sd.sort_dir), date=sd.date, raw=str(extra),
                        uv_per_bit=sd.uv_per_bit)

    if declared == 'nothing':
        with pytest.raises(ValueError, match=r'^phy: params\.py has no n_channels_dat; a raw file needs its '
                                             r'declared channel count\.$'):
            init()
        assert not lab.exists()
        return

    init()
    manifest = json.loads((lab / 'M1' / 'project.json').read_text(encoding='utf-8'))
    options = manifest['days']['D1']['reader_options']
    files = {'sort': get_reader('phy').files(options), 'raw': get_reader('binary').files(options)}
    if declared == 'offset_only':
        assert (options['raw_dtype'], options['raw_offset'], options['raw_n_channels']) == ('<i2', 16, N_CHANNELS)
        refusal = r'^binary: a byte offset is not supported\.$'
    else:
        assert (options['raw_dtype'], options['raw_offset'], options['raw_n_channels']) == ('<i2', 0, N_CHANNELS + 1)
        refusal = r'^binary: 9 raw channels differ from 8 geometry channels\.$'
    with pytest.raises(ValueError, match=refusal):
        read_day({'sort': 'phy', 'raw': 'binary'}, files, options)


@pytest.mark.parametrize('dat_exists', [False, True])
def test_no_raw_file_needs_no_declared_count(tmp_path, dat_exists):
    """A day with no raw file is added without ``n_channels_dat``, whatever ``dat_path`` names.

    The named file is gone, as when raw data is deleted after sorting (``dat_exists=False``), or
    there but unused because no ``uv_per_bit`` is given (``True``, with the found-file note).
    Neither reads the raw layout; a raw file the day uses still needs the declared count
    (``test_raw_layout_comes_from_params``).
    """
    sd = write_day(tmp_path / 'incoming', 'D1')
    dat = sd.root / 'sorted_from.bin'
    if dat_exists:
        np.zeros((N_SAMPLES, N_CHANNELS), dtype='<i2').tofile(dat)
    params = [f'dat_path = {str(dat)!r}', "dtype = 'int16'", 'offset = 0', 'sample_rate = 30000.0',
              'hp_filtered = False']
    (sd.sort_dir / 'params.py').write_text('\n'.join(params) + '\n')
    root = _root(tmp_path)

    result = init_day(root, 'P', day='D1', sort=str(sd.sort_dir), date=sd.date)
    assert result['notes'] == ([RAW_FOUND] if dat_exists else [])
    record = _record(root)
    assert 'raw' not in record['readers'] and 'raw' not in record['files']
    assert 'raw_n_channels' not in record['reader_options']
