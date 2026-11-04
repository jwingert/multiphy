"""Exercise the built Windows app outside the checkout, with no Python on PATH.

The host Python only creates synthetic input and a curator's acceptance record.
Import, raw extraction, both built-in matchers, verification, export, and Qt
rendering run through the shipped executables. Never uses a user's project.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def smoke(dist, report_dir):
    from multiphy.curation import Curation
    from multiphy.run.run_dir import load_result
    from tests.synthetic_day import write_day

    dist = Path(dist).resolve()
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    cli, gui = dist / 'multiphy.exe', dist / 'multiphy-gui.exe'
    for path in (cli, gui):
        if not path.is_file():
            raise FileNotFoundError(path)
    unused_qt = {'qtvirtualkeyboardplugin.dll', 'qt6virtualkeyboard.dll'}
    unexpected = [path for path in (dist / '_internal').rglob('*')
                  if path.name.lower() in unused_qt]
    assert not unexpected, f'Unused Qt Virtual Keyboard components were bundled: {unexpected}'
    environment = dict(os.environ)
    for key in list(environment):
        if (key.upper().startswith(('PYTHON', 'MULTIPHY', 'QT_'))
                or key.upper() in ('VIRTUAL_ENV', 'CONDA_PREFIX', 'MPLCONFIGDIR')):
            environment.pop(key)
    windows = Path(environment.get('SystemRoot', 'C:/Windows'))
    environment.update(PATH=os.pathsep.join((str(windows / 'System32'), str(windows))),
                       PYTHONUTF8='0', QT_QPA_PLATFORM='offscreen')
    checks = []
    with tempfile.TemporaryDirectory(prefix='multiphy desktop é 路径 ') as temporary:
        work = Path(temporary)
        environment['MPLCONFIGDIR'] = str(work / 'matplotlib')
        environment['APPDATA'] = str(work / 'appdata')
        environment['LOCALAPPDATA'] = str(work / 'localappdata')
        root, inputs, project = work / 'data', work / 'recordings', 'DesktopSmoke'
        root.mkdir()

        def run(label, arguments, executable=cli):
            started = time.monotonic()
            result = subprocess.run([str(executable), *map(str, arguments)], cwd=work,
                                    env=environment, capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=240,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            (report_dir / (label + '.log')).write_text(
                result.stdout + '\n' + result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'{label} exited {result.returncode}:\n{result.stdout}\n{result.stderr}')
            checks.append({'check': label, 'seconds': round(time.monotonic() - started, 2)})
            print(f'PASS {label}', flush=True)
            return result.stdout

        assert 'multiphy' in run('version', ['--version'])
        assert 'extract' in run('help', ['--help'])
        for index, day in enumerate(('D1', 'D2')):
            # These random spikes have no stimulus response. Match their raw
            # waveforms and physiology using the normal no-event chunk path.
            write_day(inputs, day, seed=index, date=f'2026-01-{index + 1:02d}', events=False)
        leading = ['--root', str(root)]
        run('import', [*leading, 'init', project, '--from', inputs, '--uv-per-bit', '0.195'])
        # The GUI uses this dispatch rather than treating the executable as python -m.
        run('extract-worker', ['--multiphy-module', 'multiphy.launch', *leading,
                               'extract', project, '--tier', 'raw'])
        status = run('verify', [*leading, 'status', project, '--verify'])
        assert status.count('sources ok') == status.count('features ok') == 2, status
        runs = root / project / 'runs'
        matched = {}
        for backend in ('functional_likelihood', 'pair_logistic'):
            before = set(runs.iterdir()) if runs.exists() else set()
            run(backend, [*leading, 'run', project, '--backend', backend,
                          '--min-spikes', '20', '--min-pairs', '1'])
            created = set(runs.iterdir()) - before
            assert len(created) == 1, created
            run_dir = created.pop()
            result = load_result(run_dir / 'result.json')
            assert set(result.units.cellid) == {f'{day}:{unit}' for day in ('D1', 'D2') for unit in range(3)}
            assert result.backend_id == backend
            matched[backend] = run_dir

        run_dir = matched['functional_likelihood']
        result = load_result(run_dir / 'result.json')
        curator = Curation.open(root / project / 'curation' / 'runs' / run_dir.name, result)
        try:
            track_id, track = max(curator.state.tracks.items(), key=lambda item: len(item[1]['members']))
            curator.do('accept', cluster_id=track_id)
            expected_members = set(track['members'])
        finally:
            curator.close()
        exported = work / 'accepted é.csv'
        run('export', [*leading, 'export', project, '--run', run_dir.name, '--out', exported])
        with exported.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert {row['cellid'] for row in rows} == expected_members
        assert all(row['review_status'] == 'accepted' for row in rows)
        assert all(row['backend_id'] == 'functional_likelihood' for row in rows)

        # Use the windowed entry point: stdout=None and its console worker are
        # separate failure cases from merely importing PySide6 in a console app.
        gui_report = report_dir / 'gui.json'
        run('gui', ['--self-check', gui_report, '--root', root, '--run', run_dir], executable=gui)
        gui_result = json.loads(gui_report.read_text(encoding='utf-8'))
        assert gui_result.get('frozen') is True, gui_result
        assert gui_result.get('status') == 'passed', gui_result
        assert gui_result['worker']['exit_code'] == 0, gui_result
        for screenshot in ('gui.launcher.png', 'gui.viewer.png'):
            assert (report_dir / screenshot).stat().st_size > 1000
        summary = {'status': 'passed', 'dist': str(dist), 'checks': checks,
                   'data_path_has_spaces_and_unicode': True, 'python_on_path': False,
                   'gui': gui_result}
        (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, required=True, help='Folder containing both executables')
    parser.add_argument('--report-dir', type=Path, help='Logs, screenshots and summary; default beside dist')
    args = parser.parse_args()
    smoke(args.dist, args.report_dir or args.dist.resolve().parent / 'smoke-report')
    print('Desktop smoke passed: import, extraction, both matchers, accepted export, launcher and viewer.')


if __name__ == '__main__':
    main()
