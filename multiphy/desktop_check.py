"""An explicit release diagnostic that exercises the installed Qt application."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from . import __version__, runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='multiphy', description=__doc__, allow_abbrev=False)
    parser.add_argument('--self-check', required=True, metavar='REPORT_JSON')
    parser.add_argument('--root', required=True, metavar='DIR')
    parser.add_argument('--run', metavar='DIR', help='Optional completed run to render without editing curation')
    args = parser.parse_args(argv)
    report_path = Path(args.self_check).expanduser().absolute()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.root).expanduser().resolve()
    report = {'status': 'failed', 'version': __version__, 'frozen': runtime.is_frozen(), 'root': str(root)}
    windows = []
    try:
        if not root.is_dir():
            raise ValueError(f'Select an existing data folder: {root}')
        from .viewer.qt import QtCore, QtSvg, QtWidgets
        from .viewer import logo, theme
        from .launch.window import CommandProcess, LauncherWindow, _child_env, child_command
        from .launch.plugins import load_plugins
        # Diagnostics have their own settings and screenshots, independent of a user's preferences.
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                                str(report_path.parent / 'diagnostic-settings'))
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        theme.apply_theme(app, 'graphite')
        app.setWindowIcon(logo.icon())
        for filename in ('Figtree.ttf', 'JetBrainsMono.ttf'):
            if not (theme.FONT_DIR / filename).is_file():
                raise RuntimeError(f'Missing packaged font: {filename}')
        if not QtSvg.QSvgRenderer(str(logo.svg_path(128))).isValid():
            raise RuntimeError('The packaged application logo could not be loaded')
        program, arguments = child_command(root, ['--help'])
        report.update(fonts=theme.fonts(), logo=True, worker={'program': program, 'arguments': arguments})
        # Exercise the same console worker, pipe decoding and private stdin setup as queued GUI jobs.
        worker = CommandProcess()
        output, errors, exit_codes = [], [], []
        loop = QtCore.QEventLoop()
        timeout = QtCore.QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        worker.stdout_line.connect(output.append)
        worker.stderr_line.connect(errors.append)
        worker.ended.connect(lambda code: (exit_codes.append(code), loop.quit()))
        worker.start(program, arguments, _child_env(), str(runtime.execution_directory()))
        timeout.start(30000)
        loop.exec()
        timeout.stop()
        if not exit_codes:
            worker.kill()
            worker._process.waitForFinished(5000)
            raise RuntimeError('The packaged command worker did not finish within 30 seconds')
        report['worker'].update(exit_code=exit_codes[0], stdout='\n'.join(output), stderr='\n'.join(errors))
        if exit_codes[0] != 0 or not output:
            raise RuntimeError('The command worker failed or returned no output: ' + '\n'.join(errors))
        launcher = LauncherWindow(root, None, plugins=load_plugins(root))
        windows.append(launcher)
        launcher.show()
        app.processEvents()
        shot = report_path.with_suffix('.launcher.png')
        if not launcher.grab().save(str(shot)):
            raise RuntimeError('The launcher screenshot could not be written')
        report['launcher_screenshot'] = str(shot)
        if args.run:
            from .viewer.app import open_prepared, prepare_run
            prepared = prepare_run(Path(args.run).expanduser().resolve())
            if prepared.selection.project.root.resolve() != root:
                raise ValueError('The diagnostic run must belong to the selected data folder')
            # Render all panels while keeping the source run's curation and layout untouched.
            viewer = open_prepared(replace(prepared, curation_dir=None,
                                           layout_file=report_path.with_suffix('.viewer-layout.json')))
            windows.append(viewer)
            viewer.show()
            app.processEvents()
            shot = report_path.with_suffix('.viewer.png')
            if not viewer.grab().save(str(shot)):
                raise RuntimeError('The viewer screenshot could not be written')
            report['viewer_screenshot'] = str(shot)
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for window in reversed(windows):
            window.close()
            window.deleteLater()
        if windows:
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(report_path))
    return 0 if report['status'] == 'passed' else 1
