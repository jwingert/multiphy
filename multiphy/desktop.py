"""Entry points for the packaged command-line and desktop applications."""
import contextlib
import io
import os
from pathlib import Path
import sys

from . import __version__
from .runtime import WORKER_FLAG, WORKER_MODULES


def _configure_streams():
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if stream is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))
        elif hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace', line_buffering=True)


def cli_main(argv=None) -> int:
    """Dispatch only supported worker modules; otherwise use the normal launcher CLI."""
    _configure_streams()
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ['--self-check']:
        from .desktop_check import main
        return main(args)
    if args[:1] == [WORKER_FLAG]:
        if len(args) < 2 or args[1] not in WORKER_MODULES:
            print('error: unsupported multiphy worker module', file=sys.stderr)
            return 1
        if args[1] == 'multiphy.run':
            from .run.__main__ import main
        else:
            from .launch.__main__ import main
        return main(args[2:])
    if args == ['--version']:
        print(f'multiphy {__version__}')
        return 0
    from .launch import __main__ as launch
    # Help must be available before a new user has selected their first data folder.
    if not args or (any(a in ('-h', '--help') for a in args)
                    and launch._leading_root(args)[0] is None and not os.environ.get('MULTIPHY_CACHE_ROOT')):
        try:
            parser = launch.build_parser(backend_spec=launch._backend_spec(args))
        except launch.UsageError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 1
        parser.prog = 'multiphy'
        if not args:
            parser.print_help()
            return 0
        try:
            parser.parse_args(args)
        except SystemExit as exc:
            return int(exc.code or 0)
        except launch.UsageError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 1
    return launch.main(args)


def _settings():
    from .viewer.qt import QtCore
    settings = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, 'multiphy', 'desktop')
    settings.setFallbacksEnabled(False)
    return settings


def remember_root(root):
    _settings().setValue('last_root', str(Path(root).expanduser().resolve()))


def _choose_root(settings, parent=None):
    from .viewer.qt import QtWidgets
    remembered = settings.value('last_root', '', type=str)
    if remembered and Path(remembered).is_dir():
        return str(Path(remembered).resolve())
    return QtWidgets.QFileDialog.getExistingDirectory(
        parent, 'Choose a multiphy data folder (projects and results)', str(Path.home()),
        QtWidgets.QFileDialog.ShowDirsOnly)


def gui_main(argv=None) -> int:
    """Open the launcher, choosing a data folder on first launch without changing CLI policy."""
    args = list(sys.argv[1:] if argv is None else argv)
    # Internal workers never enter the GUI or display a folder dialog.
    if args[:1] in ([WORKER_FLAG], ['--self-check']):
        return cli_main(args)
    _configure_streams()
    from .launch.__main__ import _leading_root
    from .viewer.qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    root, command_index = _leading_root(args)
    if command_index == len(args):
        args.append('gui')
    settings = _settings()
    if args[command_index] == 'gui':
        root = root or os.environ.get('MULTIPHY_CACHE_ROOT') or _choose_root(settings)
        if not root:
            return 0
        args = ['--root', str(root), *args[command_index:]]
        settings.setValue('last_root', str(Path(root).expanduser().resolve()))
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            result = cli_main(args)
    except Exception as exc:
        result = 1
        errors.write(str(exc))
    if result:
        QtWidgets.QMessageBox.critical(None, 'multiphy could not start', errors.getvalue().strip() or 'Startup failed.')
    # Keep ownership of a newly created QApplication until the launcher's event loop returns.
    del app
    return result
