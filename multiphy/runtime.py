"""Process boundaries shared by source installs and the desktop distribution."""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys


WORKER_MODULES = ('multiphy.launch', 'multiphy.run')
WORKER_FLAG = '--multiphy-module'


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def execution_directory() -> Path:
    """The source checkout, or the directory holding the installed executables."""
    return Path(sys.executable).parent if is_frozen() else Path(__file__).resolve().parents[1]


def _bundled_program() -> Path:
    program = execution_directory() / ('multiphy.exe' if sys.platform == 'win32' else 'multiphy')
    if not program.is_file():
        raise ValueError('The desktop installation is missing multiphy.exe. Reinstall the complete application folder.')
    return program


def module_command(module, arguments, *, interpreter=None, unbuffered=False) -> tuple:
    """Use the bundled console worker, or a real Python interpreter with ``-m``."""
    if module not in WORKER_MODULES:
        raise ValueError(f'Unsupported worker module: {module}')
    if is_frozen() and (interpreter is None or Path(interpreter).absolute() == Path(sys.executable).absolute()):
        return str(_bundled_program()), [WORKER_FLAG, module, *arguments]
    program = Path(interpreter or sys.executable).expanduser()
    return str(program), [*(['-u'] if unbuffered else []), '-m', module, *arguments]


def _inside_bundle(value) -> bool:
    try:
        candidate = Path(value).resolve()
        root = Path(getattr(sys, '_MEIPASS', execution_directory())).resolve()
        return candidate == root or root in candidate.parents
    except (OSError, ValueError):
        return False


def child_environment(*, external=False) -> dict:
    """Keep a checkout importable; never expose frozen modules to external Python."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    if not is_frozen():
        old_path = env.get('PYTHONPATH')
        env['PYTHONPATH'] = os.pathsep.join([str(execution_directory()), old_path]) if old_path else str(execution_directory())
    elif external:
        for key in ('PATH', 'PYTHONPATH'):
            if key in env:
                env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep) if p and not _inside_bundle(p))
        env.pop('PYTHONHOME', None)
        for key in ('LD_LIBRARY_PATH', 'LIBPATH'):
            original = env.pop(key + '_ORIG', None)
            if original is not None:
                env[key] = original
            else:
                env.pop(key, None)
        for key in ('QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            if key in env and _inside_bundle(env[key]):
                env.pop(key)
    return env


@contextmanager
def external_library_path():
    """Stop Windows children inheriting PyInstaller's bundled DLL search directory."""
    if not is_frozen() or sys.platform != 'win32':
        yield
        return
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetDllDirectoryW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p]
    kernel.GetDllDirectoryW.restype = ctypes.c_uint32
    kernel.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
    kernel.SetDllDirectoryW.restype = ctypes.c_bool
    size = kernel.GetDllDirectoryW(0, None)
    previous = ctypes.create_unicode_buffer(size + 1)
    kernel.GetDllDirectoryW(len(previous), previous)
    if not kernel.SetDllDirectoryW(None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not kernel.SetDllDirectoryW(previous.value or None):
            raise ctypes.WinError(ctypes.get_last_error())


def verify_external_interpreter(program) -> None:
    """An external backend environment must contain this multiphy release itself."""
    if not is_frozen():
        return
    from . import __version__
    check = ('import multiphy, sys; '
             'sys.exit(0 if multiphy.__version__ == sys.argv[1] else 2)')
    try:
        with external_library_path():
            result = subprocess.run([str(program), '-c', check, __version__],
                                    env=child_environment(external=True), cwd=execution_directory(),
                                    stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True, encoding='utf-8', timeout=30,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode == 0:
            return
    except (OSError, subprocess.SubprocessError):
        pass
    raise ValueError(f'The selected Python must have multiphy {__version__} and the backend installed. '
                     'Use the matching release with the documented uv setup, then choose its Python interpreter.')
