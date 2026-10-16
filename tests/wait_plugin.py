"""A launcher plugin for the tests: one command, ``wait``, that prints ``waiting`` and sleeps 60 s."""
import time


def _configure(parser) -> None:
    """``wait`` takes no options of its own."""


def _wait(args, root) -> list:
    print('waiting', flush=True)  # main sends a handler's prints to stderr
    time.sleep(60)
    return ['done']


def register(api) -> None:
    """Add the ``wait`` command."""
    api.add_command('wait', _configure, _wait, help='print waiting, then sleep 60 s')
