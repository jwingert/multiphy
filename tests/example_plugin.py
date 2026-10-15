# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a test plugin that adds a reader and a command through PluginApi (Plan C Task 1) | date: 2026-10-01]
"""A launcher plugin for the tests: one reader, ``example.echo``, and one command, ``hello``."""


class _EchoReader:
    """A sort reader that names no files and reads nothing."""
    id = 'example.echo'
    version = '1'
    roles = ('sort',)

    def files(self, options: dict) -> dict:
        return {}

    def read(self, files: dict, options: dict, day=None):
        raise NotImplementedError('example.echo reads nothing')


READER = _EchoReader()


def _configure(parser) -> None:
    parser.add_argument('--word', default='')


def _hello(args, root) -> list:
    return [f'hello {args.project} {args.word}']


def register(api) -> None:
    """Add ``READER`` and the ``hello`` command."""
    api.add_reader(READER)
    api.add_command('hello', _configure, _hello, help='print hello, the project and --word')
# [AGENT EDIT END]
