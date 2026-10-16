# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher window lists a root's projects, days and runs and opens a run in its viewer (launcher-gui Task 5) | date: 2026-10-07]
"""The launcher window: one window over a lab root's projects, days and completed runs.

``list_projects``, ``day_rows`` and ``run_groups`` read what the window lists. ``LauncherWindow``
shows the root, a project picker, the project's days (checkable) and its runs grouped as
``across days`` then one group per day, and opens the selected run with ``Curate…`` or
``Inspect…``. The window writes no project data itself. ``run_window`` runs it as an
application. Qt comes only from ``..viewer.qt``; no lab module is imported.
"""
from __future__ import annotations

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the command tabs decode child output, signal a child and quote its argv (launcher-gui Task 6) | date: 2026-10-07]
import codecs
import os
import shlex
import signal
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: log lines are toned by pattern (launcher-gui Task 9) | date: 2026-10-07]
import re
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the evidence tags read Selection.evidence's default min_repeats for a backend without the option (event-optional Task 11) | date: 2026-10-10]
import inspect
# [AGENT EDIT END]
from pathlib import Path
import sys
from typing import Optional

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the command tabs read the backend registry, the verified interpreter and backend_refusal (launcher-gui Task 6) | date: 2026-10-07]
from .. import dependency_setup
from ..contract import registry
from ..contract.names import backend_label
# [AGENT EDIT END]
from ..project.folder import MANIFEST_FILE, RUNS_DIR, Project
from ..run import catalog
from ..run.config import cache_root
from ..viewer.qt import QtCore, QtGui, QtWidgets
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shared Settings and Help menus and the theme (launcher-gui Task 9) | date: 2026-10-07]
from ..viewer import display, menus, theme
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run_window sets the saved unit palette (graphite-palettes Task 4) | date: 2026-10-11]
from ..viewer import unit_palettes
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run_window sets the logo icon (logo) | date: 2026-10-13]
from ..viewer import logo
# [AGENT EDIT END]
# [AGENT EDIT END]
from .__main__ import build_parser
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the command tabs build argv from the forms and start children with the CLI's PYTHONPATH (launcher-gui Task 6) | date: 2026-10-07]
from . import forms as form_module  # backend_refusal is read through the module, so tests can replace it
from .commands import _pythonpath
from .forms import BOUND, command_argv, commands_of, form_fields, has_value
# [AGENT EDIT END]
from .plugins import PluginApi, load_plugins
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run form shows the checked days' evidence as tags and greys the choices only events measure (event-optional Task 11) | date: 2026-10-10]
from ..contract.inputs import evidence_tags
from ..project.selection import Selection
from .forms import EVENT_CHOICES

# launch run's min_repeats for a backend that reads timing without declaring the option
_EVIDENCE_MIN_REPEATS = inspect.signature(Selection.evidence).parameters['min_repeats'].default
# [AGENT EDIT END]

ACROSS_DAYS = 'across days'
TITLE = 'Cross-day unit match launcher'
_ENTRY_ROLE = QtCore.Qt.UserRole
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the command tabs' labels, the repo children run from, and the commands that need a readable project (launcher-gui Task 6) | date: 2026-10-07]
TAB_LABELS = {'init': 'Initialization', 'extract': 'Extract features', 'run': 'Run matching',
              'export': 'Export tracks', 'status': 'Status'}
REPO = Path(__file__).resolve().parents[2]
_BUILTIN_TABS = ('init', 'extract', 'run', 'export', 'status')
_NEEDS_PROJECT = ('extract', 'run', 'status', 'export')  # disabled while project_ok is False (D38)
_UNSET = object()
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a cancelled command is killed when it still runs after this grace (launcher-gui Task 7) | date: 2026-10-07]
KILL_AFTER_MS = 10000  # read each time the kill timer starts
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher opens at 1280 x 860, at most 90% of the screen, with 18 px edges and room between panes (launcher spacing) | date: 2026-10-13]
OPEN_SIZE = (1280, 860)  # the size every open starts at, before the screen cap; nothing is remembered
SCREEN_SHARE = 0.9       # at most this share of the screen's available width and height
EDGE = 18                # px between the window's edge and its panes
GAP = 16                 # px between panes: the splitter handles
SPACING = 8              # px between the widgets of a pane, and around a tab's form
FORM_ROWS = 12           # px between form rows; a row's label sits FORM_ROWS + 4 px from its field
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the log tone of a line: an INFO log line is muted, an error, a kill or a nonzero exit is reject (launcher-gui Task 9) | date: 2026-10-07]
INFO_LINE = re.compile(r'^\[[^\]]* INFO\]')
_EXIT_LINE = re.compile(r'exit (-?\d+)')


def log_tone(line: str) -> str:
    """The colour token of a log line: ``muted`` for an INFO log line; ``reject`` for ``error: …``, ``killed``
    and a nonzero ``exit N``; ``text`` for any other line (D55)."""
    if INFO_LINE.match(line):
        return 'muted'
    if line.startswith('error:') or line == 'killed':
        return 'reject'
    m = _EXIT_LINE.fullmatch(line)
    if m and int(m.group(1)) != 0:
        return 'reject'
    return 'text'
# [AGENT EDIT END]


def child_command(root: Path, argv: list) -> tuple:
    """``(program, arguments)`` of the CLI child that runs ``argv`` under ``root``; ``--root`` is always
    passed, so an inherited ``MULTIPHY_CACHE_ROOT`` cannot redirect it."""
    return sys.executable, ['-m', 'multiphy.launch', '--root', str(root), *argv]


def _child_env() -> dict:
    """The child's environment: the repo first on ``PYTHONPATH``, no bytecode, UTF-8 output."""
    return dict(os.environ, PYTHONPATH=_pythonpath(REPO), PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')


class CommandProcess(QtCore.QObject):
    """One child process with separate stdout and stderr, read as UTF-8 lines (D42, D43).

    ``started`` forwards the process's start; ``stdout_line`` and ``stderr_line`` carry one complete
    line each, a trailing ``\\r`` dropped, and a last partial line is emitted at the end; ``ended(code)``
    fires exactly once, with the exit code, ``-1`` after a crash or a failed start.
    """

    started = QtCore.Signal()
    stdout_line = QtCore.Signal(str)
    stderr_line = QtCore.Signal(str)
    ended = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = QtCore.QProcess(self)
        self._process.setProcessChannelMode(QtCore.QProcess.SeparateChannels)
        self._process.started.connect(self.started.emit)
        self._process.readyReadStandardOutput.connect(lambda: self._read('out'))
        self._process.readyReadStandardError.connect(lambda: self._read('err'))
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._error)
        self._running = False
        self._reset()

    def _reset(self) -> None:
        self._decoders = {key: codecs.getincrementaldecoder('utf-8')(errors='replace') for key in ('out', 'err')}
        self._partial = {'out': '', 'err': ''}

    def _signal(self, key: str):
        return self.stdout_line if key == 'out' else self.stderr_line

    def _feed(self, key: str, data: bytes, final: bool = False) -> None:
        text = self._partial[key] + self._decoders[key].decode(data, final)
        *lines, self._partial[key] = text.split('\n')
        for line in lines:
            self._signal(key).emit(line[:-1] if line.endswith('\r') else line)

    def _read(self, key: str) -> None:
        data = self._process.readAllStandardOutput() if key == 'out' else self._process.readAllStandardError()
        self._feed(key, bytes(data))

    def _end(self, code: int) -> None:
        if not self._running:  # ended fires exactly once
            return
        for key in ('out', 'err'):
            self._read(key)
            self._feed(key, b'', final=True)
            rest, self._partial[key] = self._partial[key], ''
            if rest:
                self._signal(key).emit(rest[:-1] if rest.endswith('\r') else rest)
        self._running = False
        self.ended.emit(code)

    def _finished(self, exit_code, exit_status) -> None:
        self._end(exit_code if exit_status == QtCore.QProcess.NormalExit else -1)

    def _error(self, error) -> None:
        if error == QtCore.QProcess.FailedToStart and self._running:
            self.stderr_line.emit(f'error: {self._process.errorString()}')
            self._end(-1)

    def start(self, program: str, arguments: list, env: dict, cwd: str) -> None:
        """Start ``program`` with ``arguments`` in ``cwd`` with exactly the environment ``env``."""
        environment = QtCore.QProcessEnvironment()
        for key, value in env.items():
            environment.insert(key, value)
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(cwd)
        self._reset()
        self._running = True
        self._process.start(program, list(arguments))

    def interrupt(self) -> None:
        """Send SIGINT to the running child; a child that just exited is ignored."""
        pid = self._process.processId()
        if pid > 0 and self._process.state() == QtCore.QProcess.Running:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:  # the child just exited
                pass

    def kill(self) -> None:
        """Kill the child."""
        self._process.kill()

    def running(self) -> bool:
        """True from ``start`` until ``ended``."""
        return self._running
# [AGENT EDIT END]


def list_projects(root: Path) -> list:
    """The sorted names of the folders directly under ``root`` that hold ``project.json``; ``[]``
    when ``root`` does not exist."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(child.name for child in root.iterdir() if (child / MANIFEST_FILE).is_file())


def day_rows(project: Project) -> list:
    """``[(day, tag)]`` sorted by ``(date, day)``; ``tag`` is the day's tier, ``'none'`` before extraction."""
    records = project.read()['days']
    days = sorted(records, key=lambda day: (records[day]['date'], day))
    return [(day, records[day]['tier'] or 'none') for day in days]


def _entries(folder: Path) -> list:
    """The completed runs in ``folder``, newest first; none when the folder does not exist."""
    if not folder.exists():  # a project, or a day, before its first run
        return []
    entries = [catalog.read_run(sub) for sub in folder.iterdir() if sub.is_dir()]
    return sorted((e for e in entries if e is not None), key=lambda e: e.finished, reverse=True)


def run_groups(project: Project) -> list:
    """``[(group, entries)]``: ``'across days'`` for the project's ``runs/``, then each day's ``runs/``
    in ``day_rows`` order; a group without a completed run is left out (D4, D34)."""
    folders = [(ACROSS_DAYS, project.path / RUNS_DIR)]
    folders += [(day, project.day_dir(day) / RUNS_DIR) for day, _tag in day_rows(project)]
    groups = []
    for group, folder in folders:
        entries = _entries(folder)
        if entries:
            groups.append((group, entries))
    return groups


def _forms(plugins: PluginApi) -> dict:
    """``{command: [Field]}`` for every command of the CLI's parser but ``gui`` (D72)."""
    parser = build_parser(plugins.commands)
    return {name: form_fields(sub, BOUND.get(name, frozenset()))
            for name, sub in commands_of(parser).items() if name != 'gui'}


class _ElidedLabel(QtWidgets.QLabel):
    """A read-only label that shows its full text elided in the middle to its width (D52)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ''
        self.setMinimumWidth(1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def _elide(self) -> None:
        metrics = QtGui.QFontMetrics(self.font())
        self.setText(metrics.elidedText(self._full, QtCore.Qt.ElideMiddle, max(self.width(), 1)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room between form rows (launcher spacing) | date: 2026-10-13]
def _space_form(form: QtWidgets.QFormLayout) -> None:
    """``FORM_ROWS`` px between ``form``'s rows; ``FORM_ROWS + 4`` px between a label and its field."""
    form.setVerticalSpacing(FORM_ROWS)
    form.setHorizontalSpacing(FORM_ROWS + 4)
# [AGENT EDIT END]


class LauncherWindow(QtWidgets.QMainWindow):
    """The launcher window over ``root``; ``project`` is shown first (D31, D74).

    ``plugins`` gives the plugin commands; ``process_factory`` and ``interpreter_lookup`` are kept
    for the command tabs. ``forms`` holds each command's form fields, read off the CLI's parser.
    """

    def __init__(self, root: Path, project: Optional[str] = None, *, plugins: PluginApi, process_factory=None,
                 interpreter_lookup=None):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.root = cache_root(root)
        self.plugins = plugins
        self.process_factory = process_factory
        self.interpreter_lookup = interpreter_lookup
        self.forms = _forms(plugins)
        self.viewers = []
        self.project_ok = False
        self.new_project = None
        self._filled = None  # (root, project) of the last successful fill
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every line the log pane shows, re-coloured on a theme switch (launcher-gui Task 9) | date: 2026-10-07]
        self.log_lines = []
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the command tabs' state: the running job, the queue, the forms' widgets and the cached interpreter (launcher-gui Task 6) | date: 2026-10-07]
        self.queue = []
        self.submitted = None
        self.process = None
        self.last_stdout = ''
        self.run_buttons = {}
        self.form_widgets = {}
        self._busy = False
        self._project = None  # the opened Project of the last successful fill
        self._interpreter = _UNSET  # interpreter_lookup's answer or its exception, looked up at most once
        self._prefilled = _UNSET  # the project the init and plugin forms were last filled with
        self._backend_fields = []
        self._specs = {}
        self._building = True
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the cancel state of the running job and the close-after-end mark (launcher-gui Task 7) | date: 2026-10-07]
        self._started = False  # the running job's process emitted started
        self._cancel_requested = False  # the Cancel path ran for the running job
        self._cancel_pending = False  # cancel asked before started; sent on started (D70)
        self._killed = False  # the kill timer killed the running job
        self._close_after = False  # the close bar's Yes was clicked: close on ended (D79)
        self._kill_timer = QtCore.QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_due)
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the checked days' evidence, or the error that stopped it, as of the last refresh (event-optional Task 11) | date: 2026-10-10]
        self._evidence = None  # the RunEvidence of the checked days, None without one
        self._evidence_error = None  # the message of a bad min_repeats or an evidence error (D55)
        self._evidence_labels = []
        # [AGENT EDIT END]
        name = project
        if project is not None:
            try:
                name = Project._name_under(self.root, project)
            except ValueError:  # not a folder directly under the root: kept as given
                name = project
        self._wanted = (name,)  # the next reload selects this name, or the first project for None
        self._build()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every open starts at OPEN_SIZE within the screen (launcher spacing) | date: 2026-10-13]
        self._open_size()
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Settings holds the theme and Lab root…; the log pane follows a theme switch (launcher-gui Task 9) | date: 2026-10-07]
        menus.install_menus(self, (self.root_action,))
        theme.notifier().changed.connect(self._theme_changed)
        # [AGENT EDIT END]
        self.reload()
        if name is not None and self.project_combo.currentIndex() < 0:  # a name not under the root
            self.new_project = name
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the init and plugin forms show the new project; the Run buttons and backends start current (launcher-gui Task 6) | date: 2026-10-07]
        self._prefill_projects()
        self._refresh_commands()
        # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every open starts at OPEN_SIZE within the screen (launcher spacing) | date: 2026-10-13]
    def _open_size(self) -> None:
        """Size the window to ``OPEN_SIZE``, at most ``SCREEN_SHARE`` of the screen's available width and height;
        with no screen, ``OPEN_SIZE``."""
        width, height = OPEN_SIZE
        screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            width = min(width, int(area.width() * SCREEN_SHARE))
            height = min(height, int(area.height() * SCREEN_SHARE))
        self.resize(width, height)
    # [AGENT EDIT END]

    def _build(self) -> None:
        self.root_label = _ElidedLabel()
        self.root_label.set_full_text(str(self.root))
        self.root_action = QtGui.QAction('Lab root…', self)
        self.root_action.triggered.connect(self._choose_root)
        self.addAction(self.root_action)
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.currentIndexChanged.connect(lambda _index: self.reload())
        self.days_list = QtWidgets.QListWidget()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a check change regreys the backends and recomputes the Run buttons (launcher-gui Task 6) | date: 2026-10-07]
        self.days_list.itemChanged.connect(lambda _item: self._refresh_commands())
        # [AGENT EDIT END]
        self.run_tree = QtWidgets.QTreeWidget()
        self.run_tree.setHeaderHidden(True)
        self.run_tree.itemSelectionChanged.connect(self._update_open_button)
        self.open_button = QtWidgets.QPushButton('Curate…')
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_run)

        left = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room between the left column's widgets (launcher spacing) | date: 2026-10-13]
        column.setSpacing(SPACING)
        # [AGENT EDIT END]
        column.addWidget(self.root_label)
        column.addWidget(self.project_combo)
        column.addWidget(QtWidgets.QLabel('Days'))
        column.addWidget(self.days_list, 1)
        column.addWidget(QtWidgets.QLabel('Runs'))
        column.addWidget(self.run_tree, 2)
        column.addWidget(self.open_button)
        self.tabs = QtWidgets.QTabWidget()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one tab per command (launcher-gui Task 6) | date: 2026-10-07]
        self._build_tabs()
        # [AGENT EDIT END]
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(left)
        top.addWidget(self.tabs)
        top.setStretchFactor(1, 1)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a gap between the left column and the tabs (launcher spacing) | date: 2026-10-13]
        top.setHandleWidth(GAP)
        # [AGENT EDIT END]
        self.log_pane = QtWidgets.QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the log pane uses the system fixed-width font (launcher-gui Task 9) | date: 2026-10-07]
        self.log_pane.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the close bar above the log pane, Cancel beside it, the queue list under it (launcher-gui Task 7) | date: 2026-10-07]
        bottom = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(bottom)
        stack.setContentsMargins(0, 0, 0, 0)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room between the log pane's rows and beside Cancel (launcher spacing) | date: 2026-10-13]
        stack.setSpacing(SPACING)
        # [AGENT EDIT END]
        stack.addWidget(self._build_close_bar())
        pane = QtWidgets.QHBoxLayout()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room between the log pane's rows and beside Cancel (launcher spacing) | date: 2026-10-13]
        pane.setSpacing(SPACING)
        # [AGENT EDIT END]
        pane.addWidget(self.log_pane, 1)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(lambda _checked=False: self._cancel())
        pane.addWidget(self.cancel_button, 0, QtCore.Qt.AlignTop)
        stack.addLayout(pane, 1)
        stack.addWidget(self._build_queue_box())
        body = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        body.addWidget(top)
        body.addWidget(bottom)
        # [AGENT EDIT END]
        body.setStretchFactor(0, 3)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an edge around the panes and a gap above the log pane (launcher spacing) | date: 2026-10-13]
        body.setContentsMargins(EDGE, EDGE, EDGE, EDGE)
        body.setHandleWidth(GAP)
        # [AGENT EDIT END]
        self.setCentralWidget(body)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the close bar and the queue box (launcher-gui Task 7) | date: 2026-10-07]
    def _build_close_bar(self) -> QtWidgets.QFrame:
        """``close_bar``: the close confirmation, hidden until a close arrives while a command runs (D18)."""
        self.close_bar = QtWidgets.QFrame()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the style sheet finds the close bar by name (launcher-gui Task 9) | date: 2026-10-07]
        self.close_bar.setObjectName('close_bar')
        # [AGENT EDIT END]
        line = QtWidgets.QHBoxLayout(self.close_bar)
        line.setContentsMargins(0, 0, 0, 0)
        self.close_label = QtWidgets.QLabel()
        self.close_yes = QtWidgets.QPushButton('Yes')
        self.close_yes.clicked.connect(lambda _checked=False: self._close_confirmed())
        self.close_no = QtWidgets.QPushButton('No')
        self.close_no.clicked.connect(lambda _checked=False: self.close_bar.hide())
        line.addWidget(self.close_label, 1)
        line.addWidget(self.close_yes)
        line.addWidget(self.close_no)
        self.close_bar.hide()
        return self.close_bar

    def _build_queue_box(self) -> QtWidgets.QFrame:
        """``queue_box``: the waiting jobs, each removable, shown only while ``queue`` is not empty (D8)."""
        self.queue_box = QtWidgets.QFrame()
        column = QtWidgets.QVBoxLayout(self.queue_box)
        column.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel('waiting')
        hint.setProperty('role', 'hint')
        self.queue_list = QtWidgets.QListWidget()
        self.clear_button = QtWidgets.QPushButton('Clear')
        self.clear_button.clicked.connect(lambda _checked=False: self.clear_queue())
        column.addWidget(hint)
        column.addWidget(self.queue_list)
        column.addWidget(self.clear_button, 0, QtCore.Qt.AlignLeft)
        self.queue_box.hide()
        return self.queue_box
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each log line is kept and drawn in the colour of its tone; a theme switch redraws them (launcher-gui Task 9) | date: 2026-10-07]
    def append_log(self, line: str) -> None:
        """Append one line to ``log_lines`` and to the log pane in the colour of ``log_tone(line)``."""
        self.log_lines.append(line)
        self._insert_log_line(line, first=len(self.log_lines) == 1)
        self._scroll_log_to_end()

    def _insert_log_line(self, line: str, first: bool) -> None:
        """Insert ``line`` as the pane's last block (its only block when ``first``)."""
        fmt = QtGui.QTextCharFormat()
        fmt.setForeground(QtGui.QColor(display.colors()[log_tone(line)]))
        cursor = QtGui.QTextCursor(self.log_pane.document())
        cursor.movePosition(QtGui.QTextCursor.End)
        if not first:
            cursor.insertBlock(QtGui.QTextBlockFormat(), fmt)
        cursor.insertText(line, fmt)

    def _scroll_log_to_end(self) -> None:
        bar = self.log_pane.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _theme_changed(self, _name: str) -> None:
        """Redraw every line of ``log_lines`` in the new theme's colours."""
        self.log_pane.clear()
        for index, line in enumerate(self.log_lines):
            self._insert_log_line(line, first=index == 0)
        self._scroll_log_to_end()
    # [AGENT EDIT END]

    def _fill_combo(self) -> None:
        """Refill the project combo; keep the current name when listed (signals blocked)."""
        if self._wanted is not None:  # the constructor's project, or set_root's first project
            (wanted,), self._wanted = self._wanted, None
            first = wanted is None
        else:
            index = self.project_combo.currentIndex()
            wanted, first = (self.project_combo.currentText() if index >= 0 else None), False
        names = list_projects(self.root)
        index = names.index(wanted) if wanted in names else (0 if first and names else -1)
        blocked = self.project_combo.blockSignals(True)
        try:
            self.project_combo.clear()
            self.project_combo.addItems(names)
            self.project_combo.setCurrentIndex(index)
        finally:
            self.project_combo.blockSignals(blocked)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: reload refills the lists without a refresh per check, then prefills the project fields and recomputes the commands (launcher-gui Task 6) | date: 2026-10-07]
    def reload(self) -> None:
        """Refill the project combo, then the days and runs of its project (D38); an error is logged
        and leaves both lists empty. Then the init and plugin forms follow the combo, and the Run
        buttons and backend items are recomputed."""
        blocked = self.days_list.blockSignals(True)
        try:
            self._fill_lists()
        finally:
            self.days_list.blockSignals(blocked)
        self._prefill_projects()
        self._refresh_commands()

    def _fill_lists(self) -> None:
        """Refill the project combo, then the days and runs of its project; an error is logged and
        leaves both lists empty."""
        self._project = None
    # [AGENT EDIT END]
        checked = {self.days_list.item(i).data(_ENTRY_ROLE) for i in range(self.days_list.count())
                   if self.days_list.item(i).checkState() == QtCore.Qt.Checked}
        self.days_list.clear()
        self.run_tree.clear()
        self.project_ok = False
        try:
            self._fill_combo()
            if self.project_combo.currentIndex() < 0:
                self._filled = None
                return
            project = Project.open(self.root, self.project_combo.currentText())
            rows, groups = day_rows(project), run_groups(project)
        except Exception as exc:
            self._filled = None
            self.days_list.clear()
            self.run_tree.clear()
            self.append_log(f'error: {exc}')
            self._update_open_button()
            return
        keep = checked if self._filled == (self.root, project.name) else set()
        for day, tag in rows:
            item = QtWidgets.QListWidgetItem(f'{day} · {tag}')
            item.setData(_ENTRY_ROLE, day)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if day in keep else QtCore.Qt.Unchecked)
            self.days_list.addItem(item)
        for group, entries in groups:
            top = QtWidgets.QTreeWidgetItem([group])
            for entry, label in zip(entries, catalog.run_labels(entries)):
                child = QtWidgets.QTreeWidgetItem([label])
                child.setToolTip(0, catalog.run_tooltip(entry))
                child.setData(0, _ENTRY_ROLE, entry)
                top.addChild(child)
            self.run_tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._filled = (self.root, project.name)
        self.project_ok = True
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the opened project is kept for backend_refusal (launcher-gui Task 6) | date: 2026-10-07]
        self._project = project
        # [AGENT EDIT END]
        self._update_open_button()

    def _selected_child(self):
        """The selected run item, ``None`` when nothing or a group is selected."""
        items = self.run_tree.selectedItems()
        if not items or items[0].parent() is None:
            return None
        return items[0]

    def _update_open_button(self) -> None:
        """``Curate…`` for an across-day run, ``Inspect…`` for a day's run, disabled otherwise (D20)."""
        child = self._selected_child()
        if child is None:
            self.open_button.setText('Curate…')
            self.open_button.setEnabled(False)
            return
        across = child.parent().text(0) == ACROSS_DAYS
        self.open_button.setText('Curate…' if across else 'Inspect…')
        self.open_button.setEnabled(True)

    def open_run(self) -> None:
        """Open the selected run in its viewer; several may be open, and they outlive the launcher (D1, D27)."""
        child = self._selected_child()
        if child is None:
            return
        entry = child.data(0, _ENTRY_ROLE)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launcher-gui Task 5 fix round 1: a viewer import failure reaches the log pane | date: 2026-10-07]
        try:
            from ..viewer.app import open_run
            viewer = open_run(entry.folder)
        # [AGENT EDIT END]
        except Exception as exc:
            self.append_log(f'error: {exc}')
            return
        viewer.show()
        self.viewers.append(viewer)

    def _choose_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Lab root', str(self.root))
        if path:
            self.set_root(Path(path))

    def set_root(self, path: Path) -> bool:
        """Show the lab root ``path``; ``False``, with the error logged and nothing changed, when its
        plugins or forms cannot be prepared (D9, D29, D72).

        ``load_plugins`` registers the root's source readers process-wide, and they stay registered
        for the process's lifetime, a refused root's included (D44); the window never reads sources
        through them.
        """
        try:
            new_root = cache_root(path)
            api = load_plugins(new_root)
            forms = _forms(api)  # each plugin's configure runs here
        except Exception as exc:
            self.append_log(f'error: {exc}')
            return False
        self.root, self.plugins, self.forms = new_root, api, forms
        self.root_label.set_full_text(str(self.root))
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the tabs are rebuilt from the new root's forms (launcher-gui Task 6) | date: 2026-10-07]
        self._build_tabs()
        # [AGENT EDIT END]
        self._wanted = (None,)
        self.reload()
        return True

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: command tabs build forms off the CLI's parser and run each command as a CLI child, one at a time (launcher-gui Task 6) | date: 2026-10-07]
    # --- tabs -----------------------------------------------------------------------------------
    def _build_tabs(self) -> None:
        """One tab per command: the built-ins, then each plugin command in its order (D10, D45, D46)."""
        self._building = True
        while self.tabs.count():
            page = self.tabs.widget(0)
            self.tabs.removeTab(0)
            page.deleteLater()
        self.run_buttons, self.form_widgets, self._backend_fields = {}, {}, []
        names = [name for name in _BUILTIN_TABS if name in self.forms]
        names += [name for name in self.plugins.commands if name in self.forms]
        for name in names:
            index = self.tabs.addTab(self._build_tab(name), TAB_LABELS.get(name, name))
            self.tabs.setTabToolTip(index, f'launch {name}')
        self._prefilled = _UNSET
        self._building = False

    def _build_tab(self, name: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(page)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room around a tab's form, wider on the scrollbar side (launcher spacing) | date: 2026-10-13]
        column.setContentsMargins(SPACING, SPACING, SPACING + 4, SPACING)
        # [AGENT EDIT END]
        self.form_widgets[name] = {}
        if name == 'run':
            self._specs = {spec.id: spec for spec in registry.list_specs()}
            self.backend_combo = QtWidgets.QComboBox()
            for backend_id in self._specs:
                self.backend_combo.addItem(backend_label(backend_id), backend_id)
            self.backend_combo.currentIndexChanged.connect(lambda _index: self._backend_changed())
            column.addWidget(self.backend_combo)
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the evidence tags sit under the backend (event-optional Task 11) | date: 2026-10-10]
            column.addWidget(self._build_evidence_bar())
            # [AGENT EDIT END]
        form = QtWidgets.QFormLayout()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room between form rows (launcher spacing) | date: 2026-10-13]
        _space_form(form)
        # [AGENT EDIT END]
        column.addLayout(form)
        for field in self.forms[name]:
            self._add_row(form, name, field)
        if name == 'run':  # the backend's option rows sit below the run fields, rebuilt per backend (D73)
            self._option_form = QtWidgets.QFormLayout()
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room between form rows (launcher spacing) | date: 2026-10-13]
            _space_form(self._option_form)
            # [AGENT EDIT END]
            column.addLayout(self._option_form)
            self._fill_backend_rows()
        column.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        button = QtWidgets.QPushButton('Run')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Run button takes the accent fill (launcher-gui Task 9) | date: 2026-10-07]
        button.setProperty('primary', True)
        # [AGENT EDIT END]
        button.clicked.connect(lambda _checked=False, command=name: self.run_command(command))
        self.run_buttons[name] = button
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: room inside the tab's frame and above Run (launcher spacing) | date: 2026-10-13]
        outer.setContentsMargins(SPACING + 4, SPACING + 4, SPACING + 4, SPACING + 4)
        outer.setSpacing(SPACING + 4)
        # [AGENT EDIT END]
        outer.addWidget(scroll, 1)
        outer.addWidget(button)
        return tab

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the evidence bar holds one tag per evidence fallback of the checked days (event-optional Task 11) | date: 2026-10-10]
    def _build_evidence_bar(self) -> QtWidgets.QWidget:
        """``evidence_bar``: the run's evidence tags in a row, hidden while there is none (D20, D25, D55)."""
        self.evidence_bar = QtWidgets.QWidget()
        self._evidence_line = QtWidgets.QHBoxLayout(self.evidence_bar)
        self._evidence_line.setContentsMargins(0, 0, 0, 0)
        self._evidence_line.addStretch(1)
        self._evidence_labels = []
        self.evidence_bar.hide()
        return self.evidence_bar
    # [AGENT EDIT END]

    def _add_row(self, form: QtWidgets.QFormLayout, name: str, field) -> None:
        """One row: ``field.label`` (`` *`` when required, D37) with ``field.help`` as hover (D19)."""
        label = QtWidgets.QLabel(field.label + (' *' if field.required else ''))
        label.setToolTip(field.help)
        widget, row = self._field_widget(name, field)
        widget.setToolTip(field.help)
        form.addRow(label, row)
        self.form_widgets[name][field.dest] = widget

    def _field_widget(self, name: str, field) -> tuple:
        """``(widget, row)``: the widget read for the field's value, and what the row shows."""
        changed = lambda *_args: self._field_changed(name, field.dest)  # noqa: E731
        if field.kind in ('text', 'list'):
            edit = QtWidgets.QLineEdit()
            edit.setPlaceholderText(field.placeholder)
            edit.textChanged.connect(changed)
            if field.kind == 'text' and field.browse is not None:
                row = QtWidgets.QWidget()
                line = QtWidgets.QHBoxLayout(row)
                line.setContentsMargins(0, 0, 0, 0)
                browse = QtWidgets.QToolButton()
                browse.setText('…')
                browse.clicked.connect(lambda _checked=False: self._browse(edit, field.browse))
                line.addWidget(edit, 1)
                line.addWidget(browse)
                return edit, row
            return edit, edit
        if field.kind in ('choice', 'switch'):
            combo = QtWidgets.QComboBox()
            combo.addItems(list(field.choices))
            combo.setCurrentIndex(-1)  # not given
            combo.setPlaceholderText(field.placeholder)
            combo.currentIndexChanged.connect(changed)
            return combo, combo
        if field.kind == 'flag':
            box = QtWidgets.QCheckBox()
            box.toggled.connect(changed)
            return box, box
        label = QtWidgets.QLabel(f'CLI only · {field.flag or field.label}')  # 'unsupported' (D35)
        return label, label

    def _browse(self, edit: QtWidgets.QLineEdit, browse: str) -> None:
        start = edit.text() or str(self.root)
        if browse == 'dir':
            path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Folder', start)
        elif browse == 'save':
            path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save as', start)[0]
        else:
            path = QtWidgets.QFileDialog.getOpenFileName(self, 'File', start)[0]
        if path:
            edit.setText(path)

    def _fill_backend_rows(self) -> None:
        """Rebuild the run tab's option rows for the current backend; the run fields keep their widgets."""
        while self._option_form.rowCount():
            self._option_form.removeRow(0)
        for field in self._backend_fields:
            self.form_widgets['run'].pop(field.dest, None)
        spec = self._specs.get(self.backend_combo.currentData())
        if spec is None:
            self._backend_fields = []
            return
        names = {option.name for option in spec.options}
        parser = commands_of(build_parser(self.plugins.commands, spec))['run']
        self._backend_fields = [f for f in form_fields(parser, BOUND['run'], spec) if f.dest in names]
        for field in self._backend_fields:
            self._add_row(self._option_form, 'run', field)

    def _backend_changed(self) -> None:
        self._fill_backend_rows()
        self._refresh_commands()

    def _field_changed(self, name: str, dest: str) -> None:
        if name == 'run' and dest == 'python':
            self._grey_backends()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a min_repeats change redecides the evidence tags (event-optional Task 11) | date: 2026-10-10]
        if name == 'run' and dest == 'min_repeats':
            self._refresh_evidence()
        # [AGENT EDIT END]
        self._update_run_buttons()

    # --- values ---------------------------------------------------------------------------------
    def _shown(self) -> Optional[str]:
        """The combo's project; ``None`` at index -1."""
        return self.project_combo.currentText() if self.project_combo.currentIndex() >= 0 else None

    def _fields(self, name: str) -> list:
        return list(self.forms[name]) + (list(self._backend_fields) if name == 'run' else [])

    @staticmethod
    def _value(field, widget):
        if field.kind in ('text', 'list'):
            return widget.text().strip()
        if field.kind in ('choice', 'switch'):
            return widget.currentText() if widget.currentIndex() >= 0 else None
        if field.kind == 'flag':
            return widget.isChecked()
        return None

    def _values(self, name: str) -> dict:
        widgets = self.form_widgets.get(name, {})
        return {field.dest: self._value(field, widgets[field.dest]) for field in self._fields(name)
                if field.dest in widgets}

    def _checked_rows(self) -> list:
        """``[(day, tag)]`` of the checked days, in ``day_rows`` order."""
        rows = []
        for i in range(self.days_list.count()):
            item = self.days_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                rows.append((item.data(_ENTRY_ROLE), item.text().rsplit(' · ', 1)[-1]))
        return rows

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every listed day with its tag, for an extract with no day checked (D80) (launcher-gui Task 11) | date: 2026-10-07]
    def _all_rows(self) -> list:
        """``[(day, tag)]`` of every listed day, in ``day_rows`` order."""
        return [(self.days_list.item(i).data(_ENTRY_ROLE), self.days_list.item(i).text().rsplit(' · ', 1)[-1])
                for i in range(self.days_list.count())]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an extract with no day checked takes every listed day not yet extracted, fixed when Run is pressed (D80) (launcher-gui Task 11) | date: 2026-10-07]
    def _command_days(self, name: str) -> tuple:
        """``extract``: the checked days tagged ``none``, and with no day checked every listed day
        tagged ``none``, in ``day_rows`` order, so a waiting extract keeps the days not yet extracted
        when its Run was pressed (D80); ``run``: the checked days; else none (D12)."""
        if name == 'extract':
            rows = self._checked_rows() or self._all_rows()
            return tuple(day for day, tag in rows if tag == 'none')
        if name == 'run':
            return tuple(day for day, _tag in self._checked_rows())
        return ()
    # [AGENT EDIT END]

    def _prefill_projects(self) -> None:
        """The ``project`` field of the init and plugin forms shows the combo's project, else
        ``new_project``, and follows combo changes (D31)."""
        shown = self._shown() or self.new_project
        if shown == self._prefilled:
            return
        self._prefilled = shown
        for name in ('init', *self.plugins.commands):
            widget = self.form_widgets.get(name, {}).get('project')
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(shown or '')

    # --- Run buttons and greyed backends ----------------------------------------------------------
    def _refresh_commands(self) -> None:
        self._grey_backends()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every refresh redecides the evidence tags and the greyed event choices (event-optional Task 11) | date: 2026-10-10]
        self._refresh_evidence()
        # [AGENT EDIT END]
        self._update_run_buttons()

    def _run_interpreter(self) -> Optional[str]:
        """The run form's ``python``, else the verified interpreter, looked up once per window."""
        widget = self.form_widgets.get('run', {}).get('python')
        if isinstance(widget, QtWidgets.QLineEdit) and widget.text().strip():
            return widget.text().strip()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a raising lookup is kept and re-raised, so the lookup runs at most once per window (launcher-gui Task 6 fix round 1) | date: 2026-10-07]
        if self._interpreter is _UNSET:
            try:
                self._interpreter = (self.interpreter_lookup or dependency_setup.verified_interpreter)()
            except Exception as exc:
                self._interpreter = exc
        if isinstance(self._interpreter, Exception):
            raise self._interpreter
        # [AGENT EDIT END]
        return self._interpreter

    def _grey_backends(self) -> None:
        """Disable every backend item the checked days cannot run, its reason as hover (D5, D38)."""
        if self._building or not hasattr(self, 'backend_combo'):
            return
        days = self._command_days('run')
        model = self.backend_combo.model()
        for i in range(self.backend_combo.count()):
            reason = ''
            if days:
                try:
                    reason = form_module.backend_refusal(self._project, days, self.backend_combo.itemData(i),
                                                         interpreter=self._run_interpreter())
                except Exception as exc:
                    reason = f'error: {exc}'
            item = model.item(i)
            item.setEnabled(reason == '')
            item.setToolTip(reason)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run form's evidence: decided as launch run decides it, shown as tags, and greying the choices only events measure (event-optional Task 11) | date: 2026-10-10]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the field's min_repeats is checked against >= 1 here, so a bad value shows its error tag with or without checked days (event-optional Task 11, fix round 1) | date: 2026-10-10]
    def _min_repeats(self) -> int:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the field's min_repeats must be an int >= 2, as run_evidence refuses (event-optional Task 14) | date: 2026-10-10]
        """The run form's ``min_repeats`` as ``launch run`` reads it: the field's integer, else the backend's
        default, else ``Selection.evidence``'s; ``ValueError`` for field text that is not an integer >= 2."""
        widget = self.form_widgets.get('run', {}).get('min_repeats')
        text = widget.text().strip() if isinstance(widget, QtWidgets.QLineEdit) else ''
        if text:
            try:
                value = int(text)  # as the CLI's type=int parses it
            except ValueError:
                raise ValueError(f'min_repeats must be an int >= 2, not {text!r}') from None
            if value < 2:
                raise ValueError(f'min_repeats must be an int >= 2, not {value!r}')
            return value
        # [AGENT EDIT END]
        spec = self._specs.get(self.backend_combo.currentData())
        option = None if spec is None else next((o for o in spec.options if o.name == 'min_repeats'), None)
        return _EVIDENCE_MIN_REPEATS if option is None else option.default
    # [AGENT EDIT END]

    def _no_events(self) -> bool:
        """True when the checked days' evidence is decided and has no events."""
        return self._evidence is not None and not self._evidence.events

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: min_repeats is validated before the selection guard; only deciding the evidence needs checked days (event-optional Task 11, fix round 1) | date: 2026-10-10]
    def _refresh_evidence(self) -> None:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring says >= 2 (event-optional Task 14) | date: 2026-10-10]
        """Redecide the checked days' evidence for the current backend and rebuild its tags, then the enabled
        state and hover of every ``EVENT_CHOICES`` item (D18, D20, D25). A ``min_repeats`` that is not an
        integer >= 2, checked or not, or any error deciding the evidence, shows one ``error`` tag instead (D55)."""
        # [AGENT EDIT END]
        if self._building or not hasattr(self, 'evidence_bar'):
            return
        self._evidence, self._evidence_error, tags = None, None, []
        try:
            min_repeats = self._min_repeats()
        except ValueError as exc:
            self._evidence_error = str(exc)
        else:
            days, backend_id = self._command_days('run'), self.backend_combo.currentData()
            if days and self._project is not None and backend_id is not None:
                try:
                    self._evidence = form_module.selection_evidence(self._project, days, backend_id,
                                                                    min_repeats=min_repeats)
                    tags = [] if self._evidence is None else evidence_tags(self._evidence.reasons)
                except Exception as exc:
                    self._evidence, self._evidence_error = None, str(exc)
        if self._evidence_error is not None:
            tags = [('error', self._evidence_error)]
        self._show_evidence_tags(tags)
        self._grey_event_choices()
    # [AGENT EDIT END]

    def _show_evidence_tags(self, tags: list) -> None:
        """Replace the evidence bar's tags with one amber ``QLabel`` per ``(text, tooltip)``."""
        for label in self._evidence_labels:
            self._evidence_line.removeWidget(label)
            label.hide()
            label.deleteLater()
        self._evidence_labels = []
        for text, tooltip in tags:
            label = QtWidgets.QLabel(text)
            label.setToolTip(tooltip)
            label.setProperty('tone', 'unsure')
            label.setStyleSheet('font-size:11px;')
            theme.repolish(label)
            self._evidence_line.insertWidget(len(self._evidence_labels), label)
            self._evidence_labels.append(label)
        self.evidence_bar.setVisible(bool(tags))

    def _grey_event_choices(self) -> None:
        """Without events, disable each ``EVENT_CHOICES`` item of the run form, hover ``no events``; else
        enable it with no hover (D18)."""
        no_events = self._no_events()
        for dest, choices in EVENT_CHOICES.items():
            combo = self.form_widgets.get('run', {}).get(dest)
            if not isinstance(combo, QtWidgets.QComboBox):
                continue
            model = combo.model()
            for i in range(combo.count()):
                greyed = no_events and combo.itemText(i) in choices
                model.item(i).setEnabled(not greyed)
                model.item(i).setToolTip('no events' if greyed else '')

    def evidence_tags_shown(self) -> list:
        """``[(text, tooltip)]`` of the run form's evidence tags, in order."""
        return [(label.text(), label.toolTip()) for label in self._evidence_labels]
    # [AGENT EDIT END]

    def _ready(self, name: str) -> bool:
        """Whether ``name``'s Run is enabled; a running or waiting command changes nothing (D8, D77)."""
        fields, values = self._fields(name), self._values(name)
        if any(f.kind == 'unsupported' for f in fields):
            return False
        if any(f.required and not has_value(f, values.get(f.dest)) for f in fields):
            return False
        if name in _NEEDS_PROJECT and not self.project_ok:
            return False
        checked = self._checked_rows()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: extract is ready when it has days to send (D80) (launcher-gui Task 11) | date: 2026-10-07]
        if name == 'extract':
            return bool(self._command_days('extract'))
        # [AGENT EDIT END]
        if name == 'run':
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an error tag, or a choice only events measure on days without events, keeps Run disabled (event-optional Task 11, D55) | date: 2026-10-10]
            if self._evidence_error is not None:
                return False
            if self._no_events() and any(values.get(dest) in choices for dest, choices in EVENT_CHOICES.items()):
                return False
            # [AGENT EDIT END]
            index = self.backend_combo.currentIndex()
            greyed = index < 0 or not self.backend_combo.model().item(index).isEnabled()
            return bool(checked) and all(tag == 'raw' for _day, tag in checked) and not greyed
        return True

    def _update_run_buttons(self) -> None:
        if self._building:
            return
        for name, button in self.run_buttons.items():
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: after the close bar's Yes every Run button stays disabled (D79) (launcher-gui Task 7) | date: 2026-10-07]
            button.setEnabled(not self._close_after and self._ready(name))
            # [AGENT EDIT END]

    # --- running --------------------------------------------------------------------------------
    def running(self) -> bool:
        """True from a job's start until its ``ended``."""
        return self._busy

    def _selected_folder(self) -> Optional[Path]:
        child = self._selected_child()
        return None if child is None else child.data(0, _ENTRY_ROLE).folder

    def run_command(self, command: str) -> None:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names D79 (launcher-gui Task 7 fix round 1) | date: 2026-10-07]
        """Run ``command`` from its form as a CLI child; while one runs, it waits in ``queue`` (D8, D40).
        After the close bar's ``Yes`` nothing is queued or started (D79)."""
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a window closing on the running job's end takes no new job (launcher-gui Task 7) | date: 2026-10-07]
        if self._close_after:
            return
        # [AGENT EDIT END]
        values = self._values(command)
        shown = self._shown()
        try:
            argv = command_argv(command, self._fields(command), values, project=shown,
                                days=self._command_days(command),
                                backend=self.backend_combo.currentData() if command == 'run' else None)
        except ValueError as exc:
            self.append_log(f'error: {exc}')
            return
        program, arguments = child_command(self.root, argv)  # the root is frozen now
        project = values.get('project') if 'project' in values else shown
        if project not in (None, ''):
            try:
                project = Project._name_under(self.root, project)
            except ValueError:  # kept as given
                pass
        job = {'command': command, 'argv': argv, 'program': program, 'arguments': arguments,
               'project': project, 'shown': shown, 'selected': self._selected_folder()}
        if self._busy:
            self.queue.append(job)
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the queue list follows the queue (launcher-gui Task 7) | date: 2026-10-07]
            self._queue_changed()
            # [AGENT EDIT END]
        else:
            self._start(job)

    def _start(self, job: dict) -> None:
        self.submitted = job
        self.last_stdout = ''
        self.append_log('$ launch ' + shlex.join(job['argv']))
        process = (self.process_factory or CommandProcess)(self)
        self.process = process
        process.stdout_line.connect(self._on_stdout)
        process.stderr_line.connect(self.append_log)
        process.ended.connect(lambda code, ended_by=process: self._on_ended(ended_by, code))
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Cancel waits for the process's started (launcher-gui Task 7) | date: 2026-10-07]
        process.started.connect(lambda started_by=process: self._on_started(started_by))
        self._started = self._cancel_requested = self._cancel_pending = self._killed = False
        self.cancel_button.setEnabled(False)
        # [AGENT EDIT END]
        self._busy = True
        self.root_action.setEnabled(False)  # D32: only the root is locked
        process.start(job['program'], job['arguments'], _child_env(), str(REPO))

    def _on_stdout(self, line: str) -> None:
        self.last_stdout = line
        self.append_log(line)

    def _find_run(self, folder: Path):
        """The run item whose entry's folder resolves to ``folder``; ``None`` when not listed."""
        target = Path(folder).resolve()
        for i in range(self.run_tree.topLevelItemCount()):
            top = self.run_tree.topLevelItem(i)
            for j in range(top.childCount()):
                child = top.child(j)
                if child.data(0, _ENTRY_ROLE).folder.resolve() == target:
                    return child
        return None

    def _reselect(self, folder: Optional[Path]) -> None:
        item = None if folder is None else self._find_run(folder)
        if item is not None:
            self.run_tree.setCurrentItem(item)

    def _on_ended(self, process, code: int) -> None:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the killed line and the close after Yes (launcher-gui Task 7 fix round 1) | date: 2026-10-07]
        """Log the end (``killed`` or ``exit <code>``), reload, follow the job when the shown project has
        not moved, then start the next waiting job whatever ``code`` was, or close the window after the
        close bar's ``Yes`` (D8, D11, D30, D71)."""
        # [AGENT EDIT END]
        if process is not self.process:
            return
        job = self.submitted
        self._busy = False
        shown_now = self._shown()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the end line names a kill; the cancel marks reset and the close bar hides (launcher-gui Task 7) | date: 2026-10-07]
        self._kill_timer.stop()
        self.append_log('killed' if self._killed else f'exit {code}')
        self._started = self._cancel_requested = self._cancel_pending = self._killed = False
        self.cancel_button.setEnabled(False)
        self.close_bar.hide()
        # [AGENT EDIT END]
        self.reload()
        if shown_now == job['shown']:
            self._follow(job, code)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Yes on the close bar closes the window instead of starting a next job (launcher-gui Task 7) | date: 2026-10-07]
        if self._close_after:
            self.root_action.setEnabled(True)
            self.close()
            return
        # [AGENT EDIT END]
        if self.queue:
            self._start(self.queue.pop(0))
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the queue list follows the queue (launcher-gui Task 7) | date: 2026-10-07]
            self._queue_changed()
            # [AGENT EDIT END]
        else:
            self.root_action.setEnabled(True)

    def _follow(self, job: dict, code: int) -> None:
        command = job['command']
        if command == 'init' or command in self.plugins.commands:
            index = self.project_combo.findText(job['project']) if isinstance(job['project'], str) else -1
            if index >= 0 and index != self.project_combo.currentIndex():
                self.project_combo.setCurrentIndex(index)
                return
            self._reselect(job['selected'])
        elif command == 'run' and code == 0:
            text = self.last_stdout.rsplit(' · ', 3)[0] if self.last_stdout else ''
            item = self._find_run(Path(text)) if text else None
            if item is None:
                self.append_log(f'error: run folder not listed: {text}')
                self._reselect(job['selected'])
            else:
                self.run_tree.setCurrentItem(item)
        else:
            self._reselect(job['selected'])
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Cancel interrupts the running job, then kills it after KILL_AFTER_MS; the queue list; the close confirmation (launcher-gui Task 7) | date: 2026-10-07]
    # --- cancel ---------------------------------------------------------------------------------
    def _on_started(self, process) -> None:
        """Enable Cancel, or send a cancel asked before ``started`` (D70)."""
        if process is not self.process or not self._busy:
            return
        self._started = True
        if self._cancel_pending:
            self._send_cancel()
        else:
            self.cancel_button.setEnabled(not self._cancel_requested)

    def _cancel(self) -> None:
        """The Cancel path: acts on the running job only, once; the waiting jobs stay (D7, D8)."""
        if not self._busy or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        if self._started:
            self._send_cancel()
        else:  # nothing to signal before started (D70)
            self._cancel_pending = True

    def _send_cancel(self) -> None:
        """SIGINT now; a kill when the process still runs after ``KILL_AFTER_MS``."""
        self._cancel_pending = False
        self.process.interrupt()
        self._kill_timer.start(KILL_AFTER_MS)

    def _kill_due(self) -> None:
        if self._busy and self.process is not None and self.process.running():
            self.process.kill()
            self._killed = True

    # --- the queue list -------------------------------------------------------------------------
    def _queue_changed(self) -> None:
        """Rebuild ``queue_list`` from ``queue``; ``queue_box`` shows only while a job waits."""
        self.queue_list.clear()
        for index, job in enumerate(self.queue):
            item = QtWidgets.QListWidgetItem('launch ' + shlex.join(job['argv']))
            self.queue_list.addItem(item)
            row = QtWidgets.QWidget()
            line = QtWidgets.QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addStretch(1)
            remove = QtWidgets.QToolButton()
            remove.setText('×')
            remove.setToolTip('remove')
            remove.clicked.connect(lambda _checked=False, i=index: self.remove_queued(i))
            line.addWidget(remove)
            self.queue_list.setItemWidget(item, row)
        self.queue_box.setHidden(not self.queue)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the visible close bar's waiting count follows the queue (launcher-gui Task 7 fix round 1) | date: 2026-10-07]
        if self._busy and not self.close_bar.isHidden():
            self.close_label.setText(self._close_text())
        # [AGENT EDIT END]

    def remove_queued(self, index: int) -> None:
        """Drop the waiting job at ``index``; the running job and the log are untouched."""
        del self.queue[index]
        self._queue_changed()

    def clear_queue(self) -> None:
        """Drop every waiting job; the running job and the log are untouched."""
        self.queue.clear()
        self._queue_changed()

    # --- close ----------------------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Close at once when idle, open viewers staying open (D27); while a command runs, ask on
        ``close_bar`` instead (D18, D78)."""
        if not self._busy:
            super().closeEvent(event)
            return
        event.ignore()
        if self._close_after:  # already closing on the running job's end
            return
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the close bar's text comes from one helper the queue list also refreshes (launcher-gui Task 7 fix round 1) | date: 2026-10-07]
        self.close_label.setText(self._close_text())
        # [AGENT EDIT END]
        self.close_bar.show()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the close bar's text comes from one helper the queue list also refreshes (launcher-gui Task 7 fix round 1) | date: 2026-10-07]
    def _close_text(self) -> str:
        """``<command> running · cancel and close?``, with ``· N waiting`` while ``N`` jobs wait (D78)."""
        waiting = f' · {len(self.queue)} waiting' if self.queue else ''
        return f"{self.submitted['command']} running{waiting} · cancel and close?"
    # [AGENT EDIT END]

    def _close_confirmed(self) -> None:
        """``Yes``: drop the queue, cancel the running job and close when it ends (D70, D78, D79)."""
        self.close_bar.hide()
        self.clear_queue()
        self._cancel()
        self._close_after = True
        self._update_run_buttons()
    # [AGENT EDIT END]


def run_window(root: Path, project: Optional[str], plugins: PluginApi) -> int:
    """Show the launcher window over ``root`` and run the Qt event loop; return its exit code."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the saved theme is applied before the window is built (D66) (launcher-gui Task 9) | date: 2026-10-07]
    theme.apply_theme(app, theme.saved_theme())
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher opens in the saved unit palette (graphite-palettes Task 4) | date: 2026-10-11]
    unit_palettes.set_palette(theme.saved_unit_palette())
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every window of the launcher carries the logo icon (logo) | date: 2026-10-13]
    app.setWindowIcon(logo.icon())
    # [AGENT EDIT END]
    window = LauncherWindow(root, project, plugins=plugins)
    window.show()
    return app.exec()
# [AGENT EDIT END]
