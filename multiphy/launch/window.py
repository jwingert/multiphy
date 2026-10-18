"""The launcher window: one window over a lab root's projects, days and completed runs.

``list_projects``, ``day_rows`` and ``run_groups`` read what the window lists. ``LauncherWindow``
shows the root, a project picker, the project's days (checkable) and its runs grouped as
``across days`` then one group per day, and opens the selected run with ``Curate…`` or
``Inspect…``. The selected run is read on one worker thread while the open button shows a spinner
and stays locked. The window writes no project data itself. ``run_window`` runs it as an
application. Qt comes only from ``..viewer.qt``; no lab module is imported.
"""
from __future__ import annotations

import codecs
import concurrent.futures
import os
import shlex
import signal
import re
import inspect
from pathlib import Path
import sys
from typing import Optional

from .. import dependency_setup
from ..contract import registry
from ..contract.names import backend_label
from ..project.folder import MANIFEST_FILE, RUNS_DIR, Project
from ..run import catalog
from ..run.config import cache_root
from ..viewer.qt import QtCore, QtGui, QtWidgets
from ..viewer import display, menus, theme
from ..viewer import unit_palettes
from ..viewer import logo
from .__main__ import build_parser
from . import forms as form_module  # backend_refusal is read through the module, so tests can replace it
from .commands import _pythonpath
from .forms import BOUND, command_argv, commands_of, form_fields, has_value
from .plugins import PluginApi, load_plugins
from ..contract.inputs import evidence_tags
from ..project.selection import Selection
from .forms import EVENT_CHOICES

# launch run's min_repeats for a backend that reads timing without declaring the option
_EVIDENCE_MIN_REPEATS = inspect.signature(Selection.evidence).parameters['min_repeats'].default

ACROSS_DAYS = 'across days'
TITLE = 'Cross-day unit match launcher'
_ENTRY_ROLE = QtCore.Qt.UserRole
TAB_LABELS = {'init': 'Initialization', 'extract': 'Extract features', 'run': 'Run matching',
              'export': 'Export tracks', 'status': 'Status'}
REPO = Path(__file__).resolve().parents[2]
_BUILTIN_TABS = ('init', 'extract', 'run', 'export', 'status')
_NEEDS_PROJECT = ('extract', 'run', 'status', 'export')  # disabled while project_ok is False (D38)
_UNSET = object()
KILL_AFTER_MS = 10000  # read each time the kill timer starts
OPEN_SIZE = (1280, 860)  # the size every open starts at, before the screen cap; nothing is remembered
SCREEN_SHARE = 0.9       # at most this share of the screen's available width and height
EDGE = 18                # px between the window's edge and its panes
GAP = 16                 # px between panes: the splitter handles
SPACING = 8              # px between the widgets of a pane, and around a tab's form
FORM_ROWS = 12           # px between form rows; a row's label sits FORM_ROWS + 4 px from its field
SPIN_FRAMES = '◐◓◑◒'     # the open button's spinner while a run opens (D10)
SPIN_MS = 150            # ms per spinner frame
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


class _ElidedButton(QtWidgets.QPushButton):
    """A push button that shows its full text elided in the middle to its width, the full text as hover;
    it never widens its column, and keeps one height whether it shows text or a spinner frame (D9)."""

    def __init__(self, text: str = '', parent=None):
        super().__init__(parent)
        self._full = ''
        self.setMinimumWidth(1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def _elide(self) -> None:
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        width = self.style().subElementRect(QtWidgets.QStyle.SE_PushButtonContents, option, self).width()
        self.setText(QtGui.QFontMetrics(self.font()).elidedText(self._full, QtCore.Qt.ElideMiddle, max(width, 1)))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def sizeHint(self) -> QtCore.QSize:
        """The push button's hint, its height raised to that of a one-glyph text, so a spinner frame
        never changes the button's height."""
        hint = super().sizeHint()
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        glyph = self.fontMetrics().size(QtCore.Qt.TextShowMnemonic, SPIN_FRAMES[0])
        spun = self.style().sizeFromContents(QtWidgets.QStyle.CT_PushButton, option, glyph, self)
        return QtCore.QSize(hint.width(), max(hint.height(), spun.height()))


def _space_form(form: QtWidgets.QFormLayout) -> None:
    """``FORM_ROWS`` px between ``form``'s rows; ``FORM_ROWS + 4`` px between a label and its field."""
    form.setVerticalSpacing(FORM_ROWS)
    form.setHorizontalSpacing(FORM_ROWS + 4)


class LauncherWindow(QtWidgets.QMainWindow):
    """The launcher window over ``root``; ``project`` is shown first (D31, D74).

    ``plugins`` gives the plugin commands; ``process_factory`` and ``interpreter_lookup`` are kept
    for the command tabs. ``forms`` holds each command's form fields, read off the CLI's parser.
    """

    _read_done = QtCore.Signal(object)  # (PreparedRun, None) or (None, exception), from the read's worker

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
        self.log_lines = []
        self.queue = []
        self.submitted = None
        self.process = None
        self.last_stdout = ''
        self.run_buttons = {}
        self.form_widgets = {}
        self._busy = False
        self._project = None  # the opened Project of the last successful fill
        self._interpreter = _UNSET  # interpreter_lookup's answer or its exception, looked up at most once per root
        self._prefilled = _UNSET  # the project the init and plugin forms were last filled with
        self._backend_fields = []
        self._specs = {}
        self._building = True
        self._started = False  # the running job's process emitted started
        self._cancel_requested = False  # the Cancel path ran for the running job
        self._cancel_pending = False  # cancel asked before started; sent on started (D70)
        self._killed = False  # the kill timer killed the running job
        self._close_after = False  # the close bar's Yes was clicked: close on ended (D79)
        self._close_on_open = False  # a close came while a run opened: close once its viewer shows (D5)
        self._kill_timer = QtCore.QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_due)
        self._evidence = None  # the RunEvidence of the checked days, None without one
        self._evidence_error = None  # the message of a bad min_repeats or an evidence error (D55)
        self._evidence_labels = []
        name = project
        if project is not None:
            try:
                name = Project._name_under(self.root, project)
            except ValueError:  # not a folder directly under the root: kept as given
                name = project
        self._wanted = (name,)  # the next reload selects this name, or the first project for None
        self._opening = None  # the Future of the read in progress
        self._opening_label = ''  # the run list label of the run being opened
        self._reader = None  # the one-worker ThreadPoolExecutor, made on the first open
        self._spin_frame = 0
        self._spin_timer = QtCore.QTimer(self)
        self._spin_timer.setInterval(SPIN_MS)
        self._spin_timer.timeout.connect(self._spin)
        self._read_done.connect(self._open_ready, QtCore.Qt.QueuedConnection)
        self._build()
        self._open_size()
        menus.install_menus(self, (self.root_action,))
        theme.notifier().changed.connect(self._theme_changed)
        self.reload()
        if name is not None and self.project_combo.currentIndex() < 0:  # a name not under the root
            self.new_project = name
        self._prefill_projects()
        self._refresh_commands()

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

    def _build(self) -> None:
        self.root_label = _ElidedLabel()
        self.root_label.set_full_text(str(self.root))
        self.root_action = QtGui.QAction('Root…', self)
        self.root_action.triggered.connect(self._choose_root)
        self.addAction(self.root_action)
        self.project_combo = QtWidgets.QComboBox()
        self.project_combo.currentIndexChanged.connect(lambda _index: self.reload())
        self.days_list = QtWidgets.QListWidget()
        self.days_list.itemChanged.connect(lambda _item: self._refresh_commands())
        self.run_tree = QtWidgets.QTreeWidget()
        self.run_tree.setHeaderHidden(True)
        self.run_tree.itemSelectionChanged.connect(self._update_open_button)
        self.open_button = _ElidedButton('Curate…')
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_run)

        left = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING)
        column.addWidget(self.root_label)
        column.addWidget(self.project_combo)
        column.addWidget(QtWidgets.QLabel('Days'))
        column.addWidget(self.days_list, 1)
        column.addWidget(QtWidgets.QLabel('Runs'))
        column.addWidget(self.run_tree, 2)
        column.addWidget(self.open_button)
        self.tabs = QtWidgets.QTabWidget()
        self._build_tabs()
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top.addWidget(left)
        top.addWidget(self.tabs)
        top.setStretchFactor(1, 1)
        top.setHandleWidth(GAP)
        self.log_pane = QtWidgets.QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        self.log_pane.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        bottom = QtWidgets.QWidget()
        stack = QtWidgets.QVBoxLayout(bottom)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(SPACING)
        stack.addWidget(self._build_close_bar())
        pane = QtWidgets.QHBoxLayout()
        pane.setSpacing(SPACING)
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
        body.setStretchFactor(0, 3)
        body.setContentsMargins(EDGE, EDGE, EDGE, EDGE)
        body.setHandleWidth(GAP)
        self.setCentralWidget(body)

    def _build_close_bar(self) -> QtWidgets.QFrame:
        """``close_bar``: the close confirmation, hidden until a close arrives while a command runs (D18)."""
        self.close_bar = QtWidgets.QFrame()
        self.close_bar.setObjectName('close_bar')
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
        self._project = project
        self._update_open_button()

    def _selected_child(self):
        """The selected run item, ``None`` when nothing or a group is selected."""
        items = self.run_tree.selectedItems()
        if not items or items[0].parent() is None:
            return None
        return items[0]

    def _update_open_button(self) -> None:
        """While a run opens, ``<frame> opening · <run>``, disabled, whatever is selected; otherwise
        ``Curate…`` for an across-day run, ``Inspect…`` for a day's run, disabled without a run (D20)."""
        if self._opening is not None:
            self.open_button.set_full_text(f'{SPIN_FRAMES[self._spin_frame]} opening · {self._opening_label}')
            self.open_button.setEnabled(False)
            return
        child = self._selected_child()
        if child is None:
            self.open_button.set_full_text('Curate…')
            self.open_button.setEnabled(False)
            return
        across = child.parent().text(0) == ACROSS_DAYS
        self.open_button.set_full_text('Curate…' if across else 'Inspect…')
        self.open_button.setEnabled(True)

    def opening(self) -> bool:
        """True from an open's start until its viewer shows or the open fails."""
        return self._opening is not None

    def open_run(self) -> None:
        """Open the selected run in its viewer: the run is read on the worker thread, and the button shows
        the spinner and stays locked until the viewer shows or the open fails; a click or call meanwhile does
        nothing (D2, D3, D4). Several viewers may be open, and they outlive the launcher (D1, D27)."""
        if self._opening is not None:
            return
        child = self._selected_child()
        if child is None:
            return
        entry = child.data(0, _ENTRY_ROLE)
        try:
            from ..viewer.app import prepare_run
            if self._reader is None:
                self._reader = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='multiphy-open')
            future = self._reader.submit(self._read, prepare_run, entry.folder)
        except Exception as exc:
            self.append_log(f'error: {exc}')
            if self._reader is not None:
                self._reader.shutdown(wait=False)
            self._reader = None
            return
        self._opening = future
        self._opening_label = child.text(0)
        self._spin_frame = 0
        self._spin_timer.start()
        self._update_open_button()  # the queued result cannot arrive before this returns

    def _read(self, prepare, folder) -> None:
        """The worker thread's body: read ``folder`` with ``prepare`` and emit the outcome; no Qt object
        is touched but the emit."""
        try:
            payload = (prepare(folder), None)
        except BaseException as exc:  # every failure reaches the log pane and unlocks the button (D3)
            payload = (None, exc)
        self._read_done.emit(payload)

    def _spin(self) -> None:
        self._spin_frame = (self._spin_frame + 1) % len(SPIN_FRAMES)
        self._update_open_button()

    def _open_ready(self, payload) -> None:
        """On the GUI thread: build and show the read run's viewer, or log the failure; then unlock the
        button, after the clicks queued while the window was built reached it still locked (D3, D11)."""
        prepared, error = payload
        viewer = None
        opened = False
        try:
            if error is None:
                from ..viewer.app import open_prepared  # read at call time, so tests may replace it
                viewer = open_prepared(prepared)
                viewer.show()
                self.viewers.append(viewer)
                opened = True
        except Exception as exc:
            error = exc
        finally:
            if viewer is not None and not opened:
                try:
                    viewer.close()  # a viewer built but not shown gives its run's lock back
                except Exception as close_error:
                    self.append_log(f'error: {close_error}')
            QtWidgets.QApplication.processEvents()  # clicks queued while the window was built reach the still-locked button and are dropped (D11)
            self._opening = None
            self._spin_timer.stop()
            self._update_open_button()
            if not opened and self._close_on_open:  # the window stays, and runs commands again (D5)
                self._close_on_open = False
                if not self._busy:  # a Yes still waiting on its command's end keeps it
                    self._close_after = False
                self._update_run_buttons()
        if error is not None:
            self.append_log(f'error: {error}')
        if opened and self._close_on_open:
            self._close_on_open = False
            self.close()

    def _choose_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Root', str(self.root))
        if path:
            self.set_root(Path(path))

    def set_root(self, path: Path) -> bool:
        """Show the root ``path``; ``False``, with the error logged and nothing changed, when its
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
        self._interpreter = _UNSET  # the verified interpreter is looked up under the new root
        self.root_label.set_full_text(str(self.root))
        self._build_tabs()
        self._wanted = (None,)
        self.reload()
        return True

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
        column.setContentsMargins(SPACING, SPACING, SPACING + 4, SPACING)
        self.form_widgets[name] = {}
        if name == 'run':
            self._specs = {spec.id: spec for spec in registry.list_specs()}
            self.backend_combo = QtWidgets.QComboBox()
            for backend_id in self._specs:
                self.backend_combo.addItem(backend_label(backend_id), backend_id)
            self.backend_combo.currentIndexChanged.connect(lambda _index: self._backend_changed())
            column.addWidget(self.backend_combo)
            column.addWidget(self._build_evidence_bar())
        form = QtWidgets.QFormLayout()
        _space_form(form)
        column.addLayout(form)
        for field in self.forms[name]:
            self._add_row(form, name, field)
        if name == 'run':  # the backend's option rows sit below the run fields, rebuilt per backend (D73)
            self._option_form = QtWidgets.QFormLayout()
            _space_form(self._option_form)
            column.addLayout(self._option_form)
            self._fill_backend_rows()
        column.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        button = QtWidgets.QPushButton('Run')
        button.setProperty('primary', True)
        button.clicked.connect(lambda _checked=False, command=name: self.run_command(command))
        self.run_buttons[name] = button
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(tab)
        outer.setContentsMargins(SPACING + 4, SPACING + 4, SPACING + 4, SPACING + 4)
        outer.setSpacing(SPACING + 4)
        outer.addWidget(scroll, 1)
        outer.addWidget(button)
        return tab

    def _build_evidence_bar(self) -> QtWidgets.QWidget:
        """``evidence_bar``: the run's evidence tags in a row, hidden while there is none (D20, D25, D55)."""
        self.evidence_bar = QtWidgets.QWidget()
        self._evidence_line = QtWidgets.QHBoxLayout(self.evidence_bar)
        self._evidence_line.setContentsMargins(0, 0, 0, 0)
        self._evidence_line.addStretch(1)
        self._evidence_labels = []
        self.evidence_bar.hide()
        return self.evidence_bar

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
        if name == 'run' and dest == 'min_repeats':
            self._refresh_evidence()
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

    def _all_rows(self) -> list:
        """``[(day, tag)]`` of every listed day, in ``day_rows`` order."""
        return [(self.days_list.item(i).data(_ENTRY_ROLE), self.days_list.item(i).text().rsplit(' · ', 1)[-1])
                for i in range(self.days_list.count())]

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
        self._refresh_evidence()
        self._update_run_buttons()

    def _run_interpreter(self) -> Optional[str]:
        """The run form's ``python``, else the verified interpreter, looked up once per root."""
        widget = self.form_widgets.get('run', {}).get('python')
        if isinstance(widget, QtWidgets.QLineEdit) and widget.text().strip():
            return widget.text().strip()
        if self._interpreter is _UNSET:
            try:
                self._interpreter = (self.interpreter_lookup() if self.interpreter_lookup
                                     else dependency_setup.verified_interpreter(root=self.root))
            except Exception as exc:
                self._interpreter = exc
        if isinstance(self._interpreter, Exception):
            raise self._interpreter
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

    def _min_repeats(self) -> int:
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
        spec = self._specs.get(self.backend_combo.currentData())
        option = None if spec is None else next((o for o in spec.options if o.name == 'min_repeats'), None)
        return _EVIDENCE_MIN_REPEATS if option is None else option.default

    def _no_events(self) -> bool:
        """True when the checked days' evidence is decided and has no events."""
        return self._evidence is not None and not self._evidence.events

    def _refresh_evidence(self) -> None:
        """Redecide the checked days' evidence for the current backend and rebuild its tags, then the enabled
        state and hover of every ``EVENT_CHOICES`` item (D18, D20, D25). A ``min_repeats`` that is not an
        integer >= 2, checked or not, or any error deciding the evidence, shows one ``error`` tag instead (D55)."""
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
        if name == 'extract':
            return bool(self._command_days('extract'))
        if name == 'run':
            if self._evidence_error is not None:
                return False
            if self._no_events() and any(values.get(dest) in choices for dest, choices in EVENT_CHOICES.items()):
                return False
            index = self.backend_combo.currentIndex()
            greyed = index < 0 or not self.backend_combo.model().item(index).isEnabled()
            return bool(checked) and all(tag == 'raw' for _day, tag in checked) and not greyed
        return True

    def _update_run_buttons(self) -> None:
        if self._building:
            return
        for name, button in self.run_buttons.items():
            button.setEnabled(not self._close_after and self._ready(name))

    # --- running --------------------------------------------------------------------------------
    def running(self) -> bool:
        """True from a job's start until its ``ended``."""
        return self._busy

    def _selected_folder(self) -> Optional[Path]:
        child = self._selected_child()
        return None if child is None else child.data(0, _ENTRY_ROLE).folder

    def run_command(self, command: str) -> None:
        """Run ``command`` from its form as a CLI child; while one runs, it waits in ``queue`` (D8, D40).
        After the close bar's ``Yes`` nothing is queued or started (D79)."""
        if self._close_after:
            return
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
            self._queue_changed()
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
        process.started.connect(lambda started_by=process: self._on_started(started_by))
        self._started = self._cancel_requested = self._cancel_pending = self._killed = False
        self.cancel_button.setEnabled(False)
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
        """Log the end (``killed`` or ``exit <code>``), reload, follow the job when the shown project has
        not moved, then start the next waiting job whatever ``code`` was, or close the window after the
        close bar's ``Yes`` (D8, D11, D30, D71)."""
        if process is not self.process:
            return
        job = self.submitted
        self._busy = False
        shown_now = self._shown()
        self._kill_timer.stop()
        self.append_log('killed' if self._killed else f'exit {code}')
        self._started = self._cancel_requested = self._cancel_pending = self._killed = False
        self.cancel_button.setEnabled(False)
        self.close_bar.hide()
        self.reload()
        if shown_now == job['shown']:
            self._follow(job, code)
        if self._close_after:
            self.root_action.setEnabled(True)
            self.close()
            return
        if self.queue:
            self._start(self.queue.pop(0))
            self._queue_changed()
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
        if self._busy and not self.close_bar.isHidden():
            self.close_label.setText(self._close_text())

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
        ``close_bar`` instead (D18, D78). While a run opens and no command runs, the close is deferred:
        ``closing · once <run> opens`` is logged once and the window closes once the viewer shows; a
        failed open keeps it open (D5, D8)."""
        if not self._busy:
            if self._opening is not None:
                event.ignore()
                if not self._close_on_open:
                    self._close_on_open = True
                    self.append_log(f'closing · once {self._opening_label} opens')
                return
            if self._reader is not None:  # no read runs now
                self._reader.shutdown(wait=False)
                self._reader = None
            super().closeEvent(event)
            return
        event.ignore()
        if self._close_after:  # already closing on the running job's end
            return
        self.close_label.setText(self._close_text())
        self.close_bar.show()

    def _close_text(self) -> str:
        """``<command> running · cancel and close?``, with ``· N waiting`` while ``N`` jobs wait (D78)."""
        waiting = f' · {len(self.queue)} waiting' if self.queue else ''
        return f"{self.submitted['command']} running{waiting} · cancel and close?"

    def _close_confirmed(self) -> None:
        """``Yes``: drop the queue, cancel the running job and close when it ends (D70, D78, D79)."""
        self.close_bar.hide()
        self.clear_queue()
        self._cancel()
        self._close_after = True
        self._update_run_buttons()


def run_window(root: Path, project: Optional[str], plugins: PluginApi) -> int:
    """Show the launcher window over ``root`` and run the Qt event loop; return its exit code."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    theme.apply_theme(app, theme.saved_theme())
    unit_palettes.set_palette(theme.saved_unit_palette())
    app.setWindowIcon(logo.icon())
    window = LauncherWindow(root, project, plugins=plugins)
    window.show()
    return app.exec()
