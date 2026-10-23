"""The ``Settings`` and ``Help`` menus that the viewer and the launcher share.

``install_menus`` adds ``Settings`` (a ``Theme`` submenu, then a ``Unit colours`` submenu, then any
window-specific settings) and ``Help`` (one guide per ``GUIDES`` entry, then ``Keyboard shortcuts``, then
``About multiphy`` after a separator). The menus write only the ``theme``, ``unit_palette`` and
``shortcuts/<action id>`` settings (and their ``shortcut_versions`` metadata), through ``theme.switch_theme``, ``theme.switch_unit_palette`` and the
``Keyboard shortcuts`` window (``keys.save_binding``, ``keys.reset_bindings``); the guides and the About window
are read-only. Every window's ``Theme`` and ``Unit colours`` menus follow a switch made in any window. Qt comes
only from ``.qt``.
"""
from pathlib import Path
from typing import Optional

from multiphy.viewer import display, theme
from multiphy.viewer import unit_palettes
from multiphy.viewer.qt import QtCore, QtGui, QtWidgets
from multiphy import __version__
from multiphy.viewer import logo
from multiphy.viewer import keys

DOCS = Path(__file__).resolve().parents[1] / 'docs'
GUIDES = (('Launcher guide', 'project_folder.md'), ('Viewer guide', 'viewer.md'))
ABOUT = 'About multiphy'
ABOUT_LINES = ('multiphy', __version__, 'cross-day unit matching')
ABOUT_CREDIT = 'logo · PRN067a-338-1 · peak channel'
ABOUT_CREDIT_HOVER = ("The logo draws unit PRN067a-338-1's mean waveform on its peak channel, "
                      'tracked across three days.')
ABOUT_PX = 128  # the logo's size in the About window
SHORTCUTS = 'Keyboard shortcuts'
_THEME_LABELS = (('graphite', 'Graphite'), ('dark', 'Dark'), ('light', 'Light'))

_guides = {}  # path.resolve() -> its GuideWindow
_about = []  # the one AboutWindow, once opened
_shortcuts = []  # the one ShortcutsWindow, once opened


class _ThemeFollower(QtCore.QObject):
    """Checks the window's theme action of the active theme and its unit palette action of the active palette,
    and sets the Unit colours title; parented to the window, so the connection ends with it."""

    def __init__(self, window: QtWidgets.QMainWindow):
        super().__init__(window)
        self._window = window

    def follow(self, name: str) -> None:
        action = self._window.theme_actions.get(name)
        if action is not None:
            action.setChecked(True)
        self._window.unit_palette_actions[unit_palettes.palette_name()].setChecked(True)
        self._window.unit_palette_menu.setTitle(_unit_palette_title())


def _switch(window: QtWidgets.QMainWindow, name: str) -> None:
    """Switch to ``name``; a failed save or a failed switch is named in the status bar (D76)."""
    try:
        text = theme.switch_theme(name)
    except OSError as exc:
        window.statusBar().showMessage(f'theme not switched · {exc}')
        window.theme_actions[display.theme_name()].setChecked(True)
        return
    if text:
        window.statusBar().showMessage(text)


def _unit_palette_title() -> str:
    """The Unit colours submenu's title: it names the dark themes while the theme keeps its own unit ramp."""
    return '&Unit colours' if display.uses_unit_palette() else '&Unit colours · dark themes'


def swatch_icon(colours: list) -> QtGui.QIcon:
    """A 96 × 12 px strip of ``colours``: equal rounded cells 2 px apart on a transparent ground."""
    pixmap = QtGui.QPixmap(96, 12)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtCore.Qt.NoPen)
    cell = (96 - 2 * (len(colours) - 1)) / len(colours)
    for i in range(len(colours)):
        painter.setBrush(QtGui.QColor(colours[i]))
        painter.drawRoundedRect(QtCore.QRectF(i * (cell + 2), 0, cell, 12), 2, 2)
    painter.end()
    return QtGui.QIcon(pixmap)


def _switch_palette(window: QtWidgets.QMainWindow, name: str) -> None:
    """Switch the unit palette to ``name``; a failed save is named in the status bar."""
    text = theme.switch_unit_palette(name)
    if text:
        window.statusBar().showMessage(text)


def install_menus(window: QtWidgets.QMainWindow, extra_settings: tuple = ()) -> None:
    """Add ``&Settings`` (``&Theme``: ``Graphite``, ``Dark``, ``Light``; then ``&Unit colours``: one swatch per
    ``unit_palettes.IDS`` entry; then ``extra_settings`` after a separator) and ``&Help`` (the ``GUIDES``, then
    ``SHORTCUTS`` (``Keyboard shortcuts``), then ``ABOUT`` after a separator) to ``window``'s menu bar. The menus
    write only the ``theme``, ``unit_palette`` and ``shortcuts/<action id>`` settings."""
    bar = window.menuBar()
    window.settings_menu = bar.addMenu('&Settings')
    theme_menu = window.settings_menu.addMenu('&Theme')
    group = QtGui.QActionGroup(theme_menu)
    group.setExclusive(True)
    window.theme_actions = {}
    for name, label in _THEME_LABELS:
        action = QtGui.QAction(label, theme_menu)
        action.setCheckable(True)
        action.setChecked(name == display.theme_name())
        action.triggered.connect(lambda _checked=False, n=name: _switch(window, n))
        group.addAction(action)
        theme_menu.addAction(action)
        window.theme_actions[name] = action
    menu = window.settings_menu.addMenu('&Unit colours')
    window.unit_palette_menu = menu
    menu.setObjectName('unit_palettes')
    menu.setTitle(_unit_palette_title())
    palette_group = QtGui.QActionGroup(menu)
    palette_group.setExclusive(True)
    window.unit_palette_actions = {}
    for pid in unit_palettes.IDS:
        action = QtGui.QAction(swatch_icon(unit_palettes.swatch(pid)), unit_palettes.LABELS[pid], menu)
        action.setCheckable(True)
        action.setChecked(pid == unit_palettes.palette_name())
        action.triggered.connect(lambda _checked=False, n=pid: _switch_palette(window, n))
        palette_group.addAction(action)
        menu.addAction(action)
        window.unit_palette_actions[pid] = action
    if extra_settings:
        window.settings_menu.addSeparator()
        for action in extra_settings:
            window.settings_menu.addAction(action)
    window.help_menu = bar.addMenu('&Help')
    for title, file in GUIDES:
        action = QtGui.QAction(title, window.help_menu)
        action.triggered.connect(lambda _checked=False, t=title, f=file: open_guide(t, DOCS / f))
        window.help_menu.addAction(action)
    shortcuts = QtGui.QAction(SHORTCUTS, window.help_menu)
    shortcuts.triggered.connect(lambda _checked=False: open_shortcuts())
    window.help_menu.addAction(shortcuts)
    window.help_menu.addSeparator()
    about = QtGui.QAction(ABOUT, window.help_menu)
    about.triggered.connect(lambda _checked=False: open_about())
    window.help_menu.addAction(about)
    window._theme_follower = _ThemeFollower(window)
    theme.notifier().changed.connect(window._theme_follower.follow)


class GuideWindow(QtWidgets.QMainWindow):
    """A read-only window showing the Markdown guide at ``path``; code blocks scroll sideways (D54)."""

    def __init__(self, title: str, path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 640)
        self.guide_view = QtWidgets.QTextBrowser()
        self.guide_view.setReadOnly(True)
        self.guide_view.setOpenExternalLinks(True)
        self.guide_view.setSearchPaths([str(path.parent)])
        self.setCentralWidget(self.guide_view)
        if path.exists():
            self.guide_view.setSource(QtCore.QUrl.fromLocalFile(str(path)), QtGui.QTextDocument.MarkdownResource)
        else:
            self.guide_view.setPlainText(f'not found · {path}')
        self.restyle_code()
        theme.notifier().changed.connect(self._theme_changed)
        self.guide_view.sourceChanged.connect(self._source_changed)

    def _source_changed(self, _url) -> None:
        self.restyle_code()

    def _theme_changed(self, _name: str) -> None:
        self.restyle_code()

    def restyle_code(self) -> None:
        """Give every code-fence block the active theme's ``raised`` background and non-breakable lines."""
        fmt = QtGui.QTextBlockFormat()
        fmt.setBackground(QtGui.QColor(display.colors()['raised']))
        fmt.setNonBreakableLines(True)
        block = self.guide_view.document().begin()
        while block.isValid():
            if block.blockFormat().hasProperty(QtGui.QTextFormat.BlockCodeFence):
                QtGui.QTextCursor(block).mergeBlockFormat(fmt)
            block = block.next()


def open_guide(title: str, path: Path) -> GuideWindow:
    """Show the guide at ``path``: one window per resolved path, shown, raised and activated again on a
    later call."""
    key = Path(path).resolve()
    guide = _guides.get(key)
    if guide is None:
        guide = GuideWindow(title, Path(path))
        _guides[key] = guide
    guide.show()
    guide.raise_()
    guide.activateWindow()
    return guide


class AboutWindow(QtWidgets.QDialog):
    """The logo at ``ABOUT_PX``, then ``ABOUT_LINES`` (the name larger) and ``ABOUT_CREDIT`` as hint text, its
    prose on hover; centred."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(ABOUT)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(6)
        self.picture = QtWidgets.QLabel()
        self.picture.setPixmap(logo.pixmap(ABOUT_PX, self.devicePixelRatioF()))
        layout.addWidget(self.picture, alignment=QtCore.Qt.AlignHCenter)
        layout.addSpacing(10)
        self.lines = [QtWidgets.QLabel(text) for text in ABOUT_LINES]
        self.lines[0].setStyleSheet('font-size: 20px; font-weight: 600;')
        for label in self.lines:
            layout.addWidget(label, alignment=QtCore.Qt.AlignHCenter)
        layout.addSpacing(10)
        self.credit = QtWidgets.QLabel(ABOUT_CREDIT)
        self.credit.setProperty('role', 'hint')
        self.credit.setToolTip(ABOUT_CREDIT_HOVER)
        layout.addWidget(self.credit, alignment=QtCore.Qt.AlignHCenter)


def open_about() -> AboutWindow:
    """Show the About window: one window, shown, raised and activated again on a later call."""
    if not _about:
        _about.append(AboutWindow())
    about = _about[0]
    about.show()
    about.raise_()
    about.activateWindow()
    return about
class _NoPlaceholder(QtCore.QObject):
    """Clears a line edit's placeholder before each paint. ``QKeySequenceEdit`` sets ``Press shortcut`` again on
    every reset (a key set, a recording, a clear, a focus change), so an editor stays blank only this way."""

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Paint and watched.placeholderText():
            watched.setPlaceholderText('')
        return False


class ShortcutsWindow(QtWidgets.QDialog):
    """One row per ``keys.ACTIONS`` entry: its label (its hover as tooltip) and an editor that records one key.

    A recorded key is saved through ``keys.save_binding`` when its recording ends (``editingFinished``); a
    recording that ends empty (a modifier alone, focus left mid-recording) puts the saved key back and saves
    nothing; only the clear button saves an empty key, leaving the action unbound. A refused key or a failed save is
    named in ``status`` in the reject tone, and the editor shows the saved key again. ``Reset defaults`` calls
    ``keys.reset_bindings`` under the same status rule. The editors follow every save and reset
    (``keys.notifier().changed``).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(SHORTCUTS)
        self.resize(460, 640)
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(len(keys.ACTIONS), 2)
        self.table.setHorizontalHeaderLabels(['action', 'key'])
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self._no_placeholder = _NoPlaceholder(self)
        self.edits = {}
        for row, (aid, label, _default, hover) in enumerate(keys.ACTIONS):
            item = QtWidgets.QTableWidgetItem(label)
            item.setToolTip(hover)
            self.table.setItem(row, 0, item)
            edit = QtWidgets.QKeySequenceEdit()
            edit.setMaximumSequenceLength(1)
            edit.setClearButtonEnabled(True)
            line = edit.findChild(QtWidgets.QLineEdit)
            line.installEventFilter(self._no_placeholder)
            # the editor empties as soon as a modifier goes down, and a recording ends only with editingFinished:
            # the key is read then, never on keySequenceChanged. Only the clear button emits textEdited('')
            edit.editingFinished.connect(lambda a=aid: self._commit(a))
            line.textEdited.connect(lambda text, a=aid: self._commit(a, '') if text == '' else None)
            self.table.setCellWidget(row, 1, edit)
            self.edits[aid] = edit
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self.table)
        foot = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel('')
        self.status.setProperty('role', 'hint')
        self.status.setProperty('tone', '')
        foot.addWidget(self.status, 1)
        self.reset_button = QtWidgets.QPushButton('Reset defaults')
        # Return in the table must not reset every key: the button is never the dialog's default
        self.reset_button.setAutoDefault(False)
        self.reset_button.clicked.connect(self._reset)
        foot.addWidget(self.reset_button)
        layout.addLayout(foot)
        self.show_bindings()
        self.table.resizeRowsToContents()
        keys.notifier().changed.connect(self.show_bindings)

    def show_bindings(self) -> None:
        """Set each editor to its saved key (``keys.bindings()``) with its signals blocked, so nothing is saved;
        ``status`` is left alone."""
        bound = keys.bindings()
        for aid, edit in self.edits.items():
            with QtCore.QSignalBlocker(edit):
                edit.setKeySequence(QtGui.QKeySequence(bound[aid]))

    def _commit(self, action_id: str, key: Optional[str] = None) -> None:
        """Save ``key`` for ``action_id`` (``None``: the key its editor holds now), unless it is the saved key. A
        read ``''`` (a recording that ended empty) saves nothing and the editors show the saved keys again; a
        refusal or a failed save is named in ``status`` and the editors show the saved keys again."""
        read = key is None
        if read:
            key = self.edits[action_id].keySequence().toString(QtGui.QKeySequence.PortableText)
        if key == keys.bindings()[action_id]:
            return
        if read and key == '':
            self.show_bindings()
            return
        try:
            text = keys.save_binding(action_id, key)
        except ValueError as exc:
            text = str(exc)
        self._show_status(text)
        if text:
            self.show_bindings()

    def _reset(self) -> None:
        """Every action back to its default key; a failed save is named in ``status``."""
        self._show_status(keys.reset_bindings())

    def _show_status(self, text: str) -> None:
        """``text`` in ``status`` in the reject tone; ``''`` clears it and its tone."""
        self.status.setText(text)
        self.status.setProperty('tone', 'reject' if text else '')
        theme.repolish(self.status)


def open_shortcuts() -> ShortcutsWindow:
    """Show the Keyboard shortcuts window: one window, made on the first call; each call shows the saved keys and
    clears the status, then shows, raises and activates it."""
    if not _shortcuts:
        _shortcuts.append(ShortcutsWindow())
    window = _shortcuts[0]
    window.show_bindings()
    window._show_status('')
    window.show()
    window.raise_()
    window.activateWindow()
    return window
