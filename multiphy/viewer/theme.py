# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the dark-first Qt theme: the saved choice, the tick and chevron SVGs, the palette, the style sheet, the matplotlib settings, the live switch and the toned table cells (launcher-gui Task 8) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a missing file, a missing key or any other value opens graphite (graphite-palettes Task 2) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the settings file also holds key unit_palette, written only by switch_unit_palette (graphite-palettes Task 4) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the settings file also holds the shortcuts/<action id> keys, which every viewer window reads; test windows read no theme settings (day-table-keys Task 5) | date: 2026-10-14]
"""The viewer's Qt theme, built from the colour tokens of ``display.PALETTES``.

Only the two entry points apply a theme: ``viewer.app.main`` and ``launch.window.run_window`` call
``apply_theme`` and set the saved unit palette; windows built anywhere else (tests included) stay unstyled and
read no theme settings. ``switch_theme`` applies a theme live and writes key ``theme`` of the user's settings file,
``~/.config/multiphy/multiphy.ini``. The file also holds key ``unit_palette``, written only by
``switch_unit_palette``, and the ``shortcuts/<action id>`` keys, written only by ``keys.save_binding`` and
``keys.reset_bindings`` and read by every viewer window (``keys.bindings``); nothing else writes the file. A
missing file, a missing key or any value other than a known theme name opens graphite; any ``unit_palette`` value
other than an id of ``unit_palettes.IDS`` opens ``unit_palettes.DEFAULT``. Table cells carry a token (``tone``,
``fill``) rather than a colour, and ``ToneDelegate`` paints it from the active theme, so a switch recolours them
without a refill.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
import re
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: stderr for a font that fails to load (graphite-palettes Task 3) | date: 2026-10-11]
import sys
# [AGENT EDIT END]
from pathlib import Path
from typing import Optional

import matplotlib
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: matplotlib registers the bundled fonts (graphite-palettes Task 3) | date: 2026-10-11]
from matplotlib import font_manager
# [AGENT EDIT END]

from nems_lbhb.unit_match.viewer import display
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the saved and switched unit palette (graphite-palettes Task 4) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer import unit_palettes
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.qt import Figure, QtCore, QtGui, QtWidgets

TONE_ROLE = QtCore.Qt.UserRole + 101      # a cell's text colour token
FILL_ROLE = QtCore.Qt.UserRole + 102      # a cell's background colour token

_assets = None          # the QTemporaryDir holding the tick and chevron SVGs, made on first use
_notifier = None        # the one ThemeNotifier
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the bundled fonts' folder and the family names found on first use (graphite-palettes Task 3) | date: 2026-10-11]
FONT_DIR = Path(__file__).resolve().parent / 'fonts'
_fonts = None           # the family names fonts() found, made on its first call with a QApplication
# [AGENT EDIT END]

CHECK_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14">'
             '<path d="M3 7.2 L5.8 10 L11 4.2" fill="none" stroke="{accent_ink}" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"/></svg>')
CHEVRON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 10 10">'
               '<path d="M2 3.5 L5 6.5 L8 3.5" fill="none" stroke="{muted}" stroke-width="1.5" '
               'stroke-linecap="round" stroke-linejoin="round"/></svg>')

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the font families: Figtree for the UI, JetBrains Mono for tables and plain-text boxes; radii 8 and 10 px; flat tabs, quiet headers and rounded dock cards (graphite-palettes Task 3) | date: 2026-10-11]
# one rule per line; {token} is a colour token, {check} and {chevron} the SVG paths, {sans} and {mono} the font
# families
_RULES = [
    'QMainWindow, QDialog { background: {ground}; }',
    'QWidget { color: {text}; font-size: 13px; font-family: "{sans}"; }',
    'QToolTip { background: {raised}; color: {text}; border: 1px solid {border}; padding: 4px 6px; }',
    'QMenuBar { background: {ground}; color: {text}; border-bottom: 1px solid {border}; padding: 2px 4px; }',
    'QMenuBar::item { padding: 4px 10px; border-radius: 4px; background: transparent; }',
    'QMenuBar::item:selected { background: {raised}; }',
    'QMenu { background: {panel}; border: 1px solid {border}; padding: 4px; }',
    'QMenu::item { padding: 5px 24px 5px 22px; border-radius: 4px; }',
    'QMenu::item:selected { background: {accent_tint}; color: {text}; }',
    'QMenu::separator { height: 1px; background: {border}; margin: 4px 6px; }',
    'QSplitter::handle { background: {ground}; }',
    'QTabWidget::pane { background: {panel}; border: 1px solid {border}; border-radius: 10px; top: -1px; }',
    'QLineEdit, QComboBox, QPlainTextEdit, QTextEdit, QTextBrowser, QListWidget, QTreeWidget, QTableWidget '
    '{ background: {panel}; border: 1px solid {border}; border-radius: 8px; '
    'selection-background-color: {accent_tint}; selection-color: {text}; }',
    'QLineEdit, QComboBox { padding: 4px 8px; min-height: 20px; }',
    'QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border: 1px solid {accent}; }',
    'QLineEdit:disabled, QComboBox:disabled { color: {muted}; }',
    'QComboBox::drop-down { border: none; width: 22px; }',
    'QComboBox::down-arrow { image: url({chevron}); width: 10px; height: 10px; }',
    'QComboBox QAbstractItemView { background: {panel}; border: 1px solid {border}; padding: 2px; '
    'selection-background-color: {accent_tint}; }',
    'QListWidget::item, QTreeWidget::item { padding: 3px 4px; border-radius: 4px; }',
    'QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected '
    '{ background: {accent_tint}; color: {text}; }',
    'QPushButton, QToolButton { background: {raised}; color: {text}; border: 1px solid {border}; '
    'border-radius: 8px; padding: 6px 14px; }',
    'QToolButton { padding: 4px 8px; }',
    'QPushButton:hover, QToolButton:hover { border-color: {muted}; }',
    'QPushButton:disabled, QToolButton:disabled { color: {muted}; background: {panel}; border-color: {border}; }',
    'QPushButton[primary="true"] { background: {accent}; color: {accent_ink}; border: 1px solid {accent}; '
    'font-weight: 600; }',
    'QPushButton[primary="true"]:disabled { background: {raised}; color: {muted}; border-color: {border}; }',
    *[f'QPushButton[verdict="{v}"] {{ color: {{{v}}}; border: 1px solid {{{v}}}; background: transparent; '
      f'font-weight: 600; }}' for v in ('accept', 'reject', 'unsure')],
    'QLabel[role="hint"] { color: {muted}; font-size: 11px; }',
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's active day header is toned in the accent (day-table-keys Task 3) | date: 2026-10-14]
    *[f'QLabel[tone="{t}"] {{ color: {{{t}}}; }}' for t in ('accept', 'reject', 'unsure', 'muted', 'accent')],
    # [AGENT EDIT END]
    'QToolBar { background: {ground}; border: none; }',
    'QStatusBar { background: {ground}; color: {muted}; }',
    'QFrame#close_bar { background: {raised}; border: 1px solid {border}; border-radius: 10px; }',
    'QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }',
    'QScrollBar::handle:vertical { background: {border}; border-radius: 4px; min-height: 24px; }',
    'QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }',
    'QScrollBar::handle:horizontal { background: {border}; border-radius: 4px; min-width: 24px; }',
    'QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }',
    'QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }',
    'QCheckBox::indicator, QListWidget::indicator { width: 14px; height: 14px; border: 1px solid {muted}; '
    'border-radius: 3px; background: {panel}; }',
    'QCheckBox::indicator:checked, QListWidget::indicator:checked { background: {accent}; '
    'border-color: {accent}; image: url({check}); }',
    'QTableWidget, QPlainTextEdit { font-family: "{mono}"; font-size: 12px; }',
    'QHeaderView::section { background: {panel}; color: {muted}; border: none; '
    'border-bottom: 1px solid {border}; padding: 6px 8px; font-size: 12px; }',
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's compact headers fit their narrow columns (day-table widths) | date: 2026-10-13]
    'QHeaderView[role="compact"]::section { padding: 6px 2px; }',
    # [AGENT EDIT END]
    'QTableWidget { gridline-color: {panel}; }',
    'QTabBar::tab { background: transparent; color: {muted}; padding: 5px 12px; border: none; '
    'border-radius: 6px; margin: 2px 2px 2px 0; }',
    'QTabBar::tab:hover { color: {text}; }',
    'QTabBar::tab:selected { background: {raised}; color: {text}; }',
    'QDockWidget { color: {muted}; font-size: 12px; font-weight: 600; }',
    'QDockWidget::title { background: {ground}; padding: 6px 10px; }',
    'QDockWidget > QWidget { background: {panel}; border: 1px solid {border}; border-radius: 12px; }',
    'QCheckBox::indicator, QListWidget::indicator { border-radius: 4px; }',
    'QToolTip { border-radius: 6px; }',
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Unit colours menu shows 96 px swatch strips; the checked one is outlined in the accent (graphite-palettes Task 4) | date: 2026-10-11]
    'QMenu#unit_palettes { icon-size: 96px; }',
    'QMenu#unit_palettes::item { height: 22px; }',
    'QMenu#unit_palettes::icon:checked { border: 1px solid {accent}; border-radius: 3px; }',
    # [AGENT EDIT END]
]
# [AGENT EDIT END]
_PLACEHOLDER = re.compile(r'\{(\w+)\}')


def settings() -> QtCore.QSettings:
    """The user's settings file ``~/.config/multiphy/multiphy.ini``; the organisation-level file is never read."""
    s = QtCore.QSettings(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, 'multiphy', 'multiphy')
    s.setFallbacksEnabled(False)
    return s


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fallback theme is graphite (graphite-palettes Task 2) | date: 2026-10-11]
def saved_theme() -> str:
    """The saved theme name; a missing file, a missing key or any other value opens graphite."""
    value = settings().value('theme')
    return value if value in display.THEMES else 'graphite'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the saved unit palette; a missing file, a missing key or any other value opens the default (graphite-palettes Task 4) | date: 2026-10-11]
def saved_unit_palette() -> str:
    """The saved unit palette id; a missing file, a missing key or any value not in ``unit_palettes.IDS``
    (exact case) opens ``unit_palettes.DEFAULT``."""
    value = settings().value('unit_palette')
    return value if value in unit_palettes.IDS else unit_palettes.DEFAULT
# [AGENT EDIT END]


def write_assets(name: str) -> tuple:
    """Write theme ``name``'s tick and chevron SVGs into the process's temporary directory; their POSIX paths."""
    global _assets
    if name not in display.THEMES:
        raise ValueError(f'unknown theme {name!r}')
    if _assets is None:
        d = QtCore.QTemporaryDir(QtCore.QDir.tempPath() + '/multiphy-theme-XXXXXX')
        if not d.isValid():
            raise OSError(f'theme assets: {d.errorString()}')
        _assets = d
    p = display.PALETTES[name]
    folder = Path(_assets.path())
    check = folder / f'check_{name}.svg'
    chevron = folder / f'chevron_{name}.svg'
    check.write_text(CHECK_SVG.replace('{accent_ink}', p['accent_ink']))
    chevron.write_text(CHEVRON_SVG.replace('{muted}', p['muted']))
    return check.as_posix(), chevron.as_posix()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the bundled Figtree and JetBrains Mono, registered with Qt and matplotlib on first use; a file that fails to load falls back to the system font (graphite-palettes Task 3) | date: 2026-10-11]
def fonts() -> dict:
    """The font family names of the theme: ``qt_sans``, ``qt_mono``, ``mpl_sans`` and ``mpl_mono``.

    The first call registers ``Figtree.ttf`` (sans) and ``JetBrainsMono.ttf`` (mono) from ``FONT_DIR`` with Qt
    and matplotlib; a later call returns a copy of the cached names. A file that fails to load prints
    ``<file> not loaded · system font`` to stderr, and that kind's Qt name becomes the system general or fixed
    font and its matplotlib name ``DejaVu Sans`` or ``DejaVu Sans Mono``; the other file is unaffected. With no
    ``QGuiApplication`` it raises ``RuntimeError`` and caches nothing.
    """
    global _fonts
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every call with no QGuiApplication raises, the cached one too (graphite-palettes Task 3 fix round 1) | date: 2026-10-11]
    if QtGui.QGuiApplication.instance() is None:
        raise RuntimeError('fonts need a QApplication')
    # [AGENT EDIT END]
    if _fonts is None:
        names = {}
        for kind, file, system, fallback in (
                ('sans', 'Figtree.ttf', QtGui.QFontDatabase.GeneralFont, 'DejaVu Sans'),
                ('mono', 'JetBrainsMono.ttf', QtGui.QFontDatabase.FixedFont, 'DejaVu Sans Mono')):
            path = FONT_DIR / file
            try:
                font_manager.fontManager.addfont(str(path))
                mpl_name = font_manager.FontProperties(fname=str(path)).get_name()
                fid = QtGui.QFontDatabase.addApplicationFont(str(path))
                families = QtGui.QFontDatabase.applicationFontFamilies(fid) if fid != -1 else []
                if not families:
                    raise OSError(f'Qt could not load {path}')
                qt_name = families[0]
            except (OSError, RuntimeError):
                print(f'{path} not loaded · system font', file=sys.stderr)
                qt_name = QtGui.QFontDatabase.systemFont(system).family()
                mpl_name = fallback
            names[kind] = (qt_name, mpl_name)
        _fonts = {'qt_sans': names['sans'][0], 'qt_mono': names['mono'][0],
                  'mpl_sans': names['sans'][1], 'mpl_mono': names['mono'][1]}
    return dict(_fonts)
# [AGENT EDIT END]


def stylesheet(p: dict, check: str, chevron: str) -> str:
    """The application style sheet of palette ``p``, one rule per line."""
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the style sheet names the UI and table font families (graphite-palettes Task 3) | date: 2026-10-11]
    f = fonts()
    values = {'check': check, 'chevron': chevron, 'sans': f['qt_sans'], 'mono': f['qt_mono']}
    # [AGENT EDIT END]

    def fill_in(m):
        key = m.group(1)
        return values[key] if key in values else p[key]
    return '\n'.join(_PLACEHOLDER.sub(fill_in, rule) for rule in _RULES)


def qpalette(p: dict) -> QtGui.QPalette:
    """The Qt palette of colour tokens ``p``; disabled text is muted."""
    pal = QtGui.QPalette()
    roles = {QtGui.QPalette.Window: 'ground', QtGui.QPalette.WindowText: 'text', QtGui.QPalette.Base: 'panel',
             QtGui.QPalette.AlternateBase: 'raised', QtGui.QPalette.Text: 'text', QtGui.QPalette.Button: 'raised',
             QtGui.QPalette.ButtonText: 'text', QtGui.QPalette.ToolTipBase: 'raised',
             QtGui.QPalette.ToolTipText: 'text', QtGui.QPalette.Highlight: 'accent',
             QtGui.QPalette.HighlightedText: 'accent_ink', QtGui.QPalette.Link: 'accent',
             QtGui.QPalette.PlaceholderText: 'muted', QtGui.QPalette.Mid: 'border', QtGui.QPalette.Dark: 'ground',
             QtGui.QPalette.Light: 'raised', QtGui.QPalette.Midlight: 'raised', QtGui.QPalette.Shadow: 'ground'}
    for role, token in roles.items():
        pal.setColor(role, QtGui.QColor(p[token]))
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        pal.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(p['muted']))
    return pal


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the plot frame is the left and bottom spines over a grid, in Figtree (graphite-palettes Task 3) | date: 2026-10-11]
def plot_rc(p: dict) -> dict:
    """matplotlib settings that draw a figure in palette ``p``: left and bottom spines over a grid, text in the
    theme's sans font; Save writes the figure as shown."""
    return {'figure.facecolor': p['panel'], 'axes.facecolor': p['panel'], 'savefig.facecolor': p['panel'],
            'axes.edgecolor': p['plot_spine'], 'axes.labelcolor': p['text'], 'text.color': p['text'],
            'xtick.color': p['muted'], 'ytick.color': p['muted'], 'grid.color': p['plot_grid'],
            'legend.facecolor': p['panel'], 'legend.edgecolor': p['border'], 'legend.labelcolor': p['text'],
            'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.axisbelow': True,
            'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.4, 'xtick.major.size': 3,
            'ytick.major.size': 3, 'font.family': [fonts()['mpl_sans'], 'DejaVu Sans']}
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the application font is the theme's sans at 10 pt (graphite-palettes Task 3) | date: 2026-10-11]
def apply_theme(app: QtWidgets.QApplication, name: str) -> None:
    """Style ``app`` and matplotlib in theme ``name``, its font included; a failure to write the SVGs changes
    nothing."""
    check, chevron = write_assets(name)
    f = fonts()
    display.set_theme(name)
    p = display.colors()
    app.setStyle('Fusion')
    app.setFont(QtGui.QFont(f['qt_sans'], 10))
    app.setPalette(qpalette(p))
    app.setStyleSheet(stylesheet(p, check, chevron))
    matplotlib.rcParams.update(plot_rc(p))
# [AGENT EDIT END]


class ThemeNotifier(QtCore.QObject):
    """Emits ``changed`` with the theme name after a live switch."""
    changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)


def notifier() -> ThemeNotifier:
    """The one ``ThemeNotifier``, made on first call."""
    global _notifier
    if _notifier is None:
        _notifier = ThemeNotifier()
    return _notifier


def switch_theme(name: str) -> str:
    """Apply ``name`` live, save it and tell the windows; ``''``, or a tag naming the file when the save fails.

    A failure to apply raises and saves nothing; a failed save leaves the switch applied and emitted.
    """
    apply_theme(QtWidgets.QApplication.instance(), name)
    s = settings()
    s.setValue('theme', name)
    s.sync()
    notifier().changed.emit(name)
    if s.status() != QtCore.QSettings.NoError:
        return f'theme not saved · {s.fileName()}'
    return ''


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a Unit colours switch: set the palette, save it and tell the windows (graphite-palettes Task 4) | date: 2026-10-11]
def switch_unit_palette(name: str) -> str:
    """Make ``name`` the active unit palette, save it and tell the windows; ``''``, or a tag naming the file when
    the save fails.

    An unknown name raises ``ValueError`` and saves nothing; a failed save leaves the palette set and emitted. The
    windows hear the active theme's name, so they redraw in the new unit colours.
    """
    unit_palettes.set_palette(name)
    s = settings()
    s.setValue('unit_palette', name)
    s.sync()
    notifier().changed.emit(display.theme_name())
    if s.status() != QtCore.QSettings.NoError:
        return f'unit colours not saved · {s.fileName()}'
    return ''
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: left and bottom spines only, in plot_spine; tick labels in the theme's mono font at 8 pt (graphite-palettes Task 3) | date: 2026-10-11]
def style_figure(fig: Figure) -> None:
    """Recolour ``fig`` and its axes in the active theme: the left and bottom spines only, tick labels in the
    theme's mono font at 8 pt."""
    p = display.colors()
    mono = fonts()['mpl_mono']
    fig.set_facecolor(p['panel'])
    for ax in fig.axes:
        ax.set_facecolor(p['panel'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_edgecolor(p['plot_spine'])
        ax.spines['bottom'].set_edgecolor(p['plot_spine'])
        ax.tick_params(which='both', colors=p['muted'], labelfontfamily=mono, labelsize=8)
        ax.xaxis.label.set_color(p['text'])
        ax.yaxis.label.set_color(p['text'])
        ax.title.set_color(p['text'])
# [AGENT EDIT END]


def tone(item: QtWidgets.QTableWidgetItem, token: Optional[str]) -> None:
    """Colour ``item``'s text with ``token``; ``None`` clears it."""
    item.setData(TONE_ROLE, token)


def fill(item: QtWidgets.QTableWidgetItem, token: Optional[str]) -> None:
    """Fill ``item``'s background with ``token``; ``None`` clears it."""
    item.setData(FILL_ROLE, token)


def repolish(widget: QtWidgets.QWidget) -> None:
    """Re-read the style sheet for ``widget`` after one of its properties changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class ToneDelegate(QtWidgets.QStyledItemDelegate):
    """Paints a cell's ``TONE_ROLE`` and ``FILL_ROLE`` tokens in the active theme's colours."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        p = display.colors()
        token = index.data(TONE_ROLE)
        if token:
            option.palette.setColor(QtGui.QPalette.Text, QtGui.QColor(p[token]))
        token = index.data(FILL_ROLE)
        if token:
            option.backgroundBrush = QtGui.QBrush(QtGui.QColor(p[token]))
# [AGENT EDIT END]
