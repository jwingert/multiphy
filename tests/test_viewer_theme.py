"""The viewer's theme. Every settings file and SVG asset a test makes lives under ``tmp_path`` (``theme_env``)."""
import os
import re
import stat
import typing
from pathlib import Path

import matplotlib
import matplotlib.colors
import pandas as pd
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from multiphy.viewer import app as viewer_app
from multiphy.viewer import display, theme
from multiphy.viewer import unit_palettes
from multiphy.viewer.app import App
from multiphy.viewer.panels.plots.base import PlotPanel, PlotTools
from multiphy.viewer.panels.tables.base import _table
from multiphy.viewer.qt import Figure

from .theme_fixtures import theme_env  # noqa: F401  (a fixture, used by name)
from .viewer_fakes import FAKE_MODE, DictData

VIEWER = Path(display.__file__).parent
UNITS = ['a-01', 'b-01', 'c-01', 'd-01']

GRAPHITE = {
    'ground': '#0F1114', 'panel': '#171A1F', 'raised': '#1F232A', 'border': '#262B33',
    'text': '#ECEEF1', 'muted': '#9097A3', 'accent': '#5B9DFF', 'accent_ink': '#0F1114',
    'accent_tint': '#1A2740', 'accept': '#3DD68C', 'reject': '#FF7B72', 'unsure': '#E3B341',
    'plot_fill': '#2A2F37', 'plot_line': '#ECEEF1', 'plot_second': '#7AA2F7',
    'plot_grid': '#22262D', 'plot_spine': '#3A414C', 'proposal_tint': '#18202F'}
DARK = {
    'ground': '#0B1119', 'panel': '#18222F', 'raised': '#212D3D', 'border': '#2A3646', 'text': '#E6EDF3',
    'muted': '#8B98A9', 'accent': '#7DD3FC', 'accent_ink': '#0B1119', 'accent_tint': '#173247',
    'accept': '#3FB950', 'reject': '#F0883E', 'unsure': '#D29922',
    'plot_fill': '#3A4757', 'plot_line': '#E6EDF3', 'plot_second': '#6CB6FF',
    'plot_grid': '#243041', 'plot_spine': '#3A4757', 'proposal_tint': '#172A3B'}
LIGHT = {
    'ground': '#EEF1F5', 'panel': '#FFFFFF', 'raised': '#F3F5F8', 'border': '#D5DBE3', 'text': '#18202B',
    'muted': '#5F6B7A', 'accent': '#0284C7', 'accent_ink': '#FFFFFF', 'accent_tint': '#E0F2FE',
    'accept': '#1F6F4A', 'reject': '#B5451F', 'unsure': '#A08A2A',
    'many': ['#1F6F4A', '#B5451F', '#3F6FA0', '#7A5C9E', '#B08B1E', '#2F8F8F', '#A33D6B', '#5D7D2F',
             '#8A6A4A', '#4A4A8A'],
    'plot_fill': '#CFD6DE', 'plot_line': '#222222', 'plot_second': '#3F6FA0',
    'plot_grid': '#EDF0F4', 'plot_spine': '#B8C1CC', 'proposal_tint': '#EFF8FE'}


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _data():
    pairs = pd.DataFrame({'cellid_a': UNITS[:3], 'cellid_b': UNITS[1:], 'p_global': [.99, .98, .97]})
    return DictData(UNITS, {}, {}, {u: u.split('-')[0] for u in UNITS}, pairs)


def _plot_panels(window):
    return [d.widget() for d in window._docks.values() if isinstance(d.widget(), PlotPanel)]


def _hex(colour):
    return matplotlib.colors.to_hex(colour)


def test_palettes_exact():
    assert display.PALETTES == {'graphite': GRAPHITE, 'dark': DARK, 'light': LIGHT}
    assert list(display.PALETTES) == ['graphite', 'dark', 'light']
    assert display.THEMES == ('graphite', 'dark', 'light')
    assert display.theme_name() == 'graphite'
    for gone in ('COL_A', 'COL_B', 'COL_MANY', 'COL_MUTED', 'VERDICT_COLOR'):
        assert not hasattr(display, gone)
    with pytest.raises(ValueError, match='unknown theme'):
        display.set_theme('purple')


def test_unit_colors_follow_theme(qt, theme_env):
    assert display.unit_colors(1) == ['#E2E8F0']
    assert display.unit_colors(2) == ['#E2E8F0', '#F472B6']
    assert display.unit_colors(16)[15] == '#E2E8F0'
    assert display.unit_colors(5)[:2] == ['#E2E8F0', '#F472B6']
    assert display.unit_colors(0) == []
    assert 'many' not in display.colors()
    display.set_theme('dark')
    assert display.unit_colors(2) == ['#E2E8F0', '#F472B6']
    display.set_theme('light')
    assert display.unit_colors(2) == ['#1F6F4A', '#B5451F']
    assert display.unit_colors(12)[10] == '#1F6F4A'
    display.colors()['many'].append('x')
    assert len(display.colors()['many']) == 10


def test_saved_theme_round_trip(qt, theme_env):
    tmp_path = theme_env
    home_file = Path.home() / '.config' / 'multiphy' / 'multiphy.ini'
    home_before = home_file.read_bytes() if home_file.exists() else None
    assert theme.saved_theme() == 'graphite'
    (tmp_path / 'multiphy.ini').write_text('[General]\ntheme=light\n')
    assert theme.saved_theme() == 'graphite'          # the organisation-level file is never read
    assert theme.switch_theme('light') == ''
    assert 'theme=light' in (tmp_path / 'multiphy' / 'multiphy.ini').read_text().splitlines()
    assert theme.saved_theme() == 'light'
    theme.settings().setValue('theme', 'purple')
    assert theme.saved_theme() == 'graphite'
    assert (home_file.read_bytes() if home_file.exists() else None) == home_before


def test_saved_unit_palette_round_trip(qt, theme_env):
    tmp_path = theme_env
    assert theme.saved_unit_palette() == 'pastel'
    assert theme.switch_unit_palette('tol') == ''
    assert 'unit_palette=tol' in (tmp_path / 'multiphy' / 'multiphy.ini').read_text().splitlines()
    assert theme.saved_unit_palette() == 'tol'
    theme.settings().setValue('unit_palette', 'Tol')
    assert theme.saved_unit_palette() == 'pastel'
    theme.settings().setValue('unit_palette', 'ramp')
    assert theme.saved_unit_palette() == 'ramp'


def test_theme_failures(qt, theme_env, monkeypatch):
    tmp_path = theme_env
    app = QtWidgets.QApplication.instance()
    ro = tmp_path / 'ro'
    ro.mkdir(mode=0o500)
    os.chmod(ro, 0o500)
    try:
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(ro))
        assert theme.switch_theme('light') == f"theme not saved · {ro / 'multiphy' / 'multiphy.ini'}"
        assert display.theme_name() == 'light'
        assert '#0284C7' in app.styleSheet()
        assert list(ro.iterdir()) == []
    finally:
        os.chmod(ro, 0o700)
    display.set_theme('dark')
    sheet = app.styleSheet()
    monkeypatch.setattr(theme, '_assets', None)
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'missing'))
    with pytest.raises(OSError, match='theme assets'):
        theme.apply_theme(app, 'light')
    assert display.theme_name() == 'dark'
    assert app.styleSheet() == sheet
    with pytest.raises(ValueError):
        theme.write_assets('purple')


def _base_style_name(app):
    """The style under the style sheet: ``app.style()`` is Qt's style-sheet wrapper, unnamed, while a sheet is set."""
    sheet = app.styleSheet()
    app.setStyleSheet('')
    name = app.style().name()
    app.setStyleSheet(sheet)
    return name


def test_apply_theme_styles_app(qt, theme_env):
    app = QtWidgets.QApplication.instance()
    app.setStyle('Windows')            # so Fusion afterwards is apply_theme's doing
    theme.apply_theme(app, 'dark')
    assert _base_style_name(app).lower() == 'fusion'
    assert app.palette().color(QtGui.QPalette.Window).name() == '#0b1119'
    assert '#7DD3FC' in app.styleSheet()
    check, chevron = theme.write_assets('dark')
    assert f'url({check})' in app.styleSheet()
    assert f'url({chevron})' in app.styleSheet()
    assert 'stroke="#0B1119"' in Path(check).read_text()
    assert 'stroke="#8B98A9"' in Path(chevron).read_text()
    assert Path(check).is_relative_to(theme_env)
    assert _hex(matplotlib.rcParams['axes.facecolor']) == '#18222f'
    theme.apply_theme(app, 'light')
    assert app.palette().color(QtGui.QPalette.Window).name() == '#eef1f5'
    assert '#0284C7' in app.styleSheet()


def test_tone_delegate_paints_token(qt, theme_env):
    t = _table(['a'])
    t.setSortingEnabled(False)          # the rows stay where they are put
    t.setRowCount(3)
    items = [QtWidgets.QTableWidgetItem(s) for s in ('toned', 'filled', 'plain')]
    theme.tone(items[0], 'accept')
    theme.fill(items[1], 'accent_tint')
    for i, it in enumerate(items):
        t.setItem(i, 0, it)
    delegate = t.itemDelegate()
    assert isinstance(delegate, theme.ToneDelegate)

    def option(row):
        o = QtWidgets.QStyleOptionViewItem()
        delegate.initStyleOption(o, t.model().index(row, 0))
        return o
    assert option(0).palette.color(QtGui.QPalette.Text).name() == '#3dd68c'
    assert option(1).backgroundBrush.color().name() == '#1a2740'
    assert option(2).palette.color(QtGui.QPalette.Text) == t.palette().color(QtGui.QPalette.Text)
    display.set_theme('light')
    assert option(0).palette.color(QtGui.QPalette.Text).name() == '#1f6f4a'
    assert option(1).backgroundBrush.color().name() == '#e0f2fe'
    t.close()


def test_switch_keeps_selection_and_notes(qt, theme_env):
    tmp_path = theme_env
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(_data(), curation_dir=tmp_path / 'c')
    window.select_pair('a-01', 'b-01')
    window.evidence.notes.setPlainText('keep me')
    panels = _plot_panels(window)
    assert panels
    assert all(_hex(p.fig.get_facecolor()) == '#171a1f' for p in panels)
    selection_before = list(window.state.selection)
    spy = []

    def record():
        spy.append(True)
    window.state.changed.connect(record)
    assert theme.switch_theme('light') == ''
    assert (window.state.row.cellid_a, window.state.row.cellid_b) == ('a-01', 'b-01')
    assert list(window.state.selection) == selection_before
    assert window.evidence.notes.toPlainText() == 'keep me'
    assert spy == []
    assert all(_hex(p.fig.get_facecolor()) == '#ffffff' for p in panels)
    window.state.changed.disconnect(record)
    window.close()


def test_verdict_buttons_outlined(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    assert [b.property('verdict') for b in window.evidence.verdict_buttons] == ['accept', 'reject', 'unsure']
    buttons = window.tracked.buttons
    assert buttons['Accept'].property('verdict') == 'accept'
    assert buttons['Reject'].property('verdict') == 'reject'
    assert [b.property('verdict') for label, b in buttons.items() if label not in ('Accept', 'Reject')] == [None] * 4
    for b in [*window.evidence.verdict_buttons, *buttons.values()]:
        assert 'background' not in b.styleSheet()
    rule = ('QPushButton[verdict="accept"] { color: {c}; border: 1px solid {c}; background: transparent;'
            ' font-weight: 600; }')
    assert rule.replace('{c}', '#3FB950') in theme.stylesheet(display.PALETTES['dark'], 'c', 'v')
    assert rule.replace('{c}', '#1F6F4A') in theme.stylesheet(display.PALETTES['light'], 'c', 'v')
    window.close()


def test_plot_toolbar_icons_follow_theme(qt, tmp_path, monkeypatch):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    names = []
    original = PlotTools._icon

    def recording(self, name):
        names.append(name)
        return original(self, name)
    monkeypatch.setattr(PlotTools, '_icon', recording)
    _plot_panels(window)[0].restyle()
    assert names == ['home.png', 'move.png', 'zoom_to_rect.png', 'filesave.png']
    window.close()


def test_plot_colours_from_tokens(qt, theme_env):
    literal = re.compile(r'''(color|colors|facecolor|edgecolor)\s*=\s*['"](?!none['"])''')
    for path in sorted((VIEWER / 'panels' / 'plots').glob('*.py')):
        text = path.read_text()
        assert not literal.search(text), path.name
        assert 'COL_' not in text, path.name
    hex_literal = re.compile(r'''(['"]|color:)#[0-9a-fA-F]{3,8}\b''')
    for path in sorted(VIEWER.rglob('*.py')):
        if path.name in ('display.py', 'theme.py', 'unit_palettes.py'):
            continue
        assert not hex_literal.search(path.read_text()), path
    display.set_theme('dark')
    fig = Figure()
    axes = fig.subplots(1, 2)
    axes[0].set_xlabel('x')
    theme.style_figure(fig)
    assert _hex(fig.get_facecolor()) == '#18222f'
    assert all(_hex(ax.get_facecolor()) == '#18222f' for ax in axes)
    assert _hex(axes[0].spines['left'].get_edgecolor()) == '#3a4757'
    assert _hex(axes[0].xaxis.get_major_ticks()[0].label1.get_color()) == '#8b98a9'
    assert _hex(axes[0].xaxis.label.get_color()) == '#e6edf3'


def test_fonts_bundled_and_registered(qt):
    assert sorted(p.name for p in theme.FONT_DIR.iterdir()) == [
        'Figtree-OFL.txt', 'Figtree.ttf', 'JetBrainsMono-OFL.txt', 'JetBrainsMono.ttf']
    assert (theme.FONT_DIR / 'Figtree.ttf').stat().st_size == 62712
    assert (theme.FONT_DIR / 'JetBrainsMono.ttf').stat().st_size == 187208
    figtree = (theme.FONT_DIR / 'Figtree-OFL.txt').read_text()
    mono = (theme.FONT_DIR / 'JetBrainsMono-OFL.txt').read_text()
    assert figtree.splitlines()[0] == ('Copyright 2022 The Figtree Project Authors '
                                       '(https://github.com/erikdkennedy/figtree)')
    assert mono.splitlines()[0] == ('Copyright 2020 The JetBrains Mono Project Authors '
                                    '(https://github.com/JetBrains/JetBrainsMono)')
    assert 'SIL OPEN FONT LICENSE Version 1.1' in figtree
    assert 'SIL OPEN FONT LICENSE Version 1.1' in mono
    assert theme.fonts() == {'qt_sans': 'Figtree', 'qt_mono': 'JetBrains Mono',
                             'mpl_sans': 'Figtree', 'mpl_mono': 'JetBrains Mono'}


def test_font_failure_falls_back(qt, theme_env, monkeypatch, capsys):
    app = QtWidgets.QApplication.instance()
    d = theme_env / 'fonts'
    d.mkdir()
    (d / 'Figtree.ttf').write_text('not a font')
    monkeypatch.setattr(theme, 'FONT_DIR', d)
    monkeypatch.setattr(theme, '_fonts', None)
    general = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.GeneralFont).family()
    fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont).family()
    assert theme.fonts() == {'qt_sans': general, 'qt_mono': fixed,
                             'mpl_sans': 'DejaVu Sans', 'mpl_mono': 'DejaVu Sans Mono'}
    assert capsys.readouterr().err == (f'{d / "Figtree.ttf"} not loaded · system font\n'
                                       f'{d / "JetBrainsMono.ttf"} not loaded · system font\n')
    theme.apply_theme(app, 'graphite')
    assert app.font().family() == general


def test_fonts_need_a_qapplication(qt, monkeypatch):
    cached = theme.fonts()
    monkeypatch.setattr(QtGui.QGuiApplication, 'instance', staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match='fonts need a QApplication'):
        theme.fonts()
    monkeypatch.setattr(theme, '_fonts', None)
    with pytest.raises(RuntimeError, match='fonts need a QApplication'):
        theme.fonts()
    assert theme._fonts is None
    monkeypatch.undo()
    assert theme.fonts() == cached


def test_shape_rules_every_theme(qt, theme_env):
    app = QtWidgets.QApplication.instance()
    for name, panel, border in (('graphite', '#171A1F', '#262B33'), ('dark', '#18222F', '#2A3646'),
                                ('light', '#FFFFFF', '#D5DBE3')):
        sheet = theme.stylesheet(display.PALETTES[name], 'c', 'v')
        assert (f'QDockWidget > QWidget {{ background: {panel}; border: 1px solid {border}; '
                f'border-radius: 12px; }}') in sheet, name
        assert 'QTableWidget, QPlainTextEdit { font-family: "JetBrains Mono"; font-size: 12px; }' in sheet, name
        assert ('border-radius: 5px' in sheet) is False, name
    theme.apply_theme(app, 'graphite')
    assert app.font().family() == 'Figtree'
    assert app.font().pointSize() == 10
    window = QtWidgets.QMainWindow()
    body = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(body)
    log = QtWidgets.QPlainTextEdit()
    log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))      # as the launcher log is
    browser = QtWidgets.QTextBrowser()
    label = QtWidgets.QLabel('label')
    for w in (log, browser, label):
        lay.addWidget(w)
    window.setCentralWidget(body)
    window.show()
    qt.processEvents()
    assert [w.font().family() for w in (log, browser, label)] == ['JetBrains Mono', 'Figtree', 'Figtree']
    window.close()


def test_plot_style_every_theme(qt, theme_env):
    app = QtWidgets.QApplication.instance()
    for name, grid, spine in (('graphite', '#22262d', '#3a414c'), ('dark', '#243041', '#3a4757'),
                              ('light', '#edf0f4', '#b8c1cc')):
        theme.apply_theme(app, name)
        fig = Figure()
        ax = fig.subplots()
        theme.style_figure(fig)
        assert ax.spines['top'].get_visible() is False, name
        assert ax.spines['right'].get_visible() is False, name
        assert _hex(ax.spines['left'].get_edgecolor()) == spine, name
        line = ax.xaxis.get_gridlines()[0]
        assert line.get_visible(), name
        assert _hex(line.get_color()) == grid, name
        assert line.get_linewidth() == 0.6, name
        label = ax.xaxis.get_major_ticks()[0].label1
        assert label.get_fontfamily() == ['JetBrains Mono'], name
        assert label.get_fontsize() == 8, name
        assert theme.plot_rc(display.colors())['font.family'] == ['Figtree', 'DejaVu Sans'], name


def test_dock_bodies_are_cards(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    assert window._docks
    for panel_id, d in window._docks.items():
        assert d.widget().testAttribute(QtCore.Qt.WA_StyledBackground), panel_id
        m = d.widget().layout().contentsMargins()
        assert (m.left(), m.top(), m.right(), m.bottom()) == (8, 6, 8, 8), panel_id
    window.close()


def test_main_applies_saved_theme(qt, theme_env, monkeypatch):
    theme.settings().setValue('theme', 'light')
    theme.settings().setValue('unit_palette', 'okabe')
    recorded, opened, palettes = [], [], []

    def fake_open_run(run_dir):
        recorded.append(display.theme_name())
        palettes.append(unit_palettes.palette_name())
        opened.append(QtWidgets.QWidget())
        return opened[-1]
    monkeypatch.setattr(viewer_app, 'open_run', fake_open_run)
    QtCore.QTimer.singleShot(0, QtWidgets.QApplication.quit)
    assert viewer_app.main(['--run', str(theme_env)]) == 0
    assert recorded == ['light']
    assert palettes == ['okabe']
    assert '#0284C7' in QtWidgets.QApplication.instance().styleSheet()
    for w in opened:
        w.close()


def test_theme_signatures():
    hints = typing.get_type_hints
    none = type(None)
    assert hints(display.set_theme) == {'name': str, 'return': none}
    assert hints(display.theme_name) == {'return': str}
    assert hints(display.colors) == {'return': dict}
    assert hints(display.unit_colors) == {'n': int, 'return': list}
    assert hints(display.uses_unit_palette) == {'return': bool}
    assert hints(theme.fonts) == {'return': dict}
    assert hints(theme.saved_theme) == {'return': str}
    assert hints(theme.write_assets) == {'name': str, 'return': tuple}
    assert hints(theme.stylesheet) == {'p': dict, 'check': str, 'chevron': str, 'return': str}
    assert hints(theme.plot_rc) == {'p': dict, 'return': dict}
    assert hints(theme.apply_theme) == {'app': QtWidgets.QApplication, 'name': str, 'return': none}
    assert hints(theme.switch_theme) == {'name': str, 'return': str}
    assert hints(theme.saved_unit_palette) == {'return': str}
    assert hints(theme.switch_unit_palette) == {'name': str, 'return': str}
    assert hints(theme.tone) == {'item': QtWidgets.QTableWidgetItem, 'token': typing.Optional[str], 'return': none}
    assert hints(theme.fill) == {'item': QtWidgets.QTableWidgetItem, 'token': typing.Optional[str], 'return': none}
    assert hints(PlotPanel.restyle) == {'return': none}
    assert hints(App.restyle) == {'return': none}
