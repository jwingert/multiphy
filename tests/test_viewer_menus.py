# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Settings and Help menus of both windows, the guide window, the launcher's log tones and font, the Run accent, the close bar name and run_window applying the saved theme (launcher-gui Task 9) | date: 2026-10-07]
"""The Settings and Help menus, the guide window and the themed launcher. Every settings file and SVG asset a
theme test makes lives under ``tmp_path`` (``theme_env``)."""
from __future__ import annotations

import os
import typing
from pathlib import Path

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Unit colours tests read line colours and build waveform data (graphite-palettes Task 4) | date: 2026-10-11]
import matplotlib.colors
import numpy as np
import pandas as pd
# [AGENT EDIT END]
from PySide6 import QtCore, QtGui, QtWidgets
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the keys tests catch refusals and count the key notifier's signals (day-table-keys Task 4) | date: 2026-10-14]
import pytest
from PySide6.QtTest import QSignalSpy
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shortcuts window tests record keys into its editors (day-table-keys Task 5) | date: 2026-10-14]
from PySide6.QtTest import QTest
# [AGENT EDIT END]

from nems_lbhb.unit_match.launch import window as window_module
from nems_lbhb.unit_match.launch.plugins import load_plugins
from nems_lbhb.unit_match.launch.window import LauncherWindow, log_tone, run_window
from nems_lbhb.unit_match.viewer import display, menus, theme
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the saved keys module (day-table-keys Task 4) | date: 2026-10-14]
from nems_lbhb.unit_match.viewer import keys
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Unit colours tests read the active palette (graphite-palettes Task 4) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer import unit_palettes
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.app import App
from nems_lbhb.unit_match.viewer.menus import DOCS, GuideWindow, install_menus, open_guide

from .launch_window_fixtures import FakeProcess, qt, runs  # noqa: F401 (fixtures by name)
from .theme_fixtures import theme_env  # noqa: F401  (a fixture, used by name)
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Unit colours switch test gives the viewer waveform data (graphite-palettes Task 4) | date: 2026-10-11]
from .viewer_fakes import FAKE_MODE, DictData

# the board's first eight colours of each palette, lowercase
SWATCHES = {
    'pastel': ['#e2e8f0', '#f472b6', '#fef08a', '#c4b5fd', '#fcd34d', '#86efac', '#fda4af', '#5eead4'],
    'tol': ['#77aadd', '#ee8866', '#eedd88', '#ffaabb', '#99ddff', '#44bb99', '#bbcc33', '#aaaa00'],
    'maxdist': ['#4bffff', '#e14b96', '#f0a5c3', '#3c962d', '#d26978', '#2dd2ff', '#2d96b4', '#78ffd2'],
    'okabe': ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7', '#dddddd'],
    'tab20': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'ramp': ['#4e7cc8', '#977dcb', '#cd80c1', '#f288ae', '#ff999a', '#ffb18b', '#ffce86', '#f7ea92'],
}
UNITS = ['a-01', 'b-01', 'c-01', 'd-01']
# [AGENT EDIT END]


def _launcher(runs) -> LauncherWindow:
    return LauncherWindow(runs['root'], plugins=load_plugins(runs['root']), process_factory=FakeProcess,
                          interpreter_lookup=lambda: None)


def _texts(actions) -> list:
    return [action.text() for action in actions]


def _block_colours(edit) -> dict:
    """``{block text: foreground name}`` of a text edit's non-empty blocks."""
    colours = {}
    block = edit.document().begin()
    while block.isValid():
        if block.text():
            colours[block.text()] = block.begin().fragment().charFormat().foreground().color().name()
        block = block.next()
    return colours


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: both Help menus list Keyboard shortcuts after the guides, before the separator; tag wraps the whole test (day-table-keys Task 5) | date: 2026-10-14]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the test takes theme_env, so the App's shortcut reads use a settings file under tmp_path, never the user's multiphy.ini; tag wraps the whole test (day-table-keys Task 5 fix round 1) | date: 2026-10-14]
def test_menus_in_both_windows(qt, runs, theme_env):
    tmp_path = theme_env
    launcher = _launcher(runs)
    assert _texts(launcher.menuBar().actions()) == ['&Settings', '&Help']
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Settings holds Theme, then Unit colours, then the launcher's Lab root (graphite-palettes Task 4) | date: 2026-10-11]
    assert _texts(launcher.settings_menu.actions()) == ['&Theme', '&Unit colours', '', 'Lab root…']
    assert _texts(launcher.unit_palette_menu.actions()) == ['Graphite pastel', 'Tol light +', 'Max-distance',
                                                            'Okabe–Ito + tints', 'Tableau 20', 'Day ramp']
    assert [pid for pid, action in launcher.unit_palette_actions.items() if action.isChecked()] == ['pastel']
    assert launcher.unit_palette_menu.objectName() == 'unit_palettes'
    # [AGENT EDIT END]
    theme_action = launcher.settings_menu.actions()[0]  # kept: PySide drops menu()'s wrapper with a temporary action
    theme_menu = theme_action.menu()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Theme menu offers Graphite first, checked by default (graphite-palettes Task 2) | date: 2026-10-11]
    assert _texts(theme_menu.actions()) == ['Graphite', 'Dark', 'Light']
    assert launcher.theme_actions['graphite'].isChecked()
    assert not launcher.theme_actions['dark'].isChecked() and not launcher.theme_actions['light'].isChecked()
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Help ends with About multiphy, after a separator (logo) | date: 2026-10-13]
    assert _texts(launcher.help_menu.actions()) == ['Launcher guide', 'Viewer guide', 'Keyboard shortcuts', '',
                                                    'About multiphy']
    # [AGENT EDIT END]
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    assert _texts(viewer.menuBar().actions()) == ['&View', '&Settings', '&Help']
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer's Help ends with About multiphy too (logo) | date: 2026-10-13]
    assert _texts(viewer.help_menu.actions()) == ['Launcher guide', 'Viewer guide', 'Keyboard shortcuts', '',
                                                  'About multiphy']
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer's Settings holds Theme, then Unit colours (graphite-palettes Task 4) | date: 2026-10-11]
    assert _texts(viewer.settings_menu.actions()) == ['&Theme', '&Unit colours']
    # [AGENT EDIT END]
    launcher.close()
    viewer.close()
# [AGENT EDIT END]
# [AGENT EDIT END]


def test_theme_action_switches_all_windows(qt, runs, theme_env):
    tmp_path = theme_env
    launcher = _launcher(runs)
    launcher.append_log('error: x')
    launcher.append_log('exit 0')
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the default theme is graphite (graphite-palettes Task 2) | date: 2026-10-11]
    assert _block_colours(launcher.log_pane)['error: x'] == '#ff7b72'
    # [AGENT EDIT END]
    viewer.theme_actions['light'].trigger()
    assert display.theme_name() == 'light'
    assert viewer.theme_actions['light'].isChecked()
    assert launcher.theme_actions['light'].isChecked()
    colours = _block_colours(launcher.log_pane)
    assert colours['error: x'] == '#b5451f'
    assert colours['exit 0'] == '#18202b'
    ini = tmp_path / 'multiphy' / 'multiphy.ini'
    assert 'theme=light' in ini.read_text().splitlines()
    launcher.close()
    viewer.close()


def test_theme_failure_shown_in_status_bar(qt, theme_env, monkeypatch):
    tmp_path = theme_env
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    ro = tmp_path / 'ro'
    ro.mkdir(mode=0o500)
    os.chmod(ro, 0o500)
    try:
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(ro))
        viewer.theme_actions['light'].trigger()
        assert viewer.statusBar().currentMessage() == f"theme not saved · {ro / 'multiphy' / 'multiphy.ini'}"
        assert display.theme_name() == 'light'
    finally:
        os.chmod(ro, 0o700)
    monkeypatch.setattr(theme, '_assets', None)
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'missing'))
    viewer.theme_actions['dark'].trigger()
    assert viewer.statusBar().currentMessage().startswith('theme not switched · theme assets:')
    assert display.theme_name() == 'light'
    assert viewer.theme_actions['light'].isChecked()
    viewer.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Unit colours menu: its swatch icons and rules, a switch from one window recolouring every window, a failed save tagged in the status bar, and its title on Light (graphite-palettes Task 4) | date: 2026-10-11]
def _line_colours(viewer) -> list:
    """The colours of the waveform panel's lines, lowercase hex."""
    lines = viewer._docks['waveform'].widget().axes[0].get_lines()
    return [matplotlib.colors.to_hex(line.get_color()) for line in lines]


def test_unit_palette_menu_swatches(qt, runs, theme_env):
    launcher = _launcher(runs)
    actions = launcher.unit_palette_menu.actions()
    assert len(actions) == len(SWATCHES)
    for (pid, expected), action in zip(SWATCHES.items(), actions):
        img = action.icon().pixmap(96, 12).toImage()
        assert [img.pixelColor(int(i * 12.25 + 5), 6).name() for i in range(8)] == expected, pid
        assert img.pixelColor(11, 6).alpha() == 0, pid
    sheet = theme.stylesheet(display.PALETTES['graphite'], 'c', 'v')
    assert 'QMenu#unit_palettes { icon-size: 96px; }' in sheet
    assert 'QMenu#unit_palettes::icon:checked { border: 1px solid #5B9DFF; border-radius: 3px; }' in sheet
    launcher.close()


def test_unit_palette_action_switches_all_windows(qt, runs, theme_env):
    tmp_path = theme_env
    launcher = _launcher(runs)
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    pairs = pd.DataFrame({'cellid_a': UNITS[:3], 'cellid_b': UNITS[1:], 'p_global': [.99, .98, .97]})
    data = DictData(UNITS, {u: {'wf': np.sin(np.linspace(0, 6, 40))} for u in UNITS}, {},
                    {u: u.split('-')[0] for u in UNITS}, pairs)
    viewer.set_data(data, curation_dir=tmp_path / 'c')
    viewer.select_pair('a-01', 'b-01')
    assert _line_colours(viewer) == ['#e2e8f0', '#f472b6']
    viewer.unit_palette_actions['tol'].trigger()
    assert _line_colours(viewer) == ['#77aadd', '#ee8866']
    assert unit_palettes.palette_name() == 'tol'
    assert viewer.unit_palette_actions['tol'].isChecked()
    assert launcher.unit_palette_actions['tol'].isChecked()
    assert 'unit_palette=tol' in (tmp_path / 'multiphy' / 'multiphy.ini').read_text().splitlines()
    assert viewer.state.selection == ['a-01', 'b-01']
    assert display.theme_name() == 'graphite'
    launcher.close()
    viewer.close()


def test_unit_palette_save_failure_tagged(qt, theme_env):
    tmp_path = theme_env
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    ro = tmp_path / 'ro'
    ro.mkdir(mode=0o500)
    os.chmod(ro, 0o500)
    try:
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(ro))
        viewer.unit_palette_actions['okabe'].trigger()
        assert viewer.statusBar().currentMessage() == f"unit colours not saved · {ro / 'multiphy' / 'multiphy.ini'}"
        assert unit_palettes.palette_name() == 'okabe'
    finally:
        os.chmod(ro, 0o700)
    viewer.close()


def test_unit_palette_menu_title_on_light(qt, runs, theme_env):
    tmp_path = theme_env
    launcher = _launcher(runs)
    theme.switch_theme('light')
    assert launcher.unit_palette_menu.title() == '&Unit colours · dark themes'
    assert launcher.unit_palette_menu.isEnabled() is True
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    assert viewer.unit_palette_menu.title() == '&Unit colours · dark themes'
    theme.switch_theme('graphite')
    assert launcher.unit_palette_menu.title() == '&Unit colours'
    launcher.close()
    viewer.close()
# [AGENT EDIT END]


def _code_block(guide: GuideWindow, start: str):
    block = guide.guide_view.document().begin()
    while block.isValid():
        if block.text().startswith(start):
            return block
        block = block.next()
    raise AssertionError(f'no block starts {start!r}')


def test_guide_window_renders_doc(qt, theme_env):
    tmp_path = theme_env
    path = DOCS / 'project_folder.md'
    guide = open_guide('Launcher guide', path)
    assert 'The project folder' in guide.guide_view.toPlainText()
    start = 'python -m nems_lbhb.unit_match.launch [--root DIR] init'
    fmt = _code_block(guide, start).blockFormat()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the default theme is graphite (graphite-palettes Task 2) | date: 2026-10-11]
    assert fmt.background().color().name() == '#1f232a'
    # [AGENT EDIT END]
    assert fmt.nonBreakableLines()
    theme.switch_theme('light')
    assert _code_block(guide, start).blockFormat().background().color().name() == '#f3f5f8'
    assert open_guide('Launcher guide', path) is guide
    missing = open_guide('x', tmp_path / 'missing.md')
    assert missing.guide_view.toPlainText() == f"not found · {tmp_path / 'missing.md'}"
    guide.close()
    missing.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a followed guide link and Back keep the code-block styling (launcher-gui final review low finding) | date: 2026-10-07]
def test_guide_link_keeps_code_style(qt, theme_env):
    guide = GuideWindow('Launcher guide', DOCS / 'project_folder.md')
    view = guide.guide_view
    view.setSource(view.source().resolved(QtCore.QUrl('viewer.md#theme')))  # what a link click calls
    assert 'One window opens any completed project run' in view.toPlainText()
    fmt = _code_block(guide, 'python -m nems_lbhb.unit_match.viewer --run DIR').blockFormat()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the default theme is graphite (graphite-palettes Task 2) | date: 2026-10-11]
    assert fmt.background().color().name() == '#1f232a'
    assert fmt.nonBreakableLines()
    view.backward()
    start = 'python -m nems_lbhb.unit_match.launch [--root DIR] init'
    assert _code_block(guide, start).blockFormat().background().color().name() == '#1f232a'
    # [AGENT EDIT END]
    guide.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Help ▸ About multiphy opens one About window: the logo, the name, the version, one line and the credit (logo) | date: 2026-10-13]
def test_about_window(qt, runs, theme_env):
    launcher = _launcher(runs)
    launcher.help_menu.actions()[-1].trigger()
    about = menus.open_about()
    assert about.isVisible()
    assert about.windowTitle() == 'About multiphy'
    assert [label.text() for label in about.lines] == ['multiphy', '0.0.0', 'cross-day unit matching']
    assert about.credit.text() == 'logo · PRN067a-338-1 · peak channel'
    assert about.credit.property('role') == 'hint'
    assert about.credit.toolTip() == ("The logo draws unit PRN067a-338-1's mean waveform on its peak channel, "
                                      'tracked across three days.')
    picture = about.picture.pixmap()
    assert picture.width() / picture.devicePixelRatio() == menus.ABOUT_PX == 128
    assert picture.toImage().pixelColor(0, 0).alpha() == 0
    assert menus.open_about() is about
    about.close()
    launcher.close()
# [AGENT EDIT END]


def test_log_tones():
    assert log_tone('[numexpr.utils INFO] NumExpr defaulting to 8 threads.') == 'muted'
    for line in ('error: boom', 'killed', 'exit 1', 'exit -1', 'exit 130'):
        assert log_tone(line) == 'reject', line
    for line in ('exit 0', 'exit 10x', '$ launch run P', 'INFO done'):
        assert log_tone(line) == 'text', line


def test_log_pane_colours_and_font(qt, runs):
    launcher = _launcher(runs)
    lines = ['$ launch extract P', '[x INFO] y', 'error: z', 'exit 1']
    for line in lines:
        launcher.append_log(line)
    assert launcher.log_pane.toPlainText().splitlines() == lines
    colours = _block_colours(launcher.log_pane)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the default theme is graphite (graphite-palettes Task 2) | date: 2026-10-11]
    assert [colours[line] for line in lines] == ['#eceef1', '#9097a3', '#ff7b72', '#ff7b72']
    # [AGENT EDIT END]
    fixed = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont).family()
    assert launcher.log_pane.font().family() == fixed
    launcher.close()


def test_run_buttons_primary_and_close_bar(qt, runs):
    launcher = _launcher(runs)
    assert launcher.run_buttons
    assert all(button.property('primary') is True for button in launcher.run_buttons.values())
    assert launcher.close_bar.objectName() == 'close_bar'
    launcher.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run_window also sets the saved unit palette before the window is built (graphite-palettes Task 4) | date: 2026-10-11]
def test_run_window_applies_saved_theme(qt, runs, theme_env, monkeypatch):
    theme.settings().setValue('theme', 'light')
    theme.settings().setValue('unit_palette', 'okabe')
    recorded, opened, palettes = [], [], []

    def fake_window(*args, **kwargs):
        recorded.append(display.theme_name())
        palettes.append(unit_palettes.palette_name())
        opened.append(QtWidgets.QWidget())
        return opened[-1]
    monkeypatch.setattr(window_module, 'LauncherWindow', fake_window)
    QtCore.QTimer.singleShot(0, QtWidgets.QApplication.quit)
    assert run_window(runs['root'], None, load_plugins(runs['root'])) == 0
    assert recorded == ['light']
    assert palettes == ['okabe']
    # [AGENT EDIT END]
    assert '#0284C7' in QtWidgets.QApplication.instance().styleSheet()
    for w in opened:
        w.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open_shortcuts's and show_bindings's signatures and the SHORTCUTS label; tag wraps the whole test (day-table-keys Task 5) | date: 2026-10-14]
def test_menus_signatures():
    hints = typing.get_type_hints
    none = type(None)
    assert hints(install_menus) == {'window': QtWidgets.QMainWindow, 'extra_settings': tuple, 'return': none}
    assert hints(open_guide) == {'title': str, 'path': Path, 'return': GuideWindow}
    assert hints(log_tone) == {'line': str, 'return': str}
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: swatch_icon's signature (graphite-palettes Task 4) | date: 2026-10-11]
    assert hints(menus.swatch_icon) == {'colours': list, 'return': QtGui.QIcon}
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open_about's signature (logo) | date: 2026-10-13]
    assert hints(menus.open_about) == {'return': menus.AboutWindow}
    # [AGENT EDIT END]
    assert menus.GUIDES == (('Launcher guide', 'project_folder.md'), ('Viewer guide', 'viewer.md'))
    assert window_module.INFO_LINE.pattern == r'^\[[^\]]* INFO\]'
    assert hints(menus.open_shortcuts) == {'return': menus.ShortcutsWindow}
    assert hints(menus.ShortcutsWindow.show_bindings) == {'return': none}
    assert menus.SHORTCUTS == 'Keyboard shortcuts'
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one saved key per action: the defaults, a save and an unbind, refused keys, a malformed file, the reset, a failed save and the signatures (day-table-keys Task 4) | date: 2026-10-14]
# the default key of each action, in ACTIONS order
DEFAULTS = {'up': 'W', 'down': 'S', 'left': 'A', 'right': 'D', 'track': 'F', 'clear': 'Esc', 'skip': 'X',
            'accept': 'E', 'reject': 'R', 'unsure': 'U', 'merge': 'Q', 'accept_track': 'Shift+E',
            'reject_track': 'Shift+R', 'unmatch': 'Shift+X', 'undo': 'Ctrl+Z', 'redo': 'Ctrl+Shift+Z'}


def _refusal(action_id, key) -> str:
    """The text of the ``ValueError`` ``keys.save_binding(action_id, key)`` raises."""
    with pytest.raises(ValueError) as caught:
        keys.save_binding(action_id, key)
    return str(caught.value)


def test_bindings_default(qt, theme_env):
    assert keys.bindings() == {'up': 'W', 'down': 'S', 'left': 'A', 'right': 'D', 'track': 'F', 'clear': 'Esc',
                               'skip': 'X', 'accept': 'E', 'reject': 'R', 'unsure': 'U', 'merge': 'Q',
                               'accept_track': 'Shift+E', 'reject_track': 'Shift+R', 'unmatch': 'Shift+X',
                               'undo': 'Ctrl+Z', 'redo': 'Ctrl+Shift+Z'}
    assert list(keys.bindings()) == ['up', 'down', 'left', 'right', 'track', 'clear', 'skip', 'accept', 'reject',
                                     'unsure', 'merge', 'accept_track', 'reject_track', 'unmatch', 'undo', 'redo']


def test_save_binding_round_trips(qt, theme_env):
    spy = QSignalSpy(keys.notifier().changed)
    assert keys.save_binding('right', 'l') == ''
    assert keys.bindings()['right'] == 'L'
    assert keys.save_binding('up', '') == ''
    assert keys.bindings()['up'] == ''
    assert spy.count() == 2


def test_save_binding_refuses_clash_and_bad_keys(qt, theme_env):
    spy = QSignalSpy(keys.notifier().changed)
    assert _refusal('merge', 'shift+e') == 'Shift+E · taken by accept track'
    assert _refusal('merge', 'Ctrl+X, Ctrl+C') == 'Ctrl+X, Ctrl+C · not one key'
    assert _refusal('nope', 'W') == 'nope · no such action'
    assert _refusal('right', 76) == '76 · not one key'
    assert keys.normalized(76) == ''
    assert keys.bindings() == DEFAULTS
    # a refusal saves and emits nothing
    assert spy.count() == 0


def test_bindings_read_malformed_ini(qt, theme_env):
    (theme_env / 'multiphy').mkdir()
    (theme_env / 'multiphy' / 'multiphy.ini').write_text('[shortcuts]\nup=W, S\ndown=D\nleft=\nright=12\n')
    bound = keys.bindings()
    assert (bound['up'], bound['down'], bound['left'], bound['right']) == ('W', 'D', '', '')


def test_reset_bindings(qt, theme_env):
    keys.save_binding('right', 'L')
    keys.save_binding('merge', '')
    assert keys.reset_bindings() == ''
    assert keys.bindings() == DEFAULTS


def test_shortcuts_save_failure_tagged(qt, theme_env):
    ro = theme_env / 'ro'
    (ro / 'multiphy').mkdir(parents=True)
    # one saved key, so the reset has a line to remove: a reset with nothing saved writes nothing
    (ro / 'multiphy' / 'multiphy.ini').write_text('[shortcuts]\nmerge=Q\n')
    os.chmod(ro / 'multiphy', 0o500)
    os.chmod(ro, 0o500)
    try:
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(ro))
        assert keys.save_binding('right', 'L') == f"shortcuts not saved · {ro / 'multiphy' / 'multiphy.ini'}"
        assert keys.reset_bindings() == f"shortcuts not saved · {ro / 'multiphy' / 'multiphy.ini'}"
    finally:
        os.chmod(ro, 0o700)
        os.chmod(ro / 'multiphy', 0o700)


def test_keys_signatures():
    hints = typing.get_type_hints
    assert hints(keys.bindings) == {'return': dict}
    assert hints(keys.save_binding) == {'action_id': str, 'key': str, 'return': str}
    assert hints(keys.reset_bindings) == {'return': str}
    assert hints(keys.normalized) == {'key': str, 'return': str}
    assert hints(keys.notifier) == {'return': keys.KeyNotifier}
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Help ▸ Keyboard shortcuts: one row per action with its key, a recorded key saved and bound at once, a taken key refused and named, a cleared key unbinding, Reset defaults and a failed save tagged (day-table-keys Task 5) | date: 2026-10-14]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the D23 key recipes: a key and the event loop, Shift typed before its key, Shift alone ended by Qt's 1 s timeout; tag wraps the whole functions (day-table-keys Task 5 fix round 2) | date: 2026-10-14]
def _record(w, action_id, key, modifier=QtCore.Qt.NoModifier) -> None:
    """Record ``key`` (with ``modifier``) into the shortcuts window's editor of ``action_id``, then let the event
    loop run."""
    edit = w.edits[action_id]
    edit.setFocus()
    QTest.keyClick(edit, key, modifier)
    QtWidgets.QApplication.processEvents()


def _shift_then(w, action_id, key) -> None:
    """Shift typed before ``key`` into the editor of ``action_id``: an event-loop pass between the two, as a hand
    does it."""
    edit = w.edits[action_id]
    edit.setFocus()
    QTest.keyPress(edit, QtCore.Qt.Key_Shift, QtCore.Qt.ShiftModifier)
    QtWidgets.QApplication.processEvents()
    QTest.keyClick(edit, key, QtCore.Qt.ShiftModifier)
    QTest.keyRelease(edit, QtCore.Qt.Key_Shift, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.processEvents()


def _shift_alone(w, action_id) -> None:
    """Shift pressed and released alone in the editor of ``action_id``; Qt ends the recording 1 s after the last
    key is released."""
    edit = w.edits[action_id]
    edit.setFocus()
    QTest.keyPress(edit, QtCore.Qt.Key_Shift, QtCore.Qt.ShiftModifier)
    QtWidgets.QApplication.processEvents()
    QTest.keyRelease(edit, QtCore.Qt.Key_Shift, QtCore.Qt.NoModifier)
    QTest.qWait(1500)
# [AGENT EDIT END]


def test_shortcuts_window_lists_actions(qt, theme_env):
    w = menus.open_shortcuts()
    assert w.isVisible()
    assert w.windowTitle() == 'Keyboard shortcuts'
    assert [w.table.horizontalHeaderItem(c).text() for c in range(w.table.columnCount())] == ['action', 'key']
    assert [w.table.item(r, 0).text() for r in range(w.table.rowCount())] == [
        'up', 'down', 'previous day', 'next day', 'string track', 'clear string', 'drop day', 'accept pair',
        'reject pair', 'unsure pair', 'merge', 'accept track', 'reject track', 'unmatch unit', 'undo', 'redo']
    assert [w.edits[a[0]].keySequence().toString() for a in keys.ACTIONS] == [
        'W', 'S', 'A', 'D', 'F', 'Esc', 'X', 'E', 'R', 'U', 'Q', 'Shift+E', 'Shift+R', 'Shift+X', 'Ctrl+Z',
        'Ctrl+Shift+Z']
    assert w.table.item(10, 0).toolTip() == 'joins the tracks of the selected units'
    # one key per editor, a clear button and no placeholder once painted; nothing in the table selects
    QtWidgets.QApplication.processEvents()
    edit = w.edits['merge']
    assert edit.maximumSequenceLength() == 1
    assert edit.isClearButtonEnabled() is True
    assert edit.findChild(QtWidgets.QLineEdit).placeholderText() == ''
    assert w.table.selectionMode() == QtWidgets.QAbstractItemView.NoSelection
    assert menus.open_shortcuts() is w
    w.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the clear step clicks the editor's real clear button, the only way to unbind (D23); tag wraps the whole test (day-table-keys Task 5 fix round 2) | date: 2026-10-14]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refused Shift+E leaves accept track holding it, so both actions keep their keys (Review Focus 7); tag wraps the whole test (day-table-keys final review fix round 1) | date: 2026-10-14]
def test_shortcuts_window_edits_and_refuses(qt, theme_env):
    tmp_path = theme_env
    viewer = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    w = menus.open_shortcuts()
    _record(w, 'right', QtCore.Qt.Key_L)
    assert keys.bindings()['right'] == 'L'
    assert viewer.shortcuts['right'].key().toString() == 'L'
    assert w.status.text() == ''
    _record(w, 'merge', QtCore.Qt.Key_E, QtCore.Qt.ShiftModifier)
    assert w.status.text() == 'Shift+E · taken by accept track'
    assert w.status.property('tone') == 'reject'
    assert w.edits['merge'].keySequence().toString() == 'Q'
    assert keys.bindings()['merge'] == 'Q'
    assert keys.bindings()['accept_track'] == 'Shift+E'
    assert w.edits['accept_track'].keySequence().toString() == 'Shift+E'
    # the clear button
    QTest.mouseClick(w.edits['up'].findChild(QtWidgets.QLineEdit).findChild(QtWidgets.QToolButton),
                     QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert keys.bindings()['up'] == ''
    w.reset_button.click()
    assert keys.bindings()['up'] == 'W'
    assert keys.bindings()['right'] == 'D'
    assert w.edits['right'].keySequence().toString() == 'D'
    w.close()
    viewer.close()
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Shift alone and Shift typed before a taken key keep the saved key, Shift typed before a free key saves it (D23); tag wraps the whole test (day-table-keys Task 5 fix round 2) | date: 2026-10-14]
def test_shortcuts_window_saves_on_finish(qt, theme_env):
    w = menus.open_shortcuts()
    _shift_alone(w, 'merge')
    assert keys.bindings()['merge'] == 'Q'
    assert w.edits['merge'].keySequence().toString() == 'Q'
    assert w.status.text() == ''
    _shift_then(w, 'merge', QtCore.Qt.Key_E)
    assert keys.bindings()['merge'] == 'Q'
    assert w.edits['merge'].keySequence().toString() == 'Q'
    assert w.status.text() == 'Shift+E · taken by accept track'
    _shift_then(w, 'right', QtCore.Qt.Key_L)
    assert keys.bindings()['right'] == 'Shift+L'
    assert w.status.text() == ''
    w.close()
# [AGENT EDIT END]


def test_shortcuts_window_save_failure_tagged(qt, theme_env):
    w = menus.open_shortcuts()
    ro = theme_env / 'ro'
    ro.mkdir(mode=0o500)
    os.chmod(ro, 0o500)
    try:
        QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, str(ro))
        _record(w, 'right', QtCore.Qt.Key_K)
        assert w.status.text() == f"shortcuts not saved · {ro / 'multiphy' / 'multiphy.ini'}"
        assert w.status.property('tone') == 'reject'
    finally:
        os.chmod(ro, 0o700)
    w.close()
# [AGENT EDIT END]
