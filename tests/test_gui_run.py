# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: gui --run opens any project run (Plan C Task 6) | date: 2026-10-02]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open_run returns an App in the run's mode (viewer-split Task 8) | date: 2026-10-07]
"""``viewer.app.open_run`` opens a project run in the mode its selection records."""
import os

from PySide6 import QtWidgets

from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection
from multiphy.viewer.app import App, open_run

from .synthetic_day import extracted_project


def test_gui_run_opens_viewer_by_kind(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    folder = extracted_project(tmp_path)
    project = Project.open(folder.parent, folder.name)
    across = run_selection(Selection(project, ('D1', 'D2')), 'functional_likelihood')
    same = run_selection(Selection(project, ('D1',)), 'functional_likelihood')

    days_window = open_run(across)
    halves_window = open_run(same)

    assert isinstance(days_window, App) and days_window.mode.id == 'across_days'
    assert isinstance(halves_window, App) and halves_window.mode.id == 'same_day'
    for window in (days_window, halves_window):
        window.close()
    app.processEvents()
# [AGENT EDIT END]
# [AGENT EDIT END]
