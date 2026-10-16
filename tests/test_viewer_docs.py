"""The package docs point to ``python -m multiphy.viewer --run DIR`` and name no deleted viewer."""
from pathlib import Path
import re

PACKAGE = Path(__file__).resolve().parents[1] / 'multiphy'
OLD = ('gui --run', 'multiday_viewer', 'benchmark_viewer', 'multiphy.gui')


def test_docs_name_no_old_viewer():
    hits = []
    for path in sorted(PACKAGE.rglob('*.md')):
        # edit-tag reasons are history, so comments are removed first
        text = re.sub(r'<!--.*?-->', '', path.read_text(encoding='utf-8'), flags=re.S)
        hits += [f'{path.relative_to(PACKAGE)}: {old}' for old in OLD if old in text]
    assert hits == []


def test_viewer_doc_names_command_and_modes():
    text = (PACKAGE / 'docs' / 'viewer.md').read_text(encoding='utf-8')
    for word in ('python -m multiphy.viewer --run DIR', 'register_panel', 'across_days', 'same_day',
                 'mixture: not fitted'):
        assert word in text, word


def test_viewer_doc_names_theme():
    text = (PACKAGE / 'docs' / 'viewer.md').read_text(encoding='utf-8')
    for word in ('| `theme.py` |', '| `menus.py` |', '## Theme', 'display.colors()', 'multiphy.ini',
                 'theme not saved', 'Settings'):
        assert word in text, word
    for word in ('| `unit_palettes.py` |', '| `fonts/` |', 'unit_palette', 'unit colours not saved',
                 'Unit colours · dark themes', 'session_date', 'not loaded · system font',
                 'SIL Open Font License 1.1', 'Graphite pastel'):
        assert word in text, word
    project = (PACKAGE / 'docs' / 'project_folder.md').read_text(encoding='utf-8')
    for word in ('`Graphite`, the default', '`Unit colours`'):
        assert word in project, word


def test_docs_name_drift_rules():
    docs = PACKAGE / 'docs'
    viewer = (docs / 'viewer.md').read_text(encoding='utf-8')
    for word in ('## The drift panel', '`drift_frame.py`', 'run predates offsets', 'backend saves no offsets',
                 'registration off', 'group not registered', 'dant kept initial motion', 'unlinked',
                 'no matches to'):
        assert word in viewer, word
    contract = (docs / 'contract.md').read_text(encoding='utf-8')
    for word in ('pair_drifts', 'shank_offsets_um', 'depth_scale', 'unit_offsets_um', 'day_offsets=False',
                 '`native`'):
        assert word in contract, word
    backend = (docs / 'adding_a_backend.md').read_text(encoding='utf-8')
    for word in ('day_offsets=True', '_upstream.recording'):
        assert word in backend, word


def test_viewer_doc_names_day_table():
    text = (PACKAGE / 'docs' / 'viewer.md').read_text(encoding='utf-8')
    for word in ('## The day table', 'proposed_link', '⇄', 'proposal_tint', 'Unmatch unit', 'days.py'):
        assert word in text, word
