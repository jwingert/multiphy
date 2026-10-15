# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the table panels, which reach the window only through Actions (viewer-split Task 6) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the package exports the day table (native-same-day Task 9) | date: 2026-10-11]
"""The table panels: browse, rivals, evidence, chains, tracked units and the day table. A table reaches the
window only through the ``Actions`` it is given."""
from nems_lbhb.unit_match.viewer.panels.tables.browse import BrowsePanel
from nems_lbhb.unit_match.viewer.panels.tables.chains import ChainPanel
from nems_lbhb.unit_match.viewer.panels.tables.clusters import ClusterPanel
from nems_lbhb.unit_match.viewer.panels.tables.days import DayTablePanel
from nems_lbhb.unit_match.viewer.panels.tables.evidence import EvidencePanel
from nems_lbhb.unit_match.viewer.panels.tables.rivals import RivalPanel

__all__ = ['BrowsePanel', 'ChainPanel', 'ClusterPanel', 'DayTablePanel', 'EvidencePanel', 'RivalPanel']
# [AGENT EDIT END]
# [AGENT EDIT END]
