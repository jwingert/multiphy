"""The table panels: browse, rivals, evidence, chains, tracked units and the day table. A table reaches the
window only through the ``Actions`` it is given."""
from multiphy.viewer.panels.tables.browse import BrowsePanel
from multiphy.viewer.panels.tables.chains import ChainPanel
from multiphy.viewer.panels.tables.clusters import ClusterPanel
from multiphy.viewer.panels.tables.days import DayTablePanel
from multiphy.viewer.panels.tables.evidence import EvidencePanel
from multiphy.viewer.panels.tables.rivals import RivalPanel

__all__ = ['BrowsePanel', 'ChainPanel', 'ClusterPanel', 'DayTablePanel', 'EvidencePanel', 'RivalPanel']
