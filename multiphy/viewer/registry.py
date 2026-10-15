# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the plot panels by id (viewer-split Task 4) | date: 2026-10-06]
"""The registered plot panels by id. A plot module registers its class with ``@register_panel``;
``viewer.panels.plots`` imports every plot module so they register."""

PANELS = {}


def register_panel(cls) -> type:
    """Store ``cls`` under its ``panel_id`` and return it; an id already registered is refused."""
    panel_id = cls.panel_id
    if panel_id in PANELS:
        raise ValueError(f'panel {panel_id} is already registered')
    PANELS[panel_id] = cls
    return cls
# [AGENT EDIT END]
