# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every plot module is imported so its panel registers (viewer-split Task 4) | date: 2026-10-06]
"""The plot panels; importing this package registers each one in ``viewer.registry.PANELS``."""
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fingerprint and mixture modules register their panels (viewer-split Task 5) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the drift module registers its panel (drift-view Task 4) | date: 2026-10-08]
from nems_lbhb.unit_match.viewer.panels.plots import (drift, fingerprint, footprint, mixture, probe,  # noqa: F401
                                                    spiking, tuning, waveform, wfmap)
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
