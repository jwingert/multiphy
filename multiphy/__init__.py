# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the package docstring names an event-response fingerprint (launcher-gui Task 10) | date: 2026-10-07]
"""
Cross-day single-unit matching for semi-chronic recordings.

The probe is lowered afresh into a fixed well each day and the brain moves, so
an absolute waveform template cannot carry identity across days. What can is
electrode proximity used as a gate, plus properties that belong to the cell
rather than to where it sits: a high-repeat event-response fingerprint, firing
rate, and the inter-spike-interval histogram.

# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: layers list reflects the backend contract split (Task 13) | date: 2026-09-22]
Layers, in dependency order:

  contract · sources (sub-project 2) · features · datasets · backends · run · evaluation · curation · viewer

The backend contract (``contract``: ``MatchingInput``, ``BackendSpec``, ``MatchResult``,
``Provenance``, the registry) is documented in ``docs/contract.md``; the built-in and
published-method backends live under ``backends``, and ``run`` is the one runner loop,
option schema and CLI shared by every one of them. Adding a method is
``docs/adding_a_backend.md``.
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude-opus-5 | user: wingertj | reason: cross-day match curation tool | date: 2026-09-14]
"""
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Build cache is retired: no precompute, and blocks moved to nems_lbhb.unit_match_lbhb (Plan C Task 7) | date: 2026-10-02]
__all__ = ['features', 'evidence', 'store']
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the package version, shown in Help ▸ About; 0.0.0 until launch (logo) | date: 2026-10-13]
__version__ = '0.0.0'
# [AGENT EDIT END]
# [AGENT EDIT END]
