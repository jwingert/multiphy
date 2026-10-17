"""
Cross-day single-unit matching for semi-chronic recordings.

The probe is lowered afresh into a fixed well each day and the brain moves, so
an absolute waveform template cannot carry identity across days. What can is
electrode proximity used as a gate, plus properties that belong to the cell
rather than to where it sits: a high-repeat event-response fingerprint, firing
rate, and the inter-spike-interval histogram.

Layers, in dependency order:

  contract · sources (sub-project 2) · features · datasets · backends · run · evaluation · curation · viewer

The backend contract (``contract``: ``MatchingInput``, ``BackendSpec``, ``MatchResult``,
``Provenance``, the registry) is documented in ``docs/contract.md``; the built-in and
published-method backends live under ``backends``, and ``run`` is the one runner loop,
option schema and CLI shared by every one of them. Adding a method is
``docs/adding_a_backend.md``.

"""

__all__ = ['features', 'store']
__version__ = '0.0.0'
