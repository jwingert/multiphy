# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: evidence.py moved under backends/local; shim keeps old imports working | date: 2026-09-22]
"""Moved to backends/local/evidence.py and backends/local/calibrated.py; shim for old imports."""
from nems_lbhb.unit_match.backends.local.evidence import *  # noqa: F401,F403
from nems_lbhb.unit_match.backends.local.evidence import (  # noqa: F401
    ALL_PROPERTIES, PIPELINE_VERSION, Criteria, Ensemble, LogLR, Mixture, corr_matrix, cosine, dlog,
    estimate_offset, gates_passed, pair_evidence, pearson, register_depths, spearman_brown)
from nems_lbhb.unit_match.backends.local.calibrated import CalibratedPairModel  # noqa: F401
# [AGENT EDIT END]
