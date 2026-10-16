"""Deprecated import path for the published-method adapters.

The code lives in ``backends/unitmatch.py``, ``backends/dant.py``, ``backends/deepunitmatch.py``
and ``backends/_upstream.py``; the three backends register as built-ins on import of
``multiphy.backends``. This module keeps the old names importable.
"""
from .contract.inputs import RawWaveformBundle  # noqa: F401
from .backends._upstream import (  # noqa: F401
    DANT_REVISION, DANT_SOURCE, DEEPUNITMATCH_PAPER_REVISION, UNITMATCH_REVISION, UNITMATCH_SOURCE,
    deep_module as _deep_module, module as _module, prepare_bundle as _prepare_bundle,
    source_info as _source_info, version as _version)
from .backends.unitmatch import (  # noqa: F401
    _matrix_pairs, _um_geometry, _um_result, _um_setup, run_unitmatch)
from .backends.dant import run_dant  # noqa: F401
from .backends.deepunitmatch import (  # noqa: F401
    _TRAINING_WORKER, _deep_input_signature, _deep_preprocessing, _deep_training_metadata,
    create_deepunitmatch_model, load_deepunitmatch_checkpoint, prepare_deepunitmatch_training,
    run_deepunitmatch, train_deepunitmatch)


def register_external_backends():
    """No-op kept for old callers: built-ins register on import now."""
