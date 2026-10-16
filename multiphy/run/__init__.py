"""Everything a unit-matching run needs to find its cache and leave a result behind:
``config`` resolves the shared cache root (and the ``MULTIPHY_CACHE_ROOT`` override) and
mints unique run paths under it, while ``run_dir`` writes the format-2 ``result.json`` a
backend produces and reads it back, including the format-1 files every pilot run left.
"""
