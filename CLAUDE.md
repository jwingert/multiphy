# multiphy

Cross-day single-unit matching and curation. A standalone package: nothing in it names a lab.

- Stack: Python 3.12 (reference patch 3.12.15), numpy, pandas, scipy, matplotlib; PySide6 optional (the viewer and `launch gui`); pytest for the suite. Use uv 0.13.4 and `uv sync --locked --extra gui --extra test` for the reproducible environment; `pip install .` is an unpinned package install with the `multiphy` console script.
- Python 3.9 syntax: signatures write `Optional[X]` and `List[X]`, never `X | None`.
- Test command, from the repository root: `PYTHONPATH=. QT_QPA_PLATFORM=offscreen python -B -m pytest tests -q -p no:cacheprovider` (`-B` and `-p no:cacheprovider`: no bytecode or cache written into the repository).
- `tests/conftest.py` refuses imports of the lab packages, so the suite proves the package stands alone.
- No inline change markers in code or docs: git history is the record of who changed what and why.
- Lab-specific code (recording readers, launcher commands, well listings) lives in plugins, never here; `multiphy/docs/adding_a_source.md` is the how-to.
- Every command takes its root from `--root DIR` or `MULTIPHY_CACHE_ROOT`.
- Worktrees go under `.claude/worktrees/`; plans live in the ignored `docs/superpowers/plans/`.
