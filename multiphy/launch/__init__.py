# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: new launch/ subpackage: the init, extract and status commands behind the command line (multiphy project folder Task 14) | date: 2026-09-25]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the launcher window (launcher-gui Task 5) | date: 2026-10-07]
"""The multiphy launcher: the commands a person runs on a project folder.

``commands`` holds ``init_day`` and ``init_from`` (add days to a project), ``extract`` (extract
the days not yet extracted) and ``status`` with ``status_lines`` (check each day's sources,
features and runs). The command line calls these functions; ``window`` holds the launcher window,
which runs every command as a child ``python -m nems_lbhb.unit_match.launch``. No module here
imports a lab module.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
