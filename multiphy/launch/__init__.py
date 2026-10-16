"""The multiphy launcher: the commands a person runs on a project folder.

``commands`` holds ``init_day`` and ``init_from`` (add days to a project), ``extract`` (extract
the days not yet extracted) and ``status`` with ``status_lines`` (check each day's sources,
features and runs). The command line calls these functions; ``window`` holds the launcher window,
which runs every command as a child ``python -m multiphy.launch``. No module here
imports a lab module.
"""
