# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: new extract/ subpackage: one pipeline turns a day's source into its halves and full-day features (multiphy project folder Task 9) | date: 2026-09-25]
"""Feature extraction for a project day: its two halves, and its full day pooled from them.

``split`` holds the halves rule, which cuts a day's analysis window into two contiguous
blocks of time. ``full_day`` pools each unit's two halves rows into one full-day row without
reading voltage again, and saves and loads a hash-verified ``full/`` folder. No module here
imports a lab module.
"""
# [AGENT EDIT END]
