"""Feature extraction for a project day: its two halves, and its full day pooled from them.

``split`` holds the halves rule, which cuts a day's analysis window into two contiguous
blocks of time. ``full_day`` pools each unit's two halves rows into one full-day row without
reading voltage again, and saves and loads a hash-verified ``full/`` folder. No module here
imports a lab module.
"""
