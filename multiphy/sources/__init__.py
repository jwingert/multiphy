"""Source readers: each turns one day's original files into a ``DaySource``.

``base`` holds the reader protocol, ``DaySource``, the reader registry (built-ins plus the
``multiphy.sources`` entry point) and ``read_day``, which runs a day's readers in role order.
``events_csv`` reads trials, stimuli and baseline windows from a CSV file. No module here
imports a lab module or reads voltage.
"""
