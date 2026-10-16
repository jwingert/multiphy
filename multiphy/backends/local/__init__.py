"""Local matching code, one module per mechanism.

- ``functional_likelihood``: backend ``functional_likelihood``, firing, waveform and optional event-response likelihood ratios with a mixture posterior per day pair.
- ``pair_logistic``: backend ``pair_logistic``, a logistic pair classifier per project, calibrated on held-out sessions (alternate days, or one half of a same-day run).
- ``tracker``: the common tracker (mutual margins, one unit per session, measured contradictions) for any pair scores.
- ``evidence``: pair evidence, proximity gates, and the reference ensemble and mixture.
- ``calibrated``: ``CalibratedPairModel``, a regularized pair classifier with an independent calibration partition.
- ``registration``: match-driven depth registration, one rigid depth offset per day.
- ``examples``: labeled same/different examples pooled over every day of a project.
- ``references``: same-unit references from alternating halves and resampled splits of a day's trials, else of its 5 s chunks.
- ``neighbourhood``: the peak-threshold channel neighbourhood per unit, the pair channel union and the spike window for the local waveform properties.
"""
