"""The plot panels; importing this package registers each one in ``viewer.registry.PANELS``."""
from multiphy.viewer.panels.plots import (drift, fingerprint, footprint, mixture, probe,  # noqa: F401
                                                    spiking, tuning, waveform, wfmap)
