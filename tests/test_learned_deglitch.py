from __future__ import annotations

import math

import numpy as np

from gwyolo.learned_deglitch import signal_retention_metrics


def test_signal_retention_metrics_match_half_signal_by_hand() -> None:
    noise = np.asarray([[10.0, 20.0]])
    signal = np.asarray([[2.0, 4.0]])
    mixture = noise + signal
    cleaned = noise + 0.5 * signal
    metrics = signal_retention_metrics(mixture, cleaned, noise, signal)
    assert metrics["network_signal_projection_retention"] == 0.5
    assert metrics["signal_projection_retention_by_ifo"] == [0.5]
    assert math.isclose(metrics["waveform_change_rms"], math.sqrt(2.5))
    assert math.isclose(metrics["postclean_signal_error_rms"], math.sqrt(2.5))
