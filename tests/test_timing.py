from __future__ import annotations

import numpy as np
import pytest

from gwyolo.timing import timing_errors_seconds


def test_timing_errors_use_bin_centers_and_exact_offsets() -> None:
    errors = timing_errors_seconds(
        np.asarray([0, 3]),
        np.asarray([0.20, 1.90]),
        analysis_duration_seconds=4.0,
        time_bins=4,
    )
    assert errors == pytest.approx([0.30, 1.60])


def test_timing_network_preserves_requested_time_grid() -> None:
    torch = pytest.importorskip("torch")
    from gwyolo.numeric import CoalescenceTimingNet

    model = CoalescenceTimingNet(input_channels=3, base_channels=8)
    logits = model(torch.zeros((2, 3, 12, 32)))
    assert tuple(logits.shape) == (2, 32)
