from __future__ import annotations

import numpy as np

from gwyolo.numeric import _metrics_from_counts


def test_segmentation_metrics_match_hand_calculation() -> None:
    # Rows are class-wise true-positive, false-positive, false-negative counts.
    metrics = _metrics_from_counts(np.asarray([[8, 2, 4], [5, 5, 5]]))
    assert metrics["chirp"]["precision"] == 0.8
    assert np.isclose(metrics["chirp"]["recall"], 8 / 12)
    assert np.isclose(metrics["chirp"]["iou"], 8 / 14)
    assert np.isclose(metrics["chirp"]["dice"], 16 / 22)
    assert metrics["glitch"]["precision"] == 0.5
    assert metrics["glitch"]["recall"] == 0.5
    assert np.isclose(metrics["glitch"]["iou"], 1 / 3)
    assert np.isclose(metrics["mean_iou"], ((8 / 14) + (1 / 3)) / 2)
