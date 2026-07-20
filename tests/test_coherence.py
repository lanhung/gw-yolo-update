from __future__ import annotations

import numpy as np
import pytest

from gwyolo.coherence import (
    arrival_time_coherence_gate,
    pairwise_lag_coherence,
    stack_detector_set,
)


def test_detector_set_stack_preserves_explicit_missing_ifo_mask() -> None:
    h1 = np.ones((2, 3, 4), dtype=np.float32)
    v1 = np.full((2, 3, 4), 3.0, dtype=np.float32)
    stacked, available = stack_detector_set({"H1": h1, "V1": v1}, ("H1", "L1", "V1"))
    assert stacked.shape == (3, 2, 3, 4)
    assert available.tolist() == [1.0, 0.0, 1.0]
    assert np.count_nonzero(stacked[1]) == 0
    assert np.array_equal(stacked[2], v1)


def test_arrival_time_gate_is_hand_calculated_with_uncertainty() -> None:
    report = arrival_time_coherence_gate(
        {"H1": 100.0, "L1": 100.012},
        {"H1-L1": 0.010},
        timing_uncertainty_seconds=0.001,
    )
    pair = report["pairs"][0]
    assert pair["observed_delay_seconds"] == pytest.approx(0.012)
    assert pair["allowed_delay_seconds"] == pytest.approx(0.012)
    assert report["passed"]

    failed = arrival_time_coherence_gate(
        {"H1": 100.0, "L1": 100.0121},
        {"H1-L1": 0.010},
        timing_uncertainty_seconds=0.001,
    )
    assert not failed["passed"]


def test_pairwise_lag_coherence_recovers_one_sample_delay() -> None:
    h1 = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0])
    l1 = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
    result = pairwise_lag_coherence(
        {"H1": h1, "L1": l1}, sample_rate=10.0, limits_seconds={"H1-L1": 0.2}
    )[0]
    assert result["lag_samples"] == 1
    assert result["lag_seconds"] == pytest.approx(0.1)
    assert result["absolute_coherence"] == pytest.approx(1.0)


def test_coherence_rejects_missing_pair_contract() -> None:
    with pytest.raises(ValueError, match="missing physical delay limit"):
        arrival_time_coherence_gate({"H1": 1.0, "V1": 1.0}, {"H1-L1": 0.01})
