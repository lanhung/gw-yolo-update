from __future__ import annotations

import numpy as np
import pytest

from gwyolo.arrival_timing import (
    detector_arrival_bin_targets,
    detector_arrival_errors_seconds,
    detector_arrival_receptive_field_samples,
    detector_network_arrival_errors_seconds,
)


def test_detector_arrival_targets_preserve_ifo_identity_by_hand() -> None:
    targets, offsets, availability = detector_arrival_bin_targets(
        {"H1": 104.001, "L1": 103.999},
        ("H1", "L1", "V1"),
        analysis_start_gps=100.0,
        analysis_duration_seconds=8.0,
        output_bins=1024,
    )

    assert targets.tolist() == [512, 511, -1]
    assert offsets[:2] == pytest.approx([4.001, 3.999])
    assert np.isnan(offsets[2])
    assert availability.tolist() == [True, True, False]
    errors = detector_arrival_errors_seconds(
        targets,
        offsets,
        availability,
        analysis_duration_seconds=8.0,
        output_bins=1024,
    )
    assert errors == pytest.approx([0.00290625, 0.00290625])


def test_detector_arrival_targets_reject_single_ifo_and_out_of_window() -> None:
    with pytest.raises(ValueError, match="at least two"):
        detector_arrival_bin_targets(
            {"H1": 104.0}, ("H1", "L1"), 100.0, 8.0, 1024
        )
    with pytest.raises(ValueError, match="outside"):
        detector_arrival_bin_targets(
            {"H1": 109.0, "L1": 104.0}, ("H1", "L1"), 100.0, 8.0, 1024
        )


def test_detector_network_arrival_errors_include_pair_delay_by_hand() -> None:
    maximum, pairwise = detector_network_arrival_errors_seconds(
        predicted_bins=np.array([[1, 2, 0], [0, 3, 7]]),
        exact_offsets_seconds=np.array([[1.4, 2.7, np.nan], [0.4, np.nan, 7.4]]),
        availability=np.array([[True, True, False], [True, False, True]]),
        analysis_duration_seconds=8.0,
        output_bins=8,
    )

    assert maximum == pytest.approx([0.2, 0.1])
    assert pairwise == pytest.approx([0.3, 0.0])

    with pytest.raises(ValueError, match="two available"):
        detector_network_arrival_errors_seconds(
            np.array([[1, 2]]),
            np.array([[1.4, np.nan]]),
            np.array([[True, False]]),
            8.0,
            8,
        )


def test_detector_arrival_receptive_fields_cover_declared_context() -> None:
    assert detector_arrival_receptive_field_samples("detector_arrival_timing_net_v1") == 129
    assert (
        detector_arrival_receptive_field_samples(
            "detector_arrival_timing_context_net_v2"
        )
        == 8257
    )
    with pytest.raises(ValueError, match="unsupported"):
        detector_arrival_receptive_field_samples("unknown")


def test_detector_arrival_network_emits_per_ifo_high_resolution_logits() -> None:
    torch = pytest.importorskip("torch")
    from gwyolo.numeric import DetectorArrivalTimingContextNet, DetectorArrivalTimingNet

    strain = torch.zeros((2, 3, 64), dtype=torch.float32)
    availability = torch.tensor([[True, True, False], [True, False, True]])
    for model in (
        DetectorArrivalTimingNet(detector_count=3, base_channels=8),
        DetectorArrivalTimingContextNet(detector_count=3, base_channels=8),
    ):
        logits = model(strain, availability)

        assert logits.shape == (2, 3, 8)
        assert torch.isfinite(logits[availability]).all()
        assert torch.isneginf(logits[~availability]).all()
