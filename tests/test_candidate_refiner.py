import numpy as np
import pytest

from gwyolo.candidate_set_training import (
    candidate_pair_feature_vector,
    candidate_parent_top1_metrics,
)
from gwyolo.candidate_refiner import (
    candidate_average_precision,
    candidate_arrival_threshold_metrics,
    candidate_crop_contains_arrival,
    candidate_interval_pair_features,
    candidate_pair_truth_support,
    candidate_positive_timing_error_quantiles,
    label_candidate_refiner_rows,
)


def _parent(injection_id: str, split: str, gps: float):
    return {
        "injection_id": injection_id,
        "waveform_id": f"wave-{injection_id}",
        "split": split,
        "gps_block": f"block-{injection_id}",
        "detector_arrival_gps": {"H1": gps, "L1": gps + 0.01},
    }


def _candidate(candidate_id: str, injection_id: str, split: str, ifo: str, start: float):
    return {
        "candidate_id": candidate_id,
        "injection_id": injection_id,
        "waveform_id": f"wave-{injection_id}",
        "split": split,
        "source_family": "BBH",
        "gps_block": f"block-{injection_id}",
        "ifo": ifo,
        "gps_start": start,
        "gps_end": start + 0.1,
        "gps_peak": start + 0.05,
        "proposal_score": 0.7,
    }


def test_candidate_refiner_plan_retains_all_and_counts_coverage_by_hand() -> None:
    parents = [_parent("i1", "val", 100.0)]
    candidates = [
        _candidate("c1", "i1", "val", "H1", 99.8),
        _candidate("c2", "i1", "val", "H1", 103.0),
        _candidate("c3", "i1", "val", "L1", 99.9),
    ]
    rows, report = label_candidate_refiner_rows(
        parents,
        candidates,
        "val",
        positive_padding_seconds=0.5,
        validation_selection_fraction=0.2,
        seed=7,
    )
    assert len(rows) == 3
    assert [row["refiner_positive"] for row in rows] == [True, False, True]
    assert len({row["refiner_role"] for row in rows}) == 1
    assert report["positive_candidates"] == 2
    assert report["negative_candidates"] == 1
    assert report["expected_detector_arrivals"] == 2
    assert report["arrivals_with_positive_candidate"] == 2
    assert report["positive_candidate_coverage_fraction"] == 1.0
    assert report["all_connected_candidates_retained"] is True


def test_candidate_average_precision_by_hand() -> None:
    # Sorted labels are true, false, true: AP = (1/1 + 2/3) / 2.
    value = candidate_average_precision(
        np.asarray([True, True, False]), np.asarray([0.9, 0.7, 0.8])
    )
    assert np.isclose(value, 5 / 6)


def test_candidate_arrival_threshold_metrics_counts_abstention_by_hand() -> None:
    rows = [
        {
            "candidate_id": "a-high",
            "injection_id": "a",
            "ifo": "H1",
            "presence_score": 0.8,
            "refined_timing_error_seconds": 0.005,
        },
        {
            "candidate_id": "a-low",
            "injection_id": "a",
            "ifo": "H1",
            "presence_score": 0.2,
            "refined_timing_error_seconds": 0.1,
        },
        {
            "candidate_id": "b-only",
            "injection_id": "b",
            "ifo": "L1",
            "presence_score": 0.4,
            "refined_timing_error_seconds": 0.015,
        },
    ]
    metrics = candidate_arrival_threshold_metrics(rows, [0.3, 0.5], [0.01, 0.02])
    assert metrics[0]["accepted_arrivals"] == 2
    assert metrics[0]["retained_candidates"] == 2
    assert metrics[0]["top_score_refined_timing"]["0.02"][
        "unconditional_fraction"
    ] == 1.0
    assert metrics[1]["accepted_arrivals"] == 1
    assert metrics[1]["arrival_acceptance_fraction"] == 0.5
    assert metrics[1]["top_score_refined_timing"]["0.01"][
        "conditional_on_acceptance_fraction"
    ] == 1.0


def test_candidate_crop_supervision_uses_physical_arrival_not_interval_label() -> None:
    row = {
        "gps_peak": 100.0,
        "target_detector_arrival_gps": 101.0,
        # A connected-component interval can miss the arrival while its local crop
        # still contains enough strain to provide valid timing supervision.
        "refiner_positive": False,
    }
    assert candidate_crop_contains_arrival(row, 2.5) is True
    assert candidate_crop_contains_arrival(row, 1.0) is False


def test_candidate_positive_timing_summary_uses_local_crop_label() -> None:
    rows = [
        {
            "local_crop_contains_arrival": True,
            "refined_timing_error_seconds": 0.01,
        },
        {
            "local_crop_contains_arrival": True,
            "refined_timing_error_seconds": 0.03,
        },
        {
            "local_crop_contains_arrival": False,
            "refined_timing_error_seconds": 10.0,
        },
    ]
    summary = candidate_positive_timing_error_quantiles(rows)
    assert summary["0.5"] == 0.02
    assert summary["1.0"] == 0.03


def test_candidate_interval_pair_features_by_hand() -> None:
    first = {
        "ifo": "H1",
        "gps_start": 0.0,
        "gps_end": 1.0,
        "proposal_score": 0.5,
    }
    second = {
        "ifo": "L1",
        "gps_start": 1.005,
        "gps_end": 2.005,
        "proposal_score": 0.5,
    }
    features = candidate_interval_pair_features(first, second, 0.01, 0.25)
    assert features["compatible"] is True
    assert np.isclose(features["interval_gap_seconds"], 0.005)
    assert np.isclose(features["center_excess_normalized"], 0.995)
    assert features["width_sum_normalized"] == 8.0
    assert features["proposal_logit_sum"] == 0.0

    first.update({"gps_peak": 0.6})
    second.update({"gps_peak": 1.1})
    support = candidate_pair_truth_support(
        first, second, {"H1": 0.5, "L1": 1.006}, 0.01
    )
    assert support["exact"] is True
    assert support["padded"] is True
    assert np.isclose(support["maximum_peak_error_seconds"], 0.1)
    vector = candidate_pair_feature_vector(first, second, 0.01, 0.25)
    assert vector.shape == (16,)
    assert np.isfinite(vector).all()


def test_candidate_parent_top1_metrics_include_missing_parent_by_hand() -> None:
    metrics = candidate_parent_top1_metrics(
        ["a", "b"],
        ["a", "a"],
        np.asarray([0.1, 0.9]),
        np.asarray([False, True]),
        np.asarray([False, False]),
        np.asarray([1.0, 0.2]),
    )
    assert metrics["parents_with_compatible_pair"] == 1
    assert metrics["compatible_pair_fraction"] == 0.5
    assert metrics["top1_padded_truth_pair_fraction"] == 0.5
    assert metrics["top1_exact_interval_truth_pair_fraction"] == 0.0
    assert metrics["top1_peak_error_seconds_quantiles"]["0.9"] == 0.2


def test_candidate_local_refiner_preserves_time_bins_and_missing_ifo_mask() -> None:
    torch = pytest.importorskip("torch")
    from gwyolo.candidate_refiner import (
        _candidate_refiner_epoch,
        _candidate_timing_prediction,
    )
    from gwyolo.numeric import (
        CandidateEndpointWarmRefiner,
        CandidateLocalSpectrogramRefiner,
        DetectorArrivalSpectrogramNet,
    )

    model = CandidateLocalSpectrogramRefiner(
        detector_count=3, output_bins=64, base_channels=8
    )
    strain = torch.randn(2, 3, 380)
    availability = torch.tensor([[True, True, False], [False, True, True]])
    candidate_ifo = torch.tensor([0, 2])
    presence, timing = model(strain, availability, candidate_ifo)
    assert presence.shape == (2,)
    assert timing.shape == (2, 64)
    assert torch.isfinite(presence).all()
    assert torch.isfinite(timing).all()
    expected = _candidate_timing_prediction(torch.zeros(1, 3), "expected_probability")
    assert torch.allclose(expected, torch.tensor([1.0]))

    with pytest.raises(ValueError, match="unavailable detector"):
        model(strain, availability, torch.tensor([2, 2]))

    endpoint = DetectorArrivalSpectrogramNet(detector_count=3, base_channels=8)
    warm = CandidateEndpointWarmRefiner(
        detector_count=3, output_bins=64, base_channels=8
    )
    warm.load_endpoint_backbone(endpoint.state_dict())
    assert torch.equal(
        warm.spectral_encoder[0].weight, endpoint.spectral_encoder[0].weight
    )
    warm_presence, warm_timing = warm(strain, availability, candidate_ifo)
    assert warm_presence.shape == (2,)
    assert warm_timing.shape == (2, 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    metrics = _candidate_refiner_epoch(
        model,
        [
            (
                strain,
                availability,
                candidate_ifo,
                torch.ones(2),
                torch.tensor([25, 50]),
                torch.tensor([0.1, 0.2], dtype=torch.float64),
            )
        ],
        torch.device("cpu"),
        optimizer,
        1.0,
        2.0,
        1.0,
        0.0,
        "gaussian_coordinate",
        2.0,
        10.0,
        "expected_probability",
        0.25,
        64,
    )
    assert metrics["batches"] == 1
    assert np.isfinite(metrics["loss"])
