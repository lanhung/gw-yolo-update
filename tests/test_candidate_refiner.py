import json

import numpy as np
import pytest

from gwyolo.candidate_set_training import (
    candidate_pair_aligned_strain_crop,
    candidate_pair_feature_vector,
    candidate_pair_optimizer_budget,
    candidate_pair_strain_feature_vector,
    candidate_parent_top1_metrics,
    run_candidate_pair_scaling_plan,
    run_candidate_pair_scaling_evaluation,
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
    wave = np.sin(np.linspace(0, 8 * np.pi, 100, dtype=np.float32))
    strain_first = {**first, "gps_start": 0.2, "gps_end": 0.8, "gps_peak": 0.5}
    strain_second = {**second, "gps_start": 0.205, "gps_end": 0.805, "gps_peak": 0.505}
    strain_features = candidate_pair_strain_feature_vector(
        strain_first,
        strain_second,
        np.stack([wave, wave]),
        ("H1", "L1"),
        0.0,
        100,
        0.01,
        0.25,
    )
    assert strain_features.shape == (7,)
    assert np.isclose(strain_features[0], 1.0)


def test_candidate_pair_aligned_crop_uses_one_truth_free_gps_axis() -> None:
    first = {"ifo": "H1", "gps_start": 0.8, "gps_end": 1.0}
    second = {"ifo": "L1", "gps_start": 1.0, "gps_end": 1.2}
    strain = np.stack(
        [
            np.arange(40, dtype=np.float32),
            100 + np.arange(40, dtype=np.float32),
        ]
    )
    crop = candidate_pair_aligned_strain_crop(
        first,
        second,
        strain,
        ("H1", "L1"),
        analysis_start_gps=0.0,
        sample_rate=20,
        crop_duration_seconds=1.0,
        clip_amplitude=200.0,
    )
    assert crop.dtype == np.float16
    assert crop.shape == (2, 20)
    assert np.array_equal(crop[0], np.arange(10, 30, dtype=np.float16))
    assert np.array_equal(crop[1], 100 + np.arange(10, 30, dtype=np.float16))


def test_candidate_pair_scaling_plan_counts_physical_parents_not_candidates(
    tmp_path,
) -> None:
    parents = [
        {
            "injection_id": f"i{index}",
            "waveform_id": f"w{index}",
            "gps_block": f"b{index}",
            "split": "train",
        }
        for index in range(3)
    ]
    candidates = [
        {
            "candidate_id": "c0a",
            "injection_id": "i0",
            "split": "train",
            "ifo": "H1",
        },
        {
            "candidate_id": "c0b",
            "injection_id": "i0",
            "split": "train",
            "ifo": "L1",
        },
        {
            "candidate_id": "c2",
            "injection_id": "i2",
            "split": "train",
            "ifo": "H1",
        },
    ]
    parent_path = tmp_path / "parents.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    scale2_path = tmp_path / "scale2.jsonl"
    scale3_path = tmp_path / "scale3.jsonl"
    parent_path.write_text("".join(json.dumps(row) + "\n" for row in parents))
    candidate_path.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates)
    )
    scale2_path.write_text(
        "".join(json.dumps(row) + "\n" for row in parents[:2])
    )
    scale3_path.write_text("".join(json.dumps(row) + "\n" for row in parents))
    report = run_candidate_pair_scaling_plan(
        parent_path,
        candidate_path,
        [f"2={scale2_path}", f"3={scale3_path}"],
        tmp_path / "output",
    )
    assert [row["physical_parent_count"] for row in report["scale_records"]] == [
        2,
        3,
    ]
    assert report["scale_records"][0]["candidates"] == 2
    assert report["scale_records"][0]["zero_candidate_parents"] == 1
    planned = [
        json.loads(line)
        for line in open(report["scale_records"][1]["candidate_manifest"])
    ]
    assert all(row["refiner_role"] == "train" for row in planned)
    assert all(row["training_parent_scale"] == 3 for row in planned)


def test_candidate_pair_budget_separates_fixed_updates_from_fixed_epochs() -> None:
    assert candidate_pair_optimizer_budget(
        {"budget_mode": "fixed_updates", "epochs": 10, "max_optimizer_updates": 12},
        3,
    ) == ("fixed_updates", 12)
    assert candidate_pair_optimizer_budget(
        {"budget_mode": "fixed_epochs", "epochs": 4}, 7
    ) == ("fixed_epochs", 28)
    with pytest.raises(ValueError, match="must not set max updates"):
        candidate_pair_optimizer_budget(
            {
                "budget_mode": "fixed_epochs",
                "epochs": 4,
                "max_optimizer_updates": 28,
            },
            7,
        )


def test_candidate_pair_scaling_evaluation_requires_gain_in_both_controls(
    tmp_path,
) -> None:
    scales = (2000, 5000, 10000)
    config_path = tmp_path / "scaling.yaml"
    config_path.write_text(
        """
candidate_pair_scaling_evaluation:
  expected_scales: [2000, 5000, 10000]
  minimum_final_top1_gain: 0.05
  minimum_final_snr_8_15_top1_gain: 0.03
  minimum_final_peak_p90_reduction_seconds: 0.25
  maximum_intermediate_top1_regression: 0.01
""".lstrip()
    )
    plan_records = [
        {
            "physical_parent_count": scale,
            "parent_manifest_sha256": f"parent-{scale}",
            "candidate_manifest_sha256": f"candidate-{scale}",
        }
        for scale in scales
    ]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "status": "verified_nested_candidate_pair_scaling_plan",
                "scale_records": plan_records,
            }
        )
    )
    specs = {"fixed_updates": [], "fixed_epochs": []}
    top1 = {2000: 0.30, 5000: 0.34, 10000: 0.36}
    snr = {2000: 0.60, 5000: 0.62, 10000: 0.64}
    p90 = {2000: 4.0, 5000: 3.9, 10000: 3.7}
    for mode in specs:
        for scale in scales:
            report_path = tmp_path / f"{mode}-{scale}.json"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "validation_selection_candidate_pair_ranker",
                        "test_evaluation": None,
                        "budget_mode": mode,
                        "train_physical_parents": scale,
                        "architecture": "candidate_pair_trainable_stft_cnn_v3",
                        "optimizer_updates": scale // 2,
                        "train_unique_waveforms": scale,
                        "train_unique_gps_blocks": 12,
                        "run_identity": {
                            "train_injection_manifest_sha256": f"parent-{scale}",
                            "train_candidate_manifest_sha256": f"candidate-{scale}",
                            "validation_injection_manifest_sha256": "validation-parent",
                            "validation_selection_candidate_manifest_sha256": (
                                "validation-candidate"
                            ),
                            "seed": 7,
                        },
                        "selected_validation_metrics": {
                            "top1_padded_truth_pair_fraction": top1[scale],
                            "top1_peak_error_seconds_quantiles": {"0.9": p90[scale]},
                            "pair_average_precision": 0.2,
                        },
                        "selected_validation_strata": {
                            "snr:snr_8_15": {
                                "top1_padded_truth_pair_fraction": snr[scale]
                            }
                        },
                    }
                )
            )
            specs[mode].append(f"{scale}={report_path}")
    result = run_candidate_pair_scaling_evaluation(
        config_path,
        plan_path,
        specs["fixed_updates"],
        specs["fixed_epochs"],
        tmp_path / "evaluation.json",
    )
    assert result["representation_scaling_gate_passed"] is True
    assert result["scaling_diagnosis"] == (
        "waveform_data_limited_signal_at_fixed_gps_support"
    )
    assert result["gps_diversity_held_fixed"] is True
    assert np.isclose(
        result["curves"]["fixed_updates"][-1]["top1_gain_from_smallest"],
        0.06,
    )
    assert np.isclose(
        result["curves"]["fixed_epochs"][-1][
            "peak_p90_reduction_from_smallest_seconds"
        ],
        0.3,
    )


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
