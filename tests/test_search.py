import math
import json

import pytest

from gwyolo.search import (
    calibrate_validation_count,
    aggregate_physical_endpoint_records,
    calibrate_threshold,
    compare_search_methods,
    compare_validation_score_fields,
    detector_subset_noninferiority,
    evaluate_mask_search_robustness,
    evaluate_search,
    far_upper_limit_zero_count,
    paired_vt_comparison,
    run_frozen_search_evaluation,
    run_physical_validation_endpoint,
    run_search_calibration,
    run_validation_injection_diagnostic,
    summarize_injection_efficiency,
)
from gwyolo.io import file_sha256


def test_zero_count_far_limit_is_poisson_2p3_over_time():
    assert math.isclose(far_upper_limit_zero_count(10.0), 0.2302585093, rel_tol=1e-9)


def test_threshold_is_calibrated_only_from_background():
    result = calibrate_threshold([9.0, 8.0, 7.0, 1.0], live_time_years=10.0, target_far_per_year=0.2)
    assert result["threshold"] == 8.0
    assert result["background_count"] == 2
    assert result["far_per_year"] == 0.2


def test_validation_count_threshold_handles_ties_without_exceeding_budget():
    result = calibrate_validation_count([0.9, 0.8, 0.8, 0.1], 2)
    assert result["threshold"] == 0.9
    assert result["background_count"] == 1
    assert result["selection_data"] == "validation_background_only"


def test_physical_endpoint_record_aggregation_pairs_seed_deltas_by_hand():
    result = aggregate_physical_endpoint_records(
        [
            {"scale": 2000, "seed": 1, "weighted_efficiency": 0.2},
            {"scale": 2000, "seed": 2, "weighted_efficiency": 0.4},
            {"scale": 2000, "seed": 3, "weighted_efficiency": 0.3},
            {"scale": 5000, "seed": 1, "weighted_efficiency": 0.3},
            {"scale": 5000, "seed": 2, "weighted_efficiency": 0.35},
            {"scale": 5000, "seed": 3, "weighted_efficiency": 0.5},
        ]
    )
    assert result["minimum_three_seed_gate"] is True
    assert result["scales"][0]["weighted_efficiency_mean"] == pytest.approx(0.3)
    delta = result["adjacent_seed_deltas"][0]
    assert delta["weighted_efficiency_delta_mean"] == pytest.approx((0.1 - 0.05 + 0.2) / 3)
    assert delta["all_seed_deltas_positive"] is False


def test_search_evaluation_reports_weighted_vt():
    result = evaluate_search(
        threshold=8.0,
        background_scores=[9.0, 7.0],
        background_live_time_years=20.0,
        injections=[
            {"ranking_score": 10.0, "vt_weight": 2.0},
            {"ranking_score": 7.0, "vt_weight": 1.0},
            {"ranking_score": 9.0, "vt_weight": 3.0},
        ],
    )
    assert result["background"]["far_per_year"] == 0.05
    assert result["injections"]["recovered"] == 2
    assert result["injections"]["recovered_vt"] == 5.0
    assert result["injections"]["weighted_efficiency"] == 5.0 / 6.0
    assert result["injections"]["efficiency_wilson_95"][0] < 2 / 3


def test_paired_vt_delta_matches_hand_calculation():
    result = paired_vt_comparison(
        [
            {"raw": 10, "clean": 6, "vt_weight": 2, "stratum": "overlap"},
            {"raw": 7, "clean": 7, "vt_weight": 1, "stratum": "overlap"},
            {"raw": 9, "clean": 3, "vt_weight": 3, "stratum": "clean"},
        ],
        threshold_a=8,
        threshold_b=5,
        score_field_a="raw",
        score_field_b="clean",
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["method_a"]["recovered_vt"] == 5
    assert result["method_b"]["recovered_vt"] == 3
    assert result["delta_recovered_vt_b_minus_a"] == -2
    assert result["relative_delta"] == -2 / 5
    assert result["strata"]["overlap"]["weighted_efficiency_b"] == 1.0


def test_detector_subset_noninferiority_uses_paired_lower_bound() -> None:
    comparison = {
        "method_a": {"recovered_vt": 100.0},
        "paired_bootstrap_95": [-8.0, 2.0],
    }
    result = detector_subset_noninferiority(comparison, 0.1)
    assert result["maximum_allowed_absolute_vt_loss"] == 10.0
    assert result["passed"] is True
    comparison["paired_bootstrap_95"][0] = -11.0
    assert detector_subset_noninferiority(comparison, 0.1)["passed"] is False


def test_validation_score_comparison_calibrates_both_fields_by_hand() -> None:
    background = [
        {"split": "val", "raw": 0.9, "coherent": 0.8},
        {"split": "val", "raw": 0.7, "coherent": 0.6},
        {"split": "val", "raw": 0.1, "coherent": 0.2},
    ]
    injections = [
        {"split": "val", "raw": 0.8, "coherent": 0.9, "vt_weight": 2.0},
        {"split": "val", "raw": 0.6, "coherent": 0.7, "vt_weight": 1.0},
    ]
    result = compare_validation_score_fields(
        background, injections, 1, "raw", "coherent", bootstrap_replicates=20, seed=1
    )
    assert result["calibrations"]["raw"]["threshold"] == 0.9
    assert result["calibrations"]["coherent"]["threshold"] == 0.8
    assert result["injection_summaries"]["raw"]["recovered_vt"] == 0.0
    assert result["injection_summaries"]["coherent"]["recovered_vt"] == 2.0
    assert result["paired_comparison"]["delta_recovered_vt_b_minus_a"] == 2.0


def test_mask_search_robustness_applies_clean_margin_and_overlap_gain() -> None:
    background_raw = [
        {"window_id": "w0", "gps_start": 0, "gps_end": 8, "split": "val", "ranking_score": 0.9},
        {"window_id": "w1", "gps_start": 8, "gps_end": 16, "split": "val", "ranking_score": 0.1},
    ]
    background_mask = [
        {"window_id": "w0", "gps_start": 0, "gps_end": 8, "split": "val", "ranking_score": 0.8},
        {"window_id": "w1", "gps_start": 8, "gps_end": 16, "split": "val", "ranking_score": 0.1},
    ]

    def injections(scores):
        return [
            {"injection_id": f"i{index}", "waveform_id": f"s{index}", "vt_weight": 1.0, "split": "val", "ranking_score": score}
            for index, score in enumerate(scores)
        ]

    result = evaluate_mask_search_robustness(
        background_raw,
        background_mask,
        injections([0.95, 0.95]),
        injections([0.85, 0.85]),
        injections([0.95, 0.2]),
        injections([0.85, 0.85]),
        1,
        clean_noninferiority_margin=0.01,
        minimum_contaminated_efficiency_gain=0.4,
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["comparisons"]["clean"]["delta_recovered_vt_b_minus_a"] == 0.0
    assert result["comparisons"]["contaminated"]["delta_recovered_vt_b_minus_a"] == 1.0
    assert result["gates"]["clean_noninferiority"]["passed"] is True
    assert result["gates"]["contaminated_material_gain"]["passed"] is False


def test_search_comparison_calibrates_each_method_on_validation_only():
    validation = [
        {"raw": 9, "clean": 6},
        {"raw": 8, "clean": 5},
        {"raw": 7, "clean": 4},
        {"raw": 1, "clean": 1},
    ]
    result = compare_search_methods(
        validation,
        [{"raw": 9, "clean": 6}, {"raw": 7, "clean": 4}],
        [
            {"raw": 10, "clean": 6, "vt_weight": 2},
            {"raw": 7, "clean": 7, "vt_weight": 1},
            {"raw": 9, "clean": 3, "vt_weight": 3},
        ],
        validation_live_time_years=10,
        test_live_time_years=20,
        target_far_per_year=0.2,
        score_field_a="raw",
        score_field_b="clean",
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["calibrations"]["raw"]["threshold"] == 8
    assert result["calibrations"]["clean"]["threshold"] == 5
    assert result["test_evaluations"]["raw"]["background"]["far_per_year"] == 0.05
    assert result["test_evaluations"]["clean"]["background"]["far_per_year"] == 0.05


def test_frozen_search_command_never_recalibrates_on_test(tmp_path):
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        "".join(json.dumps({"score": value}) + "\n" for value in [9, 8, 7, 1]),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration = run_search_calibration(validation, 10.0, 0.2, "score", calibration_path)
    assert calibration["calibration"]["threshold"] == 8

    test_background = tmp_path / "test_background.jsonl"
    test_background.write_text(
        json.dumps({"score": 100}) + "\n" + json.dumps({"score": 7}) + "\n",
        encoding="utf-8",
    )
    injections = tmp_path / "injections.jsonl"
    injections.write_text(
        json.dumps({"score": 9, "vt_weight": 2})
        + "\n"
        + json.dumps({"score": 7, "vt_weight": 1})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "locked.json"
    result = run_frozen_search_evaluation(
        calibration_path,
        test_background,
        injections,
        20.0,
        output,
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["evaluation"]["threshold"] == 8
    assert result["evaluation"]["background"]["false_alarms"] == 1
    assert result["evaluation"]["injections"]["recovered_vt"] == 2
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_frozen_search_evaluation(
            calibration_path,
            test_background,
            injections,
            20.0,
            output,
            bootstrap_replicates=20,
        )


def test_validation_injection_efficiency_is_weighted_and_split_locked(tmp_path):
    summary = summarize_injection_efficiency(
        [
            {"score": 0.9, "vt_weight": 2},
            {"score": 0.7, "vt_weight": 1},
            {"score": 0.8, "vt_weight": 3},
        ],
        threshold=0.8,
        score_field="score",
        bootstrap_replicates=20,
        seed=1,
    )
    assert summary["recovered"] == 2
    assert summary["recovered_vt"] == 5
    assert summary["weighted_efficiency"] == 5 / 6

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "validation_only_threshold_frozen",
                "score_field": "score",
                "calibration": {"threshold": 0.8},
            }
        ),
        encoding="utf-8",
    )
    injections = tmp_path / "injections.jsonl"
    injections.write_text(
        json.dumps(
            {
                "split": "val",
                "source_family": "BBH",
                "score": 0.9,
                "vt_weight": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = run_validation_injection_diagnostic(
        calibration, injections, tmp_path / "diagnostic.json", bootstrap_replicates=20
    )
    assert result["strata"]["BBH"]["weighted_efficiency"] == 1

    injections.write_text(
        json.dumps(
            {"split": "test", "source_family": "BBH", "score": 0.9, "vt_weight": 2}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-val splits"):
        run_validation_injection_diagnostic(
            calibration, injections, tmp_path / "must-not-write.json", bootstrap_replicates=20
        )


def test_physical_validation_endpoint_verifies_identity_exposure_and_weighted_efficiency(
    tmp_path,
):
    background_rows = [
        {
            "window_id": "w0",
            "split": "val",
            "gps_start": 0,
            "gps_end": 8,
            "gps_block": "b0",
            "ranking_score": 0.9,
        },
        {
            "window_id": "w1",
            "split": "val",
            "gps_start": 8,
            "gps_end": 16,
            "gps_block": "b0",
            "ranking_score": 0.8,
        },
        {
            "window_id": "w2",
            "split": "val",
            "gps_start": 24,
            "gps_end": 32,
            "gps_block": "b1",
            "ranking_score": 0.1,
        },
    ]
    injection_rows = [
        {
            "injection_id": "i0",
            "waveform_id": "s0",
            "split": "val",
            "gps_block": "b0",
            "source_family": "BBH",
            "ranking_score": 0.95,
            "vt_weight": 2,
        },
        {
            "injection_id": "i1",
            "waveform_id": "s1",
            "split": "val",
            "gps_block": "b1",
            "source_family": "BNS",
            "ranking_score": 0.85,
            "vt_weight": 1,
        },
    ]
    background_path = tmp_path / "background.jsonl"
    injection_path = tmp_path / "injections.jsonl"
    background_path.write_text(
        "".join(json.dumps(row) + "\n" for row in background_rows), encoding="utf-8"
    )
    injection_path.write_text(
        "".join(json.dumps(row) + "\n" for row in injection_rows), encoding="utf-8"
    )
    common = {
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "code_commit": "commit",
        "exact_command": "python -m gwyolo.cli score",
        "environment": {"python": "test"},
        "architecture": "fixed_channel",
        "model_ifos": ["H1", "L1", "V1"],
        "enabled_ifos": ["H1", "L1", "V1"],
    }
    background_report = tmp_path / "background_report.json"
    injection_report = tmp_path / "injection_report.json"
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("training: {}\n", encoding="utf-8")
    background_report.write_text(
        json.dumps(
            {
                **common,
                "failed_windows": 0,
                "scored_windows": 3,
                "triggers_path": str(background_path),
                "triggers_sha256": file_sha256(background_path),
            }
        ),
        encoding="utf-8",
    )
    injection_report.write_text(
        json.dumps(
            {
                **common,
                "manifest_sha256": "validation-manifest",
                "failed_injections": 0,
                "scored_injections": 2,
                "triggers_path": str(injection_path),
                "triggers_sha256": file_sha256(injection_path),
            }
        ),
        encoding="utf-8",
    )
    training_report = tmp_path / "training_report.json"
    training_report.write_text(
        json.dumps(
            {
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_path": str(checkpoint),
                "code_commit": "training-commit",
                "config_path": str(config),
                "seed": 1,
                "train_manifest_sha256": "train-manifest",
                "validation_manifest_sha256": "validation-manifest",
                "test_evaluation": None,
            }
        ),
        encoding="utf-8",
    )
    common_checkpoint = file_sha256(checkpoint)
    common_config = file_sha256(config)
    for report_path in (background_report, injection_report):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["checkpoint_sha256"] = common_checkpoint
        report["config_sha256"] = common_config
        report_path.write_text(json.dumps(report), encoding="utf-8")
    result = run_physical_validation_endpoint(
        training_report,
        background_report,
        injection_report,
        maximum_validation_false_alarms=1,
        output=tmp_path / "endpoint.json",
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["background"]["live_time_seconds"] == 24
    assert result["calibration"]["threshold"] == 0.9
    assert result["injections"]["overall"]["recovered_vt"] == 2
    assert result["injections"]["overall"]["weighted_efficiency"] == 2 / 3
    assert result["scientific_claim_allowed"] is False
