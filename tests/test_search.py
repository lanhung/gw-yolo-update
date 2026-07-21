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
    run_candidate_search_calibration,
    run_frozen_candidate_search_evaluation,
    run_physical_validation_endpoint,
    run_search_calibration,
    run_validation_injection_diagnostic,
    summarize_injection_efficiency,
)
from gwyolo.io import file_sha256
from gwyolo.io import canonical_hash
from gwyolo.exposure import (
    candidate_slide_schedule_identity,
    freeze_candidate_block_permutation_schedule,
)


def test_zero_count_far_limit_is_poisson_2p3_over_time():
    assert math.isclose(far_upper_limit_zero_count(10.0), 0.2302585093, rel_tol=1e-9)


def test_threshold_is_calibrated_only_from_background():
    result = calibrate_threshold([9.0, 8.0, 7.0, 1.0], live_time_years=10.0, target_far_per_year=0.2)
    assert result["threshold"] == 8.0
    assert result["background_count"] == 2
    assert result["far_per_year"] == 0.2


def test_empty_candidate_background_freezes_above_probability_support() -> None:
    result = calibrate_threshold([], live_time_years=10.0, target_far_per_year=0.1)
    assert result["threshold"] > 1.0
    assert result["background_count"] == 0


def test_candidate_search_freezes_validation_then_evaluates_disjoint_test(tmp_path) -> None:
    identity = {
        "candidate_checkpoint_sha256": "a" * 64,
        "candidate_config_sha256": "b" * 64,
        "candidate_code_commit": "deadbee",
        "timing_calibration_report_sha256": "c" * 64,
        "physical_delay_limit_seconds": 0.010,
        "empirical_timing_uncertainty_seconds": 0.001,
    }

    def artifacts(split, background_score, injection_scores, block):
        background = tmp_path / f"{split}-background.jsonl"
        background.write_text(
            json.dumps({"split": split, "ranking_score": background_score}) + "\n",
            encoding="utf-8",
        )
        background_sha = file_sha256(background)
        schedule = {
            "schema_version": 2,
            "selection_rule": "nonzero_prefix_to_zero_count_target_within_range_v1",
            "selection_metadata": {"candidate_scores_inspected": False},
            "status": "frozen_candidate_time_slide_schedule",
            "scientific_claim_allowed": False,
            "selection_data": "background_gps_and_detector_availability_only",
            "candidate_scores_inspected": False,
            "split": split,
            "reference_ifo": "H1",
            "shifted_ifo": "L1",
            "step_seconds": 8.0,
            "slide_indices": [1],
            "background_manifest_sha256": background_sha,
            "target_far_per_year": 0.1,
            "zero_count_confidence": 0.5,
            "slide_count": 1,
            "slide_indices_sha256": canonical_hash([1], 64),
            "schedule_exposure_target_reached": True,
            "exposure_plan": {
                "equivalent_live_time_years": 10.0,
                "target_zero_count_upper_reached": True,
            },
        }
        schedule["schedule_id"] = canonical_hash(
            candidate_slide_schedule_identity(schedule), 32
        )
        schedule_path = tmp_path / f"{split}-schedule.json"
        schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
        slide = {
            "status": "subwindow_clustered_time_slide_integration_only",
            "split": split,
            "manifest_path": str(background),
            "manifest_sha256": background_sha,
            "background_manifest_sha256": background_sha,
            "equivalent_live_time_years": 10.0,
            "slide_count": 1,
            "slide_exposure": [{"slide_index": 1, "live_time_seconds": 315576000.0}],
            "execution_schedule_complete": True,
            "slide_schedule_path": str(schedule_path),
            "slide_schedule_sha256": file_sha256(schedule_path),
            "slide_schedule_id": schedule["schedule_id"],
            "slide_schedule_count": 1,
            "input_gps_blocks": [block],
            "reference_ifo": "H1",
            "shifted_ifo": "L1",
            "publication_timing_gate_passed": True,
            **identity,
        }
        slide_path = tmp_path / f"{split}-slide.json"
        slide_path.write_text(json.dumps(slide), encoding="utf-8")
        injections_path = tmp_path / f"{split}-injections.jsonl"
        injection_rows = [
            {
                "split": split,
                "injection_id": f"{split}-i{index}",
                "waveform_id": f"{split}-w{index}",
                "gps_block": f"{block}-injection",
                "source_family": "bbh",
                "stratum": "bbh",
                "vt_weight": index + 1.0,
                "ranking_score": score,
            }
            for index, score in enumerate(injection_scores)
        ]
        injections_path.write_text(
            "".join(json.dumps(row) + "\n" for row in injection_rows), encoding="utf-8"
        )
        injection_report = {
            "status": "physical_network_injection_candidate_rankings",
            "split": split,
            "manifest_path": str(injections_path),
            "manifest_sha256": file_sha256(injections_path),
            "reference_ifo": "H1",
            "second_ifo": "L1",
            "timing_calibration_consistent": True,
            "candidate_scoring_provenance_consistent": True,
            **identity,
        }
        injection_report_path = tmp_path / f"{split}-injection-report.json"
        injection_report_path.write_text(json.dumps(injection_report), encoding="utf-8")
        return slide_path, injection_report_path

    val_slide, val_injections = artifacts("val", 0.8, [0.9, 0.2], "val-block")
    calibration_path = tmp_path / "calibration.json"
    calibration = run_candidate_search_calibration(
        val_slide,
        val_injections,
        target_far_per_year=0.1,
        output=calibration_path,
        bootstrap_replicates=20,
        seed=1,
    )
    assert calibration["calibration"]["threshold"] == 0.8
    assert calibration["publication_calibration_eligible"] is True
    test_slide, test_injections = artifacts("test", 0.85, [0.9, 0.1], "test-block")
    result = run_frozen_candidate_search_evaluation(
        calibration_path,
        test_slide,
        test_injections,
        tmp_path / "locked.json",
        minimum_test_live_time_years=5.0,
        minimum_test_injections=2,
        bootstrap_replicates=20,
        seed=2,
    )
    assert result["candidate_endpoint_gates_passed"] is True
    assert result["test_evaluation"]["background"]["far_per_year"] == 0.1
    assert result["test_evaluation"]["injections"]["recovered"] == 1


def test_candidate_search_calibration_accepts_frozen_block_permutations(
    tmp_path,
) -> None:
    source_background = tmp_path / "source-background.jsonl"
    source_rows = []
    for block_index in range(3):
        block_start = 1000 + block_index * 256
        for slot in range(2):
            source_rows.append(
                {
                    "window_id": f"w-{block_index}-{slot}",
                    "split": "val",
                    "gps_start": block_start + slot * 8,
                    "gps_end": block_start + (slot + 1) * 8,
                    "gps_block": f"gps:{block_start}:256",
                    "ifos": ["H1", "L1"],
                }
            )
    source_background.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    schedule_path = tmp_path / "block-schedule.json"
    schedule = freeze_candidate_block_permutation_schedule(
        source_background,
        schedule_path,
        "val",
        "H1",
        "L1",
        target_far_per_year=1_000_000,
        maximum_shifts=2,
    )
    background = tmp_path / "candidate-background.jsonl"
    background.write_text(
        json.dumps({"split": "val", "ranking_score": 0.8}) + "\n",
        encoding="utf-8",
    )
    identity = {
        "candidate_checkpoint_sha256": "a" * 64,
        "candidate_config_sha256": "b" * 64,
        "candidate_code_commit": "deadbee",
        "timing_calibration_report_sha256": "c" * 64,
        "physical_delay_limit_seconds": 0.010,
        "empirical_timing_uncertainty_seconds": 0.001,
    }
    slide_report = {
        "status": "subwindow_clustered_time_slide_integration_only",
        "split": "val",
        "manifest_path": str(background),
        "manifest_sha256": file_sha256(background),
        "background_manifest_sha256": file_sha256(source_background),
        "background_pairing_method": schedule["method"],
        "equivalent_live_time_years": schedule["selected_equivalent_live_time_years"],
        "slide_count": 2,
        "slide_indices": schedule["shift_indices"],
        "slide_indices_sha256": schedule["shift_indices_sha256"],
        "slide_exposure": [
            {**row, "slide_index": row["shift_index"]}
            for row in schedule["selected_shifts"]
        ],
        "execution_schedule_complete": True,
        "slide_schedule_path": str(schedule_path),
        "slide_schedule_sha256": file_sha256(schedule_path),
        "slide_schedule_id": schedule["schedule_id"],
        "slide_schedule_count": 2,
        "input_gps_blocks": schedule["ordered_gps_blocks"],
        "reference_ifo": "H1",
        "shifted_ifo": "L1",
        "publication_timing_gate_passed": True,
        **identity,
    }
    slide_path = tmp_path / "block-slide-report.json"
    slide_path.write_text(json.dumps(slide_report), encoding="utf-8")
    injections = tmp_path / "injections.jsonl"
    injections.write_text(
        json.dumps(
            {
                "split": "val",
                "injection_id": "i1",
                "waveform_id": "wave1",
                "gps_block": "gps:5000:256",
                "source_family": "BBH",
                "vt_weight": 1.0,
                "ranking_score": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    injection_report = tmp_path / "injection-report.json"
    injection_report.write_text(
        json.dumps(
            {
                "status": "physical_network_injection_candidate_rankings",
                "split": "val",
                "manifest_path": str(injections),
                "manifest_sha256": file_sha256(injections),
                "reference_ifo": "H1",
                "second_ifo": "L1",
                "timing_calibration_consistent": True,
                "candidate_scoring_provenance_consistent": True,
                **identity,
            }
        ),
        encoding="utf-8",
    )
    result = run_candidate_search_calibration(
        slide_path,
        injection_report,
        target_far_per_year=1_000_000,
        output=tmp_path / "calibration.json",
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["publication_calibration_eligible"] is True
    assert result["selection_data"] == ("validation_candidate_block_permutations_only")
    assert result["slide_schedule_audit"]["schedule_kind"] == ("gps_block_permutation")
    assert result["slide_schedule_audit"]["passed"] is True


def test_locked_candidate_search_rejects_engineering_calibration_without_schedule(
    tmp_path,
) -> None:
    background = tmp_path / "background.jsonl"
    background.write_text(json.dumps({"split": "val", "ranking_score": 0.8}) + "\n")
    identity = {
        "candidate_checkpoint_sha256": "a" * 64,
        "candidate_config_sha256": "b" * 64,
        "candidate_code_commit": "deadbee",
        "timing_calibration_report_sha256": "c" * 64,
        "physical_delay_limit_seconds": 0.01,
        "empirical_timing_uncertainty_seconds": 0.001,
    }
    slide = tmp_path / "slide.json"
    slide.write_text(
        json.dumps(
            {
                "status": "subwindow_clustered_time_slide_integration_only",
                "split": "val",
                "manifest_path": str(background),
                "manifest_sha256": file_sha256(background),
                "equivalent_live_time_years": 10.0,
                "input_gps_blocks": ["val-background"],
                "reference_ifo": "H1",
                "shifted_ifo": "L1",
                "publication_timing_gate_passed": True,
                **identity,
            }
        )
    )
    injections = tmp_path / "injections.jsonl"
    injections.write_text(
        json.dumps(
            {
                "split": "val",
                "injection_id": "i",
                "waveform_id": "w",
                "gps_block": "injection-block",
                "source_family": "BBH",
                "vt_weight": 1.0,
                "ranking_score": 0.9,
            }
        )
        + "\n"
    )
    injection_report = tmp_path / "injection-report.json"
    injection_report.write_text(
        json.dumps(
            {
                "status": "physical_network_injection_candidate_rankings",
                "split": "val",
                "manifest_path": str(injections),
                "manifest_sha256": file_sha256(injections),
                "reference_ifo": "H1",
                "second_ifo": "L1",
                "timing_calibration_consistent": True,
                "candidate_scoring_provenance_consistent": True,
                **identity,
            }
        )
    )
    calibration_path = tmp_path / "engineering-calibration.json"
    calibration = run_candidate_search_calibration(
        slide,
        injection_report,
        target_far_per_year=0.1,
        output=calibration_path,
        bootstrap_replicates=20,
        seed=1,
    )
    assert calibration["publication_calibration_eligible"] is False
    with pytest.raises(ValueError, match="target-exposure frozen schedule"):
        run_frozen_candidate_search_evaluation(
            calibration_path,
            slide,
            injection_report,
            tmp_path / "must-not-exist.json",
            minimum_test_live_time_years=1.0,
            minimum_test_injections=1,
            bootstrap_replicates=20,
            seed=2,
        )


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
