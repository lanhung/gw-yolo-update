from __future__ import annotations

import json
import math
from itertools import combinations
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .background import SECONDS_PER_YEAR, _union_duration
from .exposure import (
    CANDIDATE_BLOCK_PERMUTATION_METHOD,
    CANDIDATE_BLOCK_SELECTION_DATA,
    candidate_block_schedule_identity,
    candidate_slide_schedule_identity,
)
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .metrics import wilson_interval
from .runtime import execution_provenance


def far_upper_limit_zero_count(live_time_years: float, confidence: float = 0.90) -> float:
    """Poisson upper limit on FAR when zero background events survive."""
    if live_time_years <= 0:
        raise ValueError("live_time_years must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    return -math.log(1.0 - confidence) / live_time_years


def calibrate_threshold(
    background_scores: Iterable[float], live_time_years: float, target_far_per_year: float
) -> dict[str, Any]:
    """Choose the most permissive validation threshold satisfying a target FAR."""
    if live_time_years <= 0:
        raise ValueError("live_time_years must be positive")
    if target_far_per_year < 0:
        raise ValueError("target_far_per_year must be non-negative")
    scores = sorted((float(score) for score in background_scores), reverse=True)
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("background scores must be finite")
    zero_count_threshold = (
        math.nextafter(scores[0], math.inf)
        if scores
        else float(np.finfo(np.float64).max)
    )
    candidates = [zero_count_threshold, *sorted(set(scores), reverse=True)]
    allowed: list[tuple[float, int, float]] = []
    for threshold in candidates:
        count = sum(score >= threshold for score in scores)
        far = count / live_time_years
        if far <= target_far_per_year:
            allowed.append((threshold, count, far))
    threshold, count, far = min(allowed, key=lambda row: row[0])
    return {
        "threshold": threshold,
        "background_count": count,
        "live_time_years": live_time_years,
        "far_per_year": far,
        "ifar_years": 1.0 / far if far > 0 else None,
        "far_90_upper_limit_if_zero": (
            far_upper_limit_zero_count(live_time_years) if count == 0 else None
        ),
        "target_far_per_year": target_far_per_year,
    }


def calibrate_validation_count(
    background_scores: Iterable[float], maximum_false_alarms: int
) -> dict[str, Any]:
    """Freeze a measurable validation-only threshold without inventing long FAR exposure."""
    if maximum_false_alarms < 0:
        raise ValueError("maximum_false_alarms must be non-negative")
    scores = sorted((float(score) for score in background_scores), reverse=True)
    if not scores:
        raise ValueError("background scores cannot be empty")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("background scores must be finite")
    candidates = [math.nextafter(scores[0], math.inf), *sorted(set(scores), reverse=True)]
    allowed = [
        (threshold, sum(score >= threshold for score in scores))
        for threshold in candidates
        if sum(score >= threshold for score in scores) <= maximum_false_alarms
    ]
    threshold, count = min(allowed, key=lambda item: item[0])
    return {
        "threshold": threshold,
        "background_count": count,
        "background_windows": len(scores),
        "maximum_validation_false_alarms": maximum_false_alarms,
        "empirical_survival_fraction": count / len(scores),
        "selection_data": "validation_background_only",
    }


def evaluate_search(
    threshold: float,
    background_scores: Iterable[float],
    background_live_time_years: float,
    injections: Iterable[dict[str, Any]],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260719,
) -> dict[str, Any]:
    """Evaluate a frozen threshold on background and importance-weighted injections."""
    if background_live_time_years <= 0:
        raise ValueError("background_live_time_years must be positive")
    background = [float(score) for score in background_scores]
    rows = list(injections)
    if not rows:
        raise ValueError("at least one injection is required")
    false_alarms = sum(score >= threshold for score in background)
    far = false_alarms / background_live_time_years
    recovered = [float(row["ranking_score"]) >= threshold for row in rows]
    weights = [float(row.get("vt_weight", row.get("weight", 1.0))) for row in rows]
    if any(weight < 0 for weight in weights):
        raise ValueError("injection weights must be non-negative")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("sum of injection weights must be positive")
    recovered_weight = sum(weight for weight, hit in zip(weights, recovered) if hit)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_efficiencies = []
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    recovered_array = np.asarray(recovered, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(rows), size=len(rows))
        sampled_weights = weight_array[indices]
        denominator = float(sampled_weights.sum())
        if denominator > 0:
            bootstrap_efficiencies.append(
                float((sampled_weights * recovered_array[indices]).sum() / denominator)
            )
    efficiency_interval = wilson_interval(sum(recovered), len(rows))
    return {
        "threshold": threshold,
        "background": {
            "triggers": len(background),
            "false_alarms": false_alarms,
            "live_time_years": background_live_time_years,
            "far_per_year": far,
            "ifar_years": 1.0 / far if far > 0 else None,
            "far_90_upper_limit_if_zero": (
                far_upper_limit_zero_count(background_live_time_years) if false_alarms == 0 else None
            ),
        },
        "injections": {
            "total": len(rows),
            "recovered": sum(recovered),
            "efficiency": sum(recovered) / len(rows),
            "efficiency_wilson_95": list(efficiency_interval),
            "total_vt_weight": total_weight,
            "recovered_vt": recovered_weight,
            "weighted_efficiency": recovered_weight / total_weight,
            "weighted_efficiency_bootstrap_95": [
                float(np.percentile(bootstrap_efficiencies, 2.5)),
                float(np.percentile(bootstrap_efficiencies, 97.5)),
            ],
            "bootstrap_replicates": bootstrap_replicates,
        },
    }


def summarize_injection_efficiency(
    injections: Iterable[dict[str, Any]],
    threshold: float,
    score_field: str = "ranking_score",
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    rows = list(injections)
    if not rows:
        raise ValueError("at least one injection is required")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    scores = np.asarray([float(row[score_field]) for row in rows], dtype=np.float64)
    weights = np.asarray(
        [float(row.get("vt_weight", row.get("weight", 1.0))) for row in rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(scores)) or np.any(~np.isfinite(weights)):
        raise ValueError("injection scores and weights must be finite")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("injection weights must be non-negative with positive sum")
    recovered = scores >= threshold
    recovered_count = int(recovered.sum())
    recovered_weight = float(weights[recovered].sum())
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(rows), size=len(rows))
        denominator = float(weights[indices].sum())
        if denominator > 0:
            bootstrap.append(
                float((weights[indices] * recovered[indices]).sum() / denominator)
            )
    return {
        "injections": len(rows),
        "recovered": recovered_count,
        "efficiency": recovered_count / len(rows),
        "efficiency_wilson_95": list(wilson_interval(recovered_count, len(rows))),
        "total_vt_weight": float(weights.sum()),
        "recovered_vt": recovered_weight,
        "weighted_efficiency": recovered_weight / float(weights.sum()),
        "weighted_efficiency_bootstrap_95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "bootstrap_replicates": bootstrap_replicates,
    }


def paired_vt_comparison(
    injections: Iterable[dict[str, Any]],
    threshold_a: float,
    threshold_b: float,
    score_field_a: str,
    score_field_b: str,
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    rows = list(injections)
    if not rows:
        raise ValueError("at least one injection is required")
    weights = np.asarray(
        [float(row.get("vt_weight", row.get("weight", 1.0))) for row in rows],
        dtype=np.float64,
    )
    recovered_a = np.asarray(
        [float(row[score_field_a]) >= threshold_a for row in rows], dtype=np.float64
    )
    recovered_b = np.asarray(
        [float(row[score_field_b]) >= threshold_b for row in rows], dtype=np.float64
    )
    contributions = weights * (recovered_b - recovered_a)
    delta = float(contributions.sum())
    recovered_vt_a = float((weights * recovered_a).sum())
    recovered_vt_b = float((weights * recovered_b).sum())
    rng = np.random.default_rng(seed)
    bootstrap_delta = []
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(rows), size=len(rows))
        bootstrap_delta.append(float(contributions[indices].sum()))
    strata = {}
    for stratum in sorted({str(row.get("stratum", "all")) for row in rows}):
        indices = [index for index, row in enumerate(rows) if str(row.get("stratum", "all")) == stratum]
        stratum_weight = float(weights[indices].sum())
        strata[stratum] = {
            "injections": len(indices),
            "total_weight": stratum_weight,
            "weighted_efficiency_a": (
                float((weights[indices] * recovered_a[indices]).sum() / stratum_weight)
                if stratum_weight
                else None
            ),
            "weighted_efficiency_b": (
                float((weights[indices] * recovered_b[indices]).sum() / stratum_weight)
                if stratum_weight
                else None
            ),
        }
    return {
        "method_a": {"score_field": score_field_a, "threshold": threshold_a, "recovered_vt": recovered_vt_a},
        "method_b": {"score_field": score_field_b, "threshold": threshold_b, "recovered_vt": recovered_vt_b},
        "delta_recovered_vt_b_minus_a": delta,
        "relative_delta": delta / recovered_vt_a if recovered_vt_a > 0 else None,
        "paired_bootstrap_95": [
            float(np.percentile(bootstrap_delta, 2.5)),
            float(np.percentile(bootstrap_delta, 97.5)),
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "strata": strata,
    }


def compare_search_methods(
    validation_background: Iterable[dict[str, Any]],
    test_background: Iterable[dict[str, Any]],
    test_injections: Iterable[dict[str, Any]],
    validation_live_time_years: float,
    test_live_time_years: float,
    target_far_per_year: float,
    score_field_a: str,
    score_field_b: str,
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    validation_rows = list(validation_background)
    background_rows = list(test_background)
    injection_rows = list(test_injections)
    calibrations = {
        field: calibrate_threshold(
            (row[field] for row in validation_rows), validation_live_time_years, target_far_per_year
        )
        for field in (score_field_a, score_field_b)
    }

    def with_ranking_score(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
        return [{**row, "ranking_score": row[field]} for row in rows]

    evaluations = {
        field: evaluate_search(
            calibrations[field]["threshold"],
            (row[field] for row in background_rows),
            test_live_time_years,
            with_ranking_score(injection_rows, field),
            bootstrap_replicates,
            seed,
        )
        for field in (score_field_a, score_field_b)
    }
    paired = paired_vt_comparison(
        injection_rows,
        calibrations[score_field_a]["threshold"],
        calibrations[score_field_b]["threshold"],
        score_field_a,
        score_field_b,
        bootstrap_replicates,
        seed,
    )
    return {
        "protocol": "method-specific validation thresholds at common target FAR; paired frozen test",
        "target_far_per_year": target_far_per_year,
        "calibrations": calibrations,
        "test_evaluations": evaluations,
        "paired_vt": paired,
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _verified_candidate_search_artifact(
    report_path: str | Path, expected_status: str, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with Path(report_path).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != expected_status or str(report.get("split")) != split:
        raise ValueError(f"candidate search report is not the expected {split} artifact")
    manifest = report.get("manifest_path")
    expected_sha = report.get("manifest_sha256")
    if not manifest or not expected_sha or file_sha256(manifest) != str(expected_sha):
        raise ValueError("candidate search manifest hash mismatch")
    rows = load_jsonl(manifest)
    if any(str(row.get("split")) != split for row in rows):
        raise ValueError(f"candidate search manifest contains non-{split} rows")
    return report, rows


def _audit_candidate_slide_schedule(
    slide_report: dict[str, Any], target_far_per_year: float
) -> dict[str, Any]:
    """Verify that a score-blind frozen schedule was executed in full at target exposure."""

    schedule_value = slide_report.get("slide_schedule_path")
    if schedule_value is None:
        return {
            "frozen_schedule_present": False,
            "schedule_identity_verified": False,
            "execution_schedule_complete": False,
            "target_far_matches": False,
            "target_exposure_reached": False,
            "equivalent_exposure_matches": False,
            "passed": False,
        }
    schedule_path = Path(str(schedule_value)).resolve()
    if not schedule_path.is_file():
        raise ValueError("candidate time-slide frozen schedule is missing")
    expected_sha = slide_report.get("slide_schedule_sha256")
    if not expected_sha or file_sha256(schedule_path) != str(expected_sha):
        raise ValueError("candidate time-slide frozen schedule hash mismatch")
    with schedule_path.open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    schedule_status = str(schedule.get("status"))
    if schedule_status == "frozen_candidate_time_slide_schedule":
        schedule_kind = "absolute_time_slide"
        indices = [int(value) for value in schedule.get("slide_indices", [])]
        count_matches = int(schedule.get("slide_count", -1)) == len(indices)
        indices_hash_matches = canonical_hash(indices, 64) == schedule.get(
            "slide_indices_sha256"
        )
        identity_hash_matches = canonical_hash(
            candidate_slide_schedule_identity(schedule), 32
        ) == schedule.get("schedule_id")
        selection_matches = (
            schedule.get("selection_data")
            == "background_gps_and_detector_availability_only"
        )
        exposure = schedule.get("exposure_plan", {})
        schedule_years = float(exposure.get("equivalent_live_time_years", -1))
        target_reached = bool(
            schedule.get("schedule_exposure_target_reached") is True
            and exposure.get("target_zero_count_upper_reached") is True
        )
        pairing_contract_matches = slide_report.get("background_pairing_method") in (
            None,
            "absolute_gps_time_slide_v1",
        )
    elif schedule_status == "frozen_candidate_block_permutation_schedule":
        schedule_kind = "gps_block_permutation"
        indices = [int(value) for value in schedule.get("shift_indices", [])]
        count_matches = int(schedule.get("selected_shift_count", -1)) == len(indices)
        indices_hash_matches = canonical_hash(indices, 64) == schedule.get(
            "shift_indices_sha256"
        ) and slide_report.get("slide_indices_sha256") == schedule.get(
            "shift_indices_sha256"
        )
        identity_hash_matches = canonical_hash(
            candidate_block_schedule_identity(schedule), 32
        ) == schedule.get("schedule_id")
        selection_matches = (
            schedule.get("selection_data") == CANDIDATE_BLOCK_SELECTION_DATA
        )
        schedule_years = float(schedule.get("selected_equivalent_live_time_years", -1))
        target_reached = schedule.get("schedule_exposure_target_reached") is True
        pairing_contract_matches = bool(
            schedule.get("method") == CANDIDATE_BLOCK_PERMUTATION_METHOD
            and slide_report.get("background_pairing_method") == schedule.get("method")
            and slide_report.get("input_gps_blocks")
            == schedule.get("ordered_gps_blocks")
        )
    else:
        raise ValueError("candidate background schedule has an unsupported status")
    identity_verified = bool(
        schedule.get("candidate_scores_inspected") is False
        and selection_matches
        and schedule.get("schedule_id") == slide_report.get("slide_schedule_id")
        and count_matches
        and int(slide_report.get("slide_schedule_count", -1)) == len(indices)
        and indices_hash_matches
        and identity_hash_matches
        and schedule.get("background_manifest_sha256")
        == slide_report.get("background_manifest_sha256")
        and str(schedule.get("split")) == str(slide_report.get("split"))
        and str(schedule.get("reference_ifo")) == str(slide_report.get("reference_ifo"))
        and str(schedule.get("shifted_ifo")) == str(slide_report.get("shifted_ifo"))
        and pairing_contract_matches
        and indices == sorted(set(indices))
        and all(value > 0 for value in indices)
    )
    if not identity_verified:
        raise ValueError("candidate time-slide frozen schedule identity mismatch")
    observed_indices = sorted(
        int(row["slide_index"]) for row in slide_report.get("slide_exposure", [])
    )
    execution_complete = bool(
        observed_indices == indices
        and (
            slide_report.get("execution_schedule_complete") is True
            or (
                slide_report.get("execution_schedule_complete") is None
                and int(slide_report.get("slide_count", -1)) == len(indices)
            )
        )
    )
    target_matches = bool(
        np.isclose(
            float(schedule.get("target_far_per_year", -1)),
            target_far_per_year,
            rtol=0.0,
            atol=1e-12,
        )
    )
    report_years = float(slide_report.get("equivalent_live_time_years", -2))
    exposure_matches = bool(
        schedule_years >= 0
        and np.isclose(schedule_years, report_years, rtol=0.0, atol=1e-12)
    )
    gates = {
        "frozen_schedule_present": True,
        "schedule_identity_verified": identity_verified,
        "execution_schedule_complete": execution_complete,
        "target_far_matches": target_matches,
        "target_exposure_reached": target_reached,
        "equivalent_exposure_matches": exposure_matches,
    }
    return {
        **gates,
        "passed": all(gates.values()),
        "schedule_path": str(schedule_path),
        "schedule_sha256": file_sha256(schedule_path),
        "schedule_id": schedule["schedule_id"],
        "schedule_kind": schedule_kind,
        "schedule_target_far_per_year": schedule["target_far_per_year"],
        "schedule_equivalent_live_time_years": schedule_years,
    }


def _candidate_search_identity(
    slide_report: dict[str, Any], injection_report: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "candidate_checkpoint_sha256",
        "candidate_config_sha256",
        "candidate_code_commit",
        "timing_calibration_report_sha256",
        "physical_delay_limit_seconds",
        "empirical_timing_uncertainty_seconds",
    )
    mismatches = [
        field
        for field in fields
        if slide_report.get(field) is None
        or injection_report.get(field) is None
        or str(slide_report[field]) != str(injection_report[field])
    ]
    if mismatches:
        raise ValueError(f"background/injection candidate provenance differs: {mismatches}")
    if (
        str(slide_report.get("reference_ifo"))
        != str(injection_report.get("reference_ifo"))
        or str(slide_report.get("shifted_ifo"))
        != str(injection_report.get("second_ifo"))
    ):
        raise ValueError("background/injection detector pair differs")
    if not slide_report.get("publication_timing_gate_passed"):
        raise ValueError("candidate time-slide timing gate did not pass")
    if not injection_report.get("timing_calibration_consistent") or not injection_report.get(
        "candidate_scoring_provenance_consistent"
    ):
        raise ValueError("injection candidate timing/scoring provenance is inconsistent")
    return {
        field: slide_report[field]
        for field in fields
    } | {
        "reference_ifo": slide_report["reference_ifo"],
        "second_ifo": slide_report["shifted_ifo"],
    }


def run_candidate_search_calibration(
    validation_time_slide_report: str | Path,
    validation_injection_ranking_report: str | Path,
    target_far_per_year: float,
    output: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Freeze a candidate-level threshold using validation artifacts only."""

    slide, background = _verified_candidate_search_artifact(
        validation_time_slide_report,
        "subwindow_clustered_time_slide_integration_only",
        "val",
    )
    injections, injection_rows = _verified_candidate_search_artifact(
        validation_injection_ranking_report,
        "physical_network_injection_candidate_rankings",
        "val",
    )
    identity = _candidate_search_identity(slide, injections)
    schedule_audit = _audit_candidate_slide_schedule(slide, target_far_per_year)
    live_time_years = float(slide["equivalent_live_time_years"])
    if live_time_years <= 0:
        raise ValueError("validation candidate background has no equivalent live time")
    calibration = calibrate_threshold(
        (float(row["ranking_score"]) for row in background),
        live_time_years,
        target_far_per_year,
    )
    validation_efficiency = summarize_injection_efficiency(
        injection_rows,
        float(calibration["threshold"]),
        "ranking_score",
        bootstrap_replicates,
        seed,
    )
    result = {
        "status": "frozen_validation_candidate_search_calibration",
        "scientific_claim_allowed": False,
        "selection_data": (
            "validation_candidate_block_permutations_only"
            if schedule_audit.get("schedule_kind") == "gps_block_permutation"
            else "validation_candidate_time_slides_only"
        ),
        "test_evaluation": None,
        "identity": identity,
        "target_far_per_year": target_far_per_year,
        "calibration": calibration,
        "target_far_has_at_least_one_expected_background_count": (
            live_time_years * target_far_per_year >= 1.0
        ),
        "slide_schedule_audit": schedule_audit,
        "publication_calibration_eligible": bool(schedule_audit["passed"]),
        "validation_injection_diagnostic": validation_efficiency,
        "validation_background_gps_blocks": list(slide["input_gps_blocks"]),
        "validation_injection_gps_blocks": sorted(
            {str(row["gps_block"]) for row in injection_rows}
        ),
        "validation_injection_ids_hash": canonical_hash(
            sorted(str(row["injection_id"]) for row in injection_rows), 64
        ),
        "validation_injection_ids": sorted(
            str(row["injection_id"]) for row in injection_rows
        ),
        "validation_waveform_ids_hash": canonical_hash(
            sorted(str(row["waveform_id"]) for row in injection_rows), 64
        ),
        "validation_waveform_ids": sorted(
            str(row["waveform_id"]) for row in injection_rows
        ),
        "validation_time_slide_report_path": str(validation_time_slide_report),
        "validation_time_slide_report_sha256": file_sha256(validation_time_slide_report),
        "validation_injection_ranking_report_path": str(
            validation_injection_ranking_report
        ),
        "validation_injection_ranking_report_sha256": file_sha256(
            validation_injection_ranking_report
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def run_paired_raw_mask_candidate_calibration_comparison(
    raw_calibration_report: str | Path,
    mask_calibration_report: str | Path,
    mask_validation_receipt: str | Path,
    mask_timing_receipt: str | Path,
    output: str | Path,
    minimum_absolute_weighted_efficiency_gain: float = 0.05,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Compare raw/mask validation rankings at independently frozen common-FAR thresholds."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("paired raw/mask calibration comparisons are immutable")
    if not 0 <= minimum_absolute_weighted_efficiency_gain < 1:
        raise ValueError("minimum raw/mask weighted-efficiency gain must be in [0, 1)")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    validation_path = Path(mask_validation_receipt).resolve()
    timing_path = Path(mask_timing_receipt).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    pipeline_identity = validation.get("artifacts", {}).get("pipeline_report", {})
    pipeline_path = Path(str(pipeline_identity.get("path", ""))).resolve()
    if (
        validation.get("status") != "completed_validation_only_mask_deglitch_gate"
        or validation.get("execution_passed") is not True
        or validation.get("development_gates_passed") is not True
        or validation.get("scientific_claim_allowed") is not False
        or validation.get("locked_test_allowed") is not False
        or validation.get("test_rows_read") != 0
        or not pipeline_path.is_file()
        or pipeline_identity.get("sha256") != file_sha256(pipeline_path)
    ):
        raise ValueError("paired raw/mask comparison requires a passing six-arm receipt")
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    six_arm = pipeline.get("comparison", {})
    clean_gate = six_arm.get("gates", {}).get("clean_noninferiority", {})
    contaminated_gate = six_arm.get("gates", {}).get(
        "contaminated_material_gain", {}
    )
    if (
        pipeline.get("status") != "validation_only_end_to_end_mask_search_pipeline"
        or pipeline.get("development_gates_passed") is not True
        or pipeline.get("test_rows_read") != 0
        or pipeline.get("test_evaluation") is not None
        or clean_gate.get("passed") is not True
        or contaminated_gate.get("passed") is not True
    ):
        raise ValueError("six-arm pipeline does not prove clean non-inferiority and mask gain")
    if (
        timing.get("status") != "completed_validation_only_mask_timing_gate"
        or timing.get("coherent_background_scale_allowed") is not True
        or timing.get("raw_timing_gate_passed") is not True
        or timing.get("mask_timing_gate_passed") is not True
        or timing.get("test_rows_read") != 0
        or timing.get("locked_test_allowed") is not False
        or Path(str(timing.get("mask_validation_receipt_path", ""))).resolve()
        != validation_path
        or timing.get("mask_validation_receipt_sha256")
        != file_sha256(validation_path)
    ):
        raise ValueError("paired raw/mask comparison requires a passing timing receipt")

    def load_arm(
        arm: str, calibration_value: str | Path
    ) -> tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        calibration_path = Path(calibration_value).resolve()
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        ranking_path = Path(
            str(calibration.get("validation_injection_ranking_report_path", ""))
        ).resolve()
        slide_path = Path(
            str(calibration.get("validation_time_slide_report_path", ""))
        ).resolve()
        if (
            calibration.get("status")
            != "frozen_validation_candidate_search_calibration"
            or calibration.get("scientific_claim_allowed") is not False
            or calibration.get("test_evaluation") is not None
            or calibration.get("publication_calibration_eligible") is not True
            or calibration.get("slide_schedule_audit", {}).get("passed") is not True
            or not str(calibration.get("selection_data", "")).startswith("validation_")
            or not ranking_path.is_file()
            or calibration.get("validation_injection_ranking_report_sha256")
            != file_sha256(ranking_path)
            or not slide_path.is_file()
            or calibration.get("validation_time_slide_report_sha256")
            != file_sha256(slide_path)
        ):
            raise ValueError(f"{arm} candidate calibration failed replay")
        timing_ranking = timing.get("injection_ranking_reports", {}).get(arm, {})
        if (
            Path(str(timing_ranking.get("path", ""))).resolve() != ranking_path
            or timing_ranking.get("sha256") != file_sha256(ranking_path)
        ):
            raise ValueError(f"{arm} calibration differs from the timing-gate rankings")
        ranking, rows = _verified_candidate_search_artifact(
            ranking_path,
            "physical_network_injection_candidate_rankings",
            "val",
        )
        slide = json.loads(slide_path.read_text(encoding="utf-8"))
        if slide.get("status") != "subwindow_clustered_time_slide_integration_only":
            raise ValueError(f"{arm} calibration uses an invalid block-background report")
        return calibration_path, calibration, ranking, rows, slide

    arms = {
        "raw": load_arm("raw", raw_calibration_report),
        "mask": load_arm("mask", mask_calibration_report),
    }
    raw_path, raw, raw_ranking, raw_rows, raw_slide = arms["raw"]
    mask_path, mask, mask_ranking, mask_rows, mask_slide = arms["mask"]
    if not np.isclose(
        float(raw["target_far_per_year"]),
        float(mask["target_far_per_year"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("raw/mask candidate calibrations use different target FARs")
    common_identity_fields = (
        "candidate_checkpoint_sha256",
        "candidate_config_sha256",
        "candidate_code_commit",
        "physical_delay_limit_seconds",
        "reference_ifo",
        "second_ifo",
    )
    if any(
        raw.get("identity", {}).get(field) != mask.get("identity", {}).get(field)
        for field in common_identity_fields
    ):
        raise ValueError("raw/mask candidate calibrations differ in their model/physics identity")
    common_background_fields = (
        "background_manifest_sha256",
        "background_pairing_method",
        "equivalent_live_time_years",
        "input_gps_blocks",
        "reference_ifo",
        "shifted_ifo",
        "slide_schedule_sha256",
        "slide_schedule_id",
        "slide_count",
    )
    if any(raw_slide.get(field) != mask_slide.get(field) for field in common_background_fields):
        raise ValueError("raw/mask calibrations do not use identical physical background exposure")
    raw_by_id = {str(row["injection_id"]): row for row in raw_rows}
    mask_by_id = {str(row["injection_id"]): row for row in mask_rows}
    if (
        len(raw_by_id) != len(raw_rows)
        or len(mask_by_id) != len(mask_rows)
        or set(raw_by_id) != set(mask_by_id)
    ):
        raise ValueError("raw/mask calibration rankings do not share unique injection IDs")
    joined = []
    identity_fields = (
        "waveform_id",
        "source_family",
        "stratum",
        "gps_block",
        "gps_time",
        "vt_weight",
        "vt_weight_unit",
    )
    for injection_id in sorted(raw_by_id):
        raw_row = raw_by_id[injection_id]
        mask_row = mask_by_id[injection_id]
        if any(raw_row.get(field) != mask_row.get(field) for field in identity_fields):
            raise ValueError(f"raw/mask physical injection identity differs: {injection_id}")
        joined.append(
            {
                **raw_row,
                "raw_score": float(raw_row["ranking_score"]),
                "mask_score": float(mask_row["ranking_score"]),
            }
        )
    raw_threshold = float(raw["calibration"]["threshold"])
    mask_threshold = float(mask["calibration"]["threshold"])
    paired = paired_vt_comparison(
        joined,
        raw_threshold,
        mask_threshold,
        "raw_score",
        "mask_score",
        bootstrap_replicates,
        seed,
    )
    total_vt = float(sum(float(row["vt_weight"]) for row in joined))
    delta_vt = float(paired["delta_recovered_vt_b_minus_a"])
    absolute_gain = delta_vt / total_vt
    gain_gate = {
        "minimum_absolute_weighted_efficiency_gain": (
            minimum_absolute_weighted_efficiency_gain
        ),
        "observed_absolute_weighted_efficiency_gain": absolute_gain,
        "paired_delta_recovered_vt_lower_95": float(paired["paired_bootstrap_95"][0]),
        "passed": bool(
            absolute_gain >= minimum_absolute_weighted_efficiency_gain
            and float(paired["paired_bootstrap_95"][0]) > 0
        ),
    }
    result = {
        "status": "validation_only_paired_raw_mask_candidate_calibration_comparison",
        "passed": gain_gate["passed"],
        "scientific_claim_allowed": False,
        "locked_test_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "protocol": (
            "raw and mask thresholds independently frozen at one validation target FAR; "
            "paired contaminated injections compared without threshold retuning"
        ),
        "target_far_per_year": float(raw["target_far_per_year"]),
        "paired_injections": len(joined),
        "total_vt_weight": total_vt,
        "raw_validation_diagnostic": raw["validation_injection_diagnostic"],
        "mask_validation_diagnostic": mask["validation_injection_diagnostic"],
        "paired_vt": paired,
        "six_arm_clean_noninferiority_gate": clean_gate,
        "six_arm_contaminated_gain_gate": contaminated_gate,
        "continuous_background_mask_gain_gate": gain_gate,
        "mask_locked_test_arm_eligible": gain_gate["passed"],
        "locked_test_prerequisites_satisfied": False,
        "raw_calibration_report": {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
        },
        "mask_calibration_report": {
            "path": str(mask_path),
            "sha256": file_sha256(mask_path),
        },
        "mask_validation_receipt": {
            "path": str(validation_path),
            "sha256": file_sha256(validation_path),
        },
        "mask_timing_receipt": {
            "path": str(timing_path),
            "sha256": file_sha256(timing_path),
        },
        "raw_timing_calibration_report_sha256": raw_ranking[
            "timing_calibration_report_sha256"
        ],
        "mask_timing_calibration_report_sha256": mask_ranking[
            "timing_calibration_report_sha256"
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def run_frozen_candidate_search_evaluation(
    calibration_report: str | Path,
    test_time_slide_report: str | Path,
    test_injection_ranking_report: str | Path,
    output: str | Path,
    minimum_test_live_time_years: float,
    minimum_test_injections: int,
    bootstrap_replicates: int = 10000,
    seed: int = 20260721,
) -> dict[str, Any]:
    """Apply a frozen candidate threshold once to disjoint locked-test artifacts."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("frozen candidate search output already exists")
    if minimum_test_live_time_years <= 0 or minimum_test_injections <= 0:
        raise ValueError("locked candidate endpoint minima must be positive")
    with Path(calibration_report).open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen.get("status") != "frozen_validation_candidate_search_calibration":
        raise ValueError("candidate search calibration artifact has the wrong status")
    if frozen.get("test_evaluation") is not None:
        raise ValueError("candidate search calibration already contains test information")
    if frozen.get("publication_calibration_eligible") is not True:
        raise ValueError(
            "candidate search calibration lacks a complete target-exposure frozen schedule"
        )
    slide, background = _verified_candidate_search_artifact(
        test_time_slide_report,
        "subwindow_clustered_time_slide_integration_only",
        "test",
    )
    injections, injection_rows = _verified_candidate_search_artifact(
        test_injection_ranking_report,
        "physical_network_injection_candidate_rankings",
        "test",
    )
    identity = _candidate_search_identity(slide, injections)
    if identity != frozen.get("identity"):
        raise ValueError("locked test candidate identity differs from frozen validation identity")
    test_schedule_audit = _audit_candidate_slide_schedule(
        slide, float(frozen["target_far_per_year"])
    )
    if not test_schedule_audit["passed"]:
        raise ValueError(
            "locked candidate test lacks a complete target-exposure frozen schedule"
        )
    background_overlap = sorted(
        set(str(value) for value in frozen["validation_background_gps_blocks"])
        & set(str(value) for value in slide["input_gps_blocks"])
    )
    injection_blocks = {str(row["gps_block"]) for row in injection_rows}
    injection_overlap = sorted(
        set(str(value) for value in frozen["validation_injection_gps_blocks"])
        & injection_blocks
    )
    injection_id_overlap = sorted(
        set(str(value) for value in frozen["validation_injection_ids"])
        & {str(row["injection_id"]) for row in injection_rows}
    )
    waveform_id_overlap = sorted(
        set(str(value) for value in frozen["validation_waveform_ids"])
        & {str(row["waveform_id"]) for row in injection_rows}
    )
    if background_overlap or injection_overlap or injection_id_overlap or waveform_id_overlap:
        raise ValueError(
            "locked candidate test overlaps validation physical groups: "
            f"background={background_overlap[:5]}, injection_blocks={injection_overlap[:5]}, "
            f"injection_ids={injection_id_overlap[:5]}, waveform_ids={waveform_id_overlap[:5]}"
        )
    live_time_years = float(slide["equivalent_live_time_years"])
    evaluation = evaluate_search(
        float(frozen["calibration"]["threshold"]),
        (float(row["ranking_score"]) for row in background),
        live_time_years,
        injection_rows,
        bootstrap_replicates,
        seed,
    )
    endpoint_gates = {
        "minimum_test_live_time": live_time_years >= minimum_test_live_time_years,
        "minimum_test_injections": len(injection_rows) >= minimum_test_injections,
        "zero_cross_split_background_gps_blocks": not background_overlap,
        "zero_cross_split_injection_gps_blocks": not injection_overlap,
        "zero_cross_split_injection_ids": not injection_id_overlap,
        "zero_cross_split_waveform_ids": not waveform_id_overlap,
        "frozen_candidate_identity": True,
        "publication_timing_gate": bool(slide["publication_timing_gate_passed"]),
        "frozen_test_slide_schedule": bool(test_schedule_audit["passed"]),
    }
    result = {
        "status": "locked_candidate_search_evaluation",
        "candidate_endpoint_gates_passed": all(endpoint_gates.values()),
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "candidate endpoint gates are necessary but final claims additionally require the "
            "predeclared five-seed model comparison and O4b/GWTC-5 one-time access-log gate"
        ),
        "calibration_report_path": str(calibration_report),
        "calibration_report_sha256": file_sha256(calibration_report),
        "test_time_slide_report_path": str(test_time_slide_report),
        "test_time_slide_report_sha256": file_sha256(test_time_slide_report),
        "test_injection_ranking_report_path": str(test_injection_ranking_report),
        "test_injection_ranking_report_sha256": file_sha256(
            test_injection_ranking_report
        ),
        "identity": identity,
        "threshold_source": "frozen_validation_candidate_search_calibration",
        "target_far_per_year": frozen["target_far_per_year"],
        "test_slide_schedule_audit": test_schedule_audit,
        "endpoint_gates": endpoint_gates,
        "minimum_test_live_time_years": minimum_test_live_time_years,
        "minimum_test_injections": minimum_test_injections,
        "test_evaluation": evaluation,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def run_search_benchmark(
    validation_background: str | Path,
    test_background: str | Path,
    test_injections: str | Path,
    validation_live_time_years: float,
    test_live_time_years: float,
    target_far_per_year: float,
    output: str | Path,
) -> dict[str, Any]:
    validation_rows = load_jsonl(validation_background)
    test_background_rows = load_jsonl(test_background)
    injection_rows = load_jsonl(test_injections)
    calibration = calibrate_threshold(
        (row["ranking_score"] for row in validation_rows),
        validation_live_time_years,
        target_far_per_year,
    )
    evaluation = evaluate_search(
        calibration["threshold"],
        (row["ranking_score"] for row in test_background_rows),
        test_live_time_years,
        injection_rows,
    )
    result = {
        "protocol": "threshold calibrated on validation background and frozen on test",
        "calibration": calibration,
        "test": evaluation,
    }
    atomic_write_json(output, result)
    return result


def run_search_calibration(
    validation_background: str | Path,
    validation_live_time_years: float,
    target_far_per_year: float,
    score_field: str,
    output: str | Path,
) -> dict[str, Any]:
    rows = load_jsonl(validation_background)
    if not rows:
        raise ValueError("Validation background cannot be empty")
    missing = [index for index, row in enumerate(rows) if score_field not in row]
    if missing:
        raise ValueError(f"Validation background lacks {score_field!r} at rows {missing[:10]}")
    calibration = calibrate_threshold(
        (row[score_field] for row in rows),
        validation_live_time_years,
        target_far_per_year,
    )
    result = {
        "status": "validation_only_threshold_frozen",
        "protocol": "threshold selected exclusively from validation background",
        "score_field": score_field,
        "validation_background_path": str(validation_background),
        "validation_background_sha256": file_sha256(validation_background),
        "validation_rows": len(rows),
        "calibration": calibration,
    }
    atomic_write_json(output, result)
    return result


def run_frozen_search_evaluation(
    calibration_report: str | Path,
    test_background: str | Path,
    test_injections: str | Path,
    test_live_time_years: float,
    output: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite locked test evaluation: {output_path}")
    with Path(calibration_report).open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen.get("status") != "validation_only_threshold_frozen":
        raise ValueError("Calibration report is not a frozen validation-only threshold")
    score_field = str(frozen["score_field"])
    background_rows = load_jsonl(test_background)
    injection_rows = load_jsonl(test_injections)
    if not background_rows:
        raise ValueError("Test background cannot be empty")
    for label, rows in (("background", background_rows), ("injection", injection_rows)):
        missing = [index for index, row in enumerate(rows) if score_field not in row]
        if missing:
            raise ValueError(f"Test {label} lacks {score_field!r} at rows {missing[:10]}")
    evaluation = evaluate_search(
        float(frozen["calibration"]["threshold"]),
        (row[score_field] for row in background_rows),
        test_live_time_years,
        [{**row, "ranking_score": row[score_field]} for row in injection_rows],
        bootstrap_replicates,
        seed,
    )
    result = {
        "status": "locked_test_evaluated_with_frozen_validation_threshold",
        "protocol": "test evaluated without validation input or threshold adjustment",
        "score_field": score_field,
        "calibration_report_path": str(calibration_report),
        "calibration_report_sha256": file_sha256(calibration_report),
        "test_background_sha256": file_sha256(test_background),
        "test_injections_sha256": file_sha256(test_injections),
        "test_background_rows": len(background_rows),
        "test_injection_rows": len(injection_rows),
        "bootstrap_seed": seed,
        "evaluation": evaluation,
    }
    atomic_write_json(output_path, result)
    return result


def run_validation_injection_diagnostic(
    calibration_report: str | Path,
    validation_injections: str | Path,
    output: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    with Path(calibration_report).open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen.get("status") != "validation_only_threshold_frozen":
        raise ValueError("Calibration report is not a frozen validation-only threshold")
    rows = load_jsonl(validation_injections)
    if not rows:
        raise ValueError("Validation injection manifest cannot be empty")
    invalid_splits = sorted({str(row.get("split")) for row in rows if row.get("split") != "val"})
    if invalid_splits:
        raise ValueError(f"Validation diagnostic received non-val splits: {invalid_splits}")
    score_field = str(frozen["score_field"])
    threshold = float(frozen["calibration"]["threshold"])
    overall = summarize_injection_efficiency(
        rows, threshold, score_field, bootstrap_replicates, seed
    )
    stratum_field = "source_family" if all("source_family" in row for row in rows) else "stratum"
    strata = {}
    for index, stratum in enumerate(sorted({str(row.get(stratum_field, "all")) for row in rows})):
        selected = [row for row in rows if str(row.get(stratum_field, "all")) == stratum]
        strata[stratum] = summarize_injection_efficiency(
            selected,
            threshold,
            score_field,
            bootstrap_replicates,
            seed + index + 1,
        )
    result = {
        "status": "validation_only_physical_injection_diagnostic",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "validation injections may guide development; publication sensitivity requires "
            "the independently locked test corpus and adequate background exposure"
        ),
        "score_field": score_field,
        "threshold": threshold,
        "calibration_report_path": str(calibration_report),
        "calibration_report_sha256": file_sha256(calibration_report),
        "validation_injections_path": str(validation_injections),
        "validation_injections_sha256": file_sha256(validation_injections),
        "bootstrap_seed": seed,
        "overall": overall,
        "strata": strata,
    }
    atomic_write_json(output, result)
    return result


def _verified_score_artifact(
    report_path: str | Path,
    report_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(report_path)
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    count_field = "failed_windows" if report_kind == "background" else "failed_injections"
    rows_field = "scored_windows" if report_kind == "background" else "scored_injections"
    if int(report.get(count_field, -1)) != 0:
        raise ValueError(f"{report_kind} score report contains failures")
    required = (
        "checkpoint_sha256",
        "config_sha256",
        "code_commit",
        "exact_command",
        "environment",
        "triggers_path",
        "triggers_sha256",
    )
    missing = [field for field in required if not report.get(field)]
    if missing:
        raise ValueError(f"{report_kind} score report lacks provenance: {missing}")
    triggers_path = Path(report["triggers_path"])
    if file_sha256(triggers_path) != report["triggers_sha256"]:
        raise ValueError(f"{report_kind} trigger artifact hash mismatch")
    rows = load_jsonl(triggers_path)
    if len(rows) != int(report.get(rows_field, -1)):
        raise ValueError(f"{report_kind} trigger row count differs from score report")
    return report, rows


def run_physical_validation_endpoint(
    training_report: str | Path,
    background_score_report: str | Path,
    injection_score_report: str | Path,
    maximum_validation_false_alarms: int,
    output: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    """Evaluate one checkpoint at a frozen, exposure-limited O4a validation endpoint."""
    with Path(training_report).open("r", encoding="utf-8") as handle:
        training = json.load(handle)
    required_training = (
        "checkpoint_sha256",
        "checkpoint_path",
        "code_commit",
        "config_path",
        "seed",
        "train_manifest_sha256",
        "validation_manifest_sha256",
    )
    missing_training = [field for field in required_training if training.get(field) is None]
    if missing_training:
        raise ValueError(f"Training report lacks provenance: {missing_training}")
    if training.get("test_evaluation") is not None:
        raise ValueError("Physical validation endpoint refuses a training report with test evaluation")
    if file_sha256(training["checkpoint_path"]) != training["checkpoint_sha256"]:
        raise ValueError("Training checkpoint hash differs from training report")
    background_report, background_rows = _verified_score_artifact(
        background_score_report, "background"
    )
    injection_report, injection_rows = _verified_score_artifact(
        injection_score_report, "injection"
    )
    identities = {}
    for field in ("checkpoint_sha256", "config_sha256", "code_commit"):
        values = {str(background_report[field]), str(injection_report[field])}
        if len(values) != 1:
            raise ValueError(f"Background/injection score reports disagree on {field}")
        identities[field] = next(iter(values))
    architectures = {
        str(background_report.get("architecture", "fixed_channel")),
        str(injection_report.get("architecture", "fixed_channel")),
    }
    if len(architectures) != 1:
        raise ValueError("Background/injection score reports disagree on architecture")
    architecture = next(iter(architectures))
    enabled_contracts = {
        tuple(report.get("enabled_ifos", report["model_ifos"]))
        for report in (background_report, injection_report)
    }
    if len(enabled_contracts) != 1:
        raise ValueError("Background/injection score reports disagree on enabled_ifos")
    enabled_ifos = next(iter(enabled_contracts))
    training_architecture = str(training.get("architecture", "fixed_channel"))
    if training_architecture != architecture:
        raise ValueError("Scoring architecture differs from the training report")
    if identities["checkpoint_sha256"] != str(training["checkpoint_sha256"]):
        raise ValueError("Scored checkpoint differs from training report checkpoint")
    training_config_sha256 = str(
        training.get("config_file_sha256") or file_sha256(training["config_path"])
    )
    if identities["config_sha256"] != training_config_sha256:
        raise ValueError("Scoring config differs from training report config")
    if str(injection_report.get("manifest_sha256")) != str(
        training["validation_manifest_sha256"]
    ):
        raise ValueError("Scored injections differ from training validation manifest")
    if not background_rows or not injection_rows:
        raise ValueError("Physical validation endpoint requires background and injection rows")
    for label, rows in (("background", background_rows), ("injection", injection_rows)):
        invalid = sorted({str(row.get("split")) for row in rows if row.get("split") != "val"})
        if invalid:
            raise ValueError(f"Physical validation {label} contains non-val splits: {invalid}")
        if any("ranking_score" not in row for row in rows):
            raise ValueError(f"Physical validation {label} lacks ranking_score")
    window_ids = [str(row["window_id"]) for row in background_rows]
    injection_ids = [str(row["injection_id"]) for row in injection_rows]
    waveform_ids = [str(row["waveform_id"]) for row in injection_rows]
    if len(set(window_ids)) != len(window_ids):
        raise ValueError("Physical validation background contains duplicate window IDs")
    if len(set(injection_ids)) != len(injection_ids):
        raise ValueError("Physical validation injections contain duplicate injection IDs")
    if len(set(waveform_ids)) != len(waveform_ids):
        raise ValueError("Physical validation injections contain duplicate waveform IDs")
    intervals = [(float(row["gps_start"]), float(row["gps_end"])) for row in background_rows]
    if any(end <= start for start, end in intervals):
        raise ValueError("Physical validation background contains invalid GPS intervals")
    live_time_seconds = _union_duration(intervals)
    live_time_years = live_time_seconds / SECONDS_PER_YEAR
    calibration = calibrate_validation_count(
        (row["ranking_score"] for row in background_rows), maximum_validation_false_alarms
    )
    false_alarms = int(calibration["background_count"])
    nominal_far = false_alarms / live_time_years if false_alarms else 0.0
    threshold = float(calibration["threshold"])
    overall = summarize_injection_efficiency(
        injection_rows, threshold, "ranking_score", bootstrap_replicates, seed
    )
    stratum_field = (
        "source_family" if all("source_family" in row for row in injection_rows) else "stratum"
    )
    strata = {}
    for index, stratum in enumerate(
        sorted({str(row.get(stratum_field, "all")) for row in injection_rows})
    ):
        selected = [
            row for row in injection_rows if str(row.get(stratum_field, "all")) == stratum
        ]
        strata[stratum] = summarize_injection_efficiency(
            selected,
            threshold,
            "ranking_score",
            bootstrap_replicates,
            seed + index + 1,
        )
    result = {
        "status": "validation_only_exposure_limited_physical_endpoint",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "window-level O4a validation exposure is insufficient for an astrophysical FAR/IFAR; "
            "candidate clustering, time slides and the independently locked test corpus remain required"
        ),
        "protocol": (
            "threshold frozen from validation background count only, then applied once to the same "
            "checkpoint's validation injections"
        ),
        **identities,
        "detector_contract": {
            "architecture": architecture,
            "model_ifos": list(injection_report["model_ifos"]),
            "enabled_ifos": list(enabled_ifos),
        },
        "training": {
            "report_path": str(training_report),
            "report_sha256": file_sha256(training_report),
            "code_commit": training["code_commit"],
            "seed": training["seed"],
            "train_manifest_sha256": training["train_manifest_sha256"],
            "validation_manifest_sha256": training["validation_manifest_sha256"],
            "checkpoint_selection": training.get("checkpoint_selection"),
            "selected_epoch": training.get("selected_epoch"),
        },
        "background_score_report_path": str(background_score_report),
        "background_score_report_sha256": file_sha256(background_score_report),
        "injection_score_report_path": str(injection_score_report),
        "injection_score_report_sha256": file_sha256(injection_score_report),
        "background": {
            "windows": len(background_rows),
            "gps_blocks": len({str(row["gps_block"]) for row in background_rows}),
            "live_time_seconds": live_time_seconds,
            "live_time_days": live_time_seconds / 86400.0,
            "live_time_years": live_time_years,
            "adequate_for_astrophysical_far": False,
            "nominal_window_far_per_year_diagnostic_only": nominal_far,
            "nominal_window_ifar_years_diagnostic_only": (
                1.0 / nominal_far if nominal_far > 0 else None
            ),
        },
        "calibration": calibration,
        "injections": {
            "unique_injection_ids": len(set(injection_ids)),
            "unique_waveform_ids": len(set(waveform_ids)),
            "gps_blocks": len({str(row["gps_block"]) for row in injection_rows}),
            "overall": overall,
            "strata": strata,
        },
        "bootstrap_seed": seed,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def compare_validation_score_fields(
    background_rows: list[dict[str, Any]],
    injection_rows: list[dict[str, Any]],
    maximum_validation_false_alarms: int,
    score_field_a: str,
    score_field_b: str,
    bootstrap_replicates: int = 2000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Compare two predeclared rankings using validation-only count calibration."""
    if not background_rows or not injection_rows:
        raise ValueError("validation score comparison requires background and injections")
    for label, rows in (("background", background_rows), ("injection", injection_rows)):
        if any(row.get("split") != "val" for row in rows):
            raise ValueError(f"validation score comparison received non-val {label} rows")
        missing = [
            index
            for index, row in enumerate(rows)
            if score_field_a not in row or score_field_b not in row
        ]
        if missing:
            raise ValueError(f"validation {label} rows lack compared scores: {missing[:10]}")
    calibrations = {
        field: calibrate_validation_count(
            (float(row[field]) for row in background_rows),
            maximum_validation_false_alarms,
        )
        for field in (score_field_a, score_field_b)
    }
    summaries = {
        field: summarize_injection_efficiency(
            injection_rows,
            float(calibrations[field]["threshold"]),
            field,
            bootstrap_replicates,
            seed + index,
        )
        for index, field in enumerate((score_field_a, score_field_b))
    }
    paired = paired_vt_comparison(
        injection_rows,
        float(calibrations[score_field_a]["threshold"]),
        float(calibrations[score_field_b]["threshold"]),
        score_field_a,
        score_field_b,
        bootstrap_replicates,
        seed + 2,
    )
    return {
        "score_field_a": score_field_a,
        "score_field_b": score_field_b,
        "maximum_validation_false_alarms": maximum_validation_false_alarms,
        "calibrations": calibrations,
        "injection_summaries": summaries,
        "paired_comparison": paired,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


def evaluate_mask_search_robustness(
    background_raw: list[dict[str, Any]],
    background_mask: list[dict[str, Any]],
    clean_raw: list[dict[str, Any]],
    clean_mask: list[dict[str, Any]],
    contaminated_raw: list[dict[str, Any]],
    contaminated_mask: list[dict[str, Any]],
    maximum_validation_false_alarms: int,
    clean_noninferiority_margin: float = 0.01,
    minimum_contaminated_efficiency_gain: float = 0.05,
    score_field: str = "ranking_score",
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Evaluate paired mask gains and clean-data non-inferiority at frozen thresholds."""
    if not 0 <= clean_noninferiority_margin < 1:
        raise ValueError("clean non-inferiority margin must be in [0,1)")
    if not 0 <= minimum_contaminated_efficiency_gain < 1:
        raise ValueError("minimum contaminated gain must be in [0,1)")
    for name, rows in (
        ("background_raw", background_raw),
        ("background_mask", background_mask),
        ("clean_raw", clean_raw),
        ("clean_mask", clean_mask),
        ("contaminated_raw", contaminated_raw),
        ("contaminated_mask", contaminated_mask),
    ):
        if not rows or any(row.get("split") != "val" for row in rows):
            raise ValueError(f"mask search robustness requires non-empty val-only {name}")
        if any(score_field not in row for row in rows):
            raise ValueError(f"mask search robustness {name} lacks {score_field}")
    raw_windows = {
        str(row["window_id"]): (float(row["gps_start"]), float(row["gps_end"]))
        for row in background_raw
    }
    mask_windows = {
        str(row["window_id"]): (float(row["gps_start"]), float(row["gps_end"]))
        for row in background_mask
    }
    if len(raw_windows) != len(background_raw) or raw_windows != mask_windows:
        raise ValueError("raw/mask validation backgrounds use different windows or GPS intervals")
    calibrations = {
        "raw": calibrate_validation_count(
            (float(row[score_field]) for row in background_raw),
            maximum_validation_false_alarms,
        ),
        "mask_conditioned": calibrate_validation_count(
            (float(row[score_field]) for row in background_mask),
            maximum_validation_false_alarms,
        ),
    }

    def paired_rows(
        raw_rows: list[dict[str, Any]], mask_rows: list[dict[str, Any]], condition: str
    ) -> list[dict[str, Any]]:
        raw_by_id = {str(row["injection_id"]): row for row in raw_rows}
        mask_by_id = {str(row["injection_id"]): row for row in mask_rows}
        if len(raw_by_id) != len(raw_rows) or len(mask_by_id) != len(mask_rows):
            raise ValueError(f"duplicate {condition} injection IDs")
        if set(raw_by_id) != set(mask_by_id):
            raise ValueError(f"raw/mask {condition} injection IDs differ")
        joined = []
        for injection_id in sorted(raw_by_id):
            raw = raw_by_id[injection_id]
            masked = mask_by_id[injection_id]
            if (
                str(raw["waveform_id"]) != str(masked["waveform_id"])
                or float(raw["vt_weight"]) != float(masked["vt_weight"])
            ):
                raise ValueError(f"raw/mask {condition} injection provenance differs")
            joined.append(
                {
                    **raw,
                    "stratum": raw.get("contamination_stratum", condition),
                    "raw_score": raw[score_field],
                    "mask_score": masked[score_field],
                }
            )
        return joined

    clean = paired_rows(clean_raw, clean_mask, "clean")
    contaminated = paired_rows(contaminated_raw, contaminated_mask, "contaminated")
    clean_waveforms = {str(row["waveform_id"]) for row in clean}
    contaminated_waveforms = {str(row["waveform_id"]) for row in contaminated}
    if clean_waveforms != contaminated_waveforms:
        raise ValueError("clean and contaminated arms use different waveform populations")
    comparisons = {
        "clean": paired_vt_comparison(
            clean,
            float(calibrations["raw"]["threshold"]),
            float(calibrations["mask_conditioned"]["threshold"]),
            "raw_score",
            "mask_score",
            bootstrap_replicates,
            seed,
        ),
        "contaminated": paired_vt_comparison(
            contaminated,
            float(calibrations["raw"]["threshold"]),
            float(calibrations["mask_conditioned"]["threshold"]),
            "raw_score",
            "mask_score",
            bootstrap_replicates,
            seed + 1,
        ),
    }
    clean_total_vt = float(sum(float(row["vt_weight"]) for row in clean))
    contaminated_total_vt = float(
        sum(float(row["vt_weight"]) for row in contaminated)
    )
    allowed_clean_loss = clean_noninferiority_margin * clean_total_vt
    clean_lower = float(comparisons["clean"]["paired_bootstrap_95"][0])
    contaminated_lower = float(
        comparisons["contaminated"]["paired_bootstrap_95"][0]
    )
    contaminated_gain = float(
        comparisons["contaminated"]["delta_recovered_vt_b_minus_a"]
        / contaminated_total_vt
    )
    gates = {
        "clean_noninferiority": {
            "absolute_efficiency_margin": clean_noninferiority_margin,
            "maximum_allowed_vt_loss": allowed_clean_loss,
            "paired_delta_lower_95": clean_lower,
            "passed": clean_lower >= -allowed_clean_loss,
        },
        "contaminated_material_gain": {
            "minimum_absolute_efficiency_gain": minimum_contaminated_efficiency_gain,
            "observed_absolute_efficiency_gain": contaminated_gain,
            "paired_delta_lower_95": contaminated_lower,
            "passed": contaminated_gain >= minimum_contaminated_efficiency_gain
            and contaminated_lower > 0,
        },
    }
    return {
        "protocol": (
            "raw and mask thresholds independently frozen on paired validation background, then "
            "applied unchanged to clean and contaminated waveform-matched injections"
        ),
        "score_field": score_field,
        "maximum_validation_false_alarms": maximum_validation_false_alarms,
        "background_windows": len(raw_windows),
        "background_live_time_seconds": _union_duration(raw_windows.values()),
        "calibrations": calibrations,
        "comparisons": comparisons,
        "gates": gates,
        "development_gates_passed": all(item["passed"] for item in gates.values()),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


def run_mask_search_validation(
    background_raw_path: str | Path,
    background_mask_path: str | Path,
    clean_raw_path: str | Path,
    clean_mask_path: str | Path,
    contaminated_raw_path: str | Path,
    contaminated_mask_path: str | Path,
    output: str | Path,
    maximum_validation_false_alarms: int,
    clean_noninferiority_margin: float = 0.01,
    minimum_contaminated_efficiency_gain: float = 0.05,
    score_field: str = "ranking_score",
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    paths = {
        "background_raw": background_raw_path,
        "background_mask": background_mask_path,
        "clean_raw": clean_raw_path,
        "clean_mask": clean_mask_path,
        "contaminated_raw": contaminated_raw_path,
        "contaminated_mask": contaminated_mask_path,
    }
    loaded = {name: load_jsonl(path) for name, path in paths.items()}
    comparison = evaluate_mask_search_robustness(
        **loaded,
        maximum_validation_false_alarms=maximum_validation_false_alarms,
        clean_noninferiority_margin=clean_noninferiority_margin,
        minimum_contaminated_efficiency_gain=minimum_contaminated_efficiency_gain,
        score_field=score_field,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    result = {
        "status": "validation_only_mask_search_robustness",
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "scientific_blocker": (
            "continuous clustered background/time-slide exposure and locked injection evaluation "
            "remain required even when development gates pass"
        ),
        "artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        **comparison,
        "test_evaluation": None,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def run_coherence_validation_comparison(
    background_score_report: str | Path,
    injection_score_report: str | Path,
    output: str | Path,
    maximum_validation_false_alarms: int,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    background_report, background_rows = _verified_score_artifact(
        background_score_report, "background"
    )
    injection_report, injection_rows = _verified_score_artifact(
        injection_score_report, "injection"
    )
    for field in ("checkpoint_sha256", "config_sha256", "code_commit"):
        if str(background_report[field]) != str(injection_report[field]):
            raise ValueError(f"coherence comparison score reports disagree on {field}")
    if not background_report.get("coherence") or not injection_report.get("coherence"):
        raise ValueError("coherence comparison requires coherence-enabled score reports")
    if background_report["coherence"] != injection_report["coherence"]:
        raise ValueError("background/injection coherence protocols differ")
    comparison = compare_validation_score_fields(
        background_rows,
        injection_rows,
        maximum_validation_false_alarms,
        "ranking_score",
        "coherence_assisted_score",
        bootstrap_replicates,
        seed,
    )
    timing_errors: dict[str, list[float]] = {}
    network_errors = []
    for row in injection_rows:
        peaks = row.get("strain_envelope_peak_times")
        if not peaks or "gps_time" not in row:
            raise ValueError("coherence injection scores lack envelope peaks or injection GPS")
        truth = float(row["gps_time"])
        network_peaks = []
        for ifo, peak in peaks.items():
            value = float(peak["gps"])
            timing_errors.setdefault(str(ifo), []).append(abs(value - truth))
            network_peaks.append(value)
        network_errors.append(abs(float(np.median(network_peaks)) - truth))

    def timing_summary(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "injections": int(array.size),
            "median_absolute_error_seconds": float(np.median(array)),
            "p90_absolute_error_seconds": float(np.quantile(array, 0.9)),
            "within_10ms": int(np.count_nonzero(array <= 0.01)),
            "within_10ms_rate": float(np.mean(array <= 0.01)),
        }

    timing = {
        "network_median_peak": timing_summary(network_errors),
        "by_ifo": {
            ifo: timing_summary(values) for ifo, values in sorted(timing_errors.items())
        },
    }
    timing["empirical_10ms_gate_passed"] = (
        timing["network_median_peak"]["p90_absolute_error_seconds"] <= 0.01
    )
    result = {
        "status": "validation_only_morphology_vs_physical_coherence",
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "scientific_blocker": (
            "short validation-window exposure is a model-selection diagnostic; continuous "
            "clustered background/time slides and locked injections remain required"
        ),
        "protocol": (
            "each ranking threshold calibrated independently on the same validation background "
            "count, then compared on paired validation injections"
        ),
        "checkpoint_sha256": background_report["checkpoint_sha256"],
        "config_sha256": background_report["config_sha256"],
        "coherence": background_report["coherence"],
        "strain_envelope_timing": timing,
        "background_score_report_sha256": file_sha256(background_score_report),
        "injection_score_report_sha256": file_sha256(injection_score_report),
        **comparison,
        "test_evaluation": None,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def detector_subset_noninferiority(
    paired_comparison: dict[str, Any],
    relative_margin: float,
) -> dict[str, Any]:
    """Apply a predeclared one-sided loss margin to a paired VT bootstrap."""
    if not 0 <= relative_margin < 1:
        raise ValueError("detector subset non-inferiority margin must be in [0, 1)")
    reference_vt = float(paired_comparison["method_a"]["recovered_vt"])
    lower = float(paired_comparison["paired_bootstrap_95"][0])
    allowed_loss = relative_margin * reference_vt
    return {
        "relative_margin": relative_margin,
        "reference_recovered_vt": reference_vt,
        "maximum_allowed_absolute_vt_loss": allowed_loss,
        "paired_delta_lower_95": lower,
        "passed": lower >= -allowed_loss,
    }


def summarize_detector_subset_endpoints(
    endpoint_reports: list[str | Path],
    output: str | Path,
    reference_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    relative_noninferiority_margin: float = 0.1,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Compare one checkpoint across independently calibrated detector subsets."""
    if len(endpoint_reports) < 2:
        raise ValueError("detector-subset summary requires at least two endpoint reports")
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    controls: dict[str, set[str]] = {
        "checkpoint_sha256": set(),
        "config_sha256": set(),
        "training_report_sha256": set(),
        "training_seed": set(),
        "injection_manifest_sha256": set(),
        "background_manifest_sha256": set(),
        "maximum_validation_false_alarms": set(),
    }
    model_contract: tuple[str, ...] | None = None
    architecture: str | None = None
    for endpoint_value in endpoint_reports:
        endpoint_path = Path(endpoint_value)
        with endpoint_path.open("r", encoding="utf-8") as handle:
            endpoint = json.load(handle)
        if endpoint.get("status") != "validation_only_exposure_limited_physical_endpoint":
            raise ValueError(f"invalid detector-subset endpoint: {endpoint_path}")
        contract = endpoint.get("detector_contract")
        if not contract:
            raise ValueError(f"endpoint lacks an explicit detector contract: {endpoint_path}")
        current_model = tuple(str(ifo) for ifo in contract["model_ifos"])
        current_architecture = str(contract["architecture"])
        if model_contract is None:
            model_contract = current_model
            architecture = current_architecture
        if current_model != model_contract or current_architecture != architecture:
            raise ValueError("detector-subset endpoints use different model contracts")
        enabled_set = set(str(ifo) for ifo in contract["enabled_ifos"])
        enabled = tuple(ifo for ifo in current_model if ifo in enabled_set)
        if len(enabled) < 2 or len(enabled) != len(enabled_set):
            raise ValueError("detector-subset endpoints require unique network-mode IFO sets")
        if enabled in records:
            raise ValueError(f"duplicate detector-subset endpoint: {enabled}")
        injection_report, injection_rows = _verified_score_artifact(
            endpoint["injection_score_report_path"], "injection"
        )
        background_report, _ = _verified_score_artifact(
            endpoint["background_score_report_path"], "background"
        )
        records[enabled] = {
            "endpoint_path": str(endpoint_path),
            "endpoint_sha256": file_sha256(endpoint_path),
            "threshold": float(endpoint["calibration"]["threshold"]),
            "overall": endpoint["injections"]["overall"],
            "rows": injection_rows,
        }
        controls["checkpoint_sha256"].add(str(endpoint["checkpoint_sha256"]))
        controls["config_sha256"].add(str(endpoint["config_sha256"]))
        controls["training_report_sha256"].add(str(endpoint["training"]["report_sha256"]))
        controls["training_seed"].add(str(endpoint["training"]["seed"]))
        controls["injection_manifest_sha256"].add(str(injection_report["manifest_sha256"]))
        controls["background_manifest_sha256"].add(str(background_report["manifest_sha256"]))
        controls["maximum_validation_false_alarms"].add(
            str(endpoint["calibration"]["maximum_validation_false_alarms"])
        )
    disagreements = {name: sorted(values) for name, values in controls.items() if len(values) != 1}
    if disagreements:
        raise ValueError(f"detector-subset endpoints disagree on controls: {disagreements}")
    assert model_contract is not None and architecture is not None
    if reference_ifos not in records:
        raise ValueError("reference detector set is absent from endpoint reports")
    expected_subsets = {
        tuple(subset)
        for size in range(2, len(model_contract) + 1)
        for subset in combinations(model_contract, size)
    }
    reference = records[reference_ifos]
    reference_by_id = {str(row["injection_id"]): row for row in reference["rows"]}
    comparisons = {}
    for subset, record in sorted(records.items()):
        candidate_by_id = {str(row["injection_id"]): row for row in record["rows"]}
        if set(candidate_by_id) != set(reference_by_id):
            raise ValueError("detector-subset endpoints use different injection IDs")
        joined = []
        for injection_id in sorted(reference_by_id):
            reference_row = reference_by_id[injection_id]
            candidate_row = candidate_by_id[injection_id]
            if (
                str(reference_row["waveform_id"]) != str(candidate_row["waveform_id"])
                or float(reference_row["vt_weight"]) != float(candidate_row["vt_weight"])
            ):
                raise ValueError("detector-subset injection provenance differs")
            joined.append(
                {
                    **reference_row,
                    "stratum": reference_row.get("source_family", "all"),
                    "reference_score": reference_row["ranking_score"],
                    "subset_score": candidate_row["ranking_score"],
                }
            )
        comparison = paired_vt_comparison(
            joined,
            float(reference["threshold"]),
            float(record["threshold"]),
            "reference_score",
            "subset_score",
            bootstrap_replicates,
            seed + len(comparisons),
        )
        comparisons["+".join(subset)] = {
            "enabled_ifos": list(subset),
            "endpoint_path": record["endpoint_path"],
            "endpoint_sha256": record["endpoint_sha256"],
            "weighted_efficiency": record["overall"]["weighted_efficiency"],
            "paired_vs_reference": comparison,
            "noninferiority": detector_subset_noninferiority(
                comparison, relative_noninferiority_margin
            ),
        }
    complete = set(records) == expected_subsets
    all_passed = complete and all(
        row["noninferiority"]["passed"] for row in comparisons.values()
    )
    result = {
        "status": "validation_only_detector_subset_robustness",
        "scientific_claim_allowed": False,
        "protocol": (
            "same checkpoint and paired injections; each detector subset threshold is calibrated "
            "only on its matching validation background"
        ),
        "architecture": architecture,
        "model_ifos": list(model_contract),
        "reference_ifos": list(reference_ifos),
        "expected_network_subsets": [list(item) for item in sorted(expected_subsets)],
        "observed_network_subsets": [list(item) for item in sorted(records)],
        "complete_predeclared_subset_gate": complete,
        "all_subset_noninferiority_passed": all_passed,
        "relative_noninferiority_margin": relative_noninferiority_margin,
        "controls": {name: next(iter(values)) for name, values in controls.items()},
        "comparisons": comparisons,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def aggregate_physical_endpoint_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate pre-audited endpoint records and retain seed-level null results."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one physical endpoint record is required")
    keys = [(int(row["scale"]), int(row["seed"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Physical endpoint records contain duplicate scale/seed pairs")
    scales = []
    for scale in sorted({item[0] for item in keys}):
        selected = sorted(
            (row for row in rows if int(row["scale"]) == scale),
            key=lambda row: int(row["seed"]),
        )
        values = np.asarray(
            [float(row["weighted_efficiency"]) for row in selected], dtype=np.float64
        )
        scales.append(
            {
                "scale": scale,
                "seeds": [int(row["seed"]) for row in selected],
                "seed_count": len(selected),
                "weighted_efficiency_mean": float(values.mean()),
                "weighted_efficiency_sample_std": (
                    float(values.std(ddof=1)) if len(values) >= 2 else None
                ),
                "minimum_three_seed_gate": len(selected) >= 3,
                "runs": selected,
            }
        )
    adjacent_seed_deltas = []
    for lower, upper in zip(scales, scales[1:]):
        lower_by_seed = {int(row["seed"]): row for row in lower["runs"]}
        upper_by_seed = {int(row["seed"]): row for row in upper["runs"]}
        common = sorted(set(lower_by_seed) & set(upper_by_seed))
        deltas = np.asarray(
            [
                float(upper_by_seed[seed]["weighted_efficiency"])
                - float(lower_by_seed[seed]["weighted_efficiency"])
                for seed in common
            ],
            dtype=np.float64,
        )
        adjacent_seed_deltas.append(
            {
                "lower_scale": lower["scale"],
                "upper_scale": upper["scale"],
                "paired_seeds": common,
                "seed_count": len(common),
                "weighted_efficiency_delta_mean": (
                    float(deltas.mean()) if len(deltas) else None
                ),
                "weighted_efficiency_delta_sample_std": (
                    float(deltas.std(ddof=1)) if len(deltas) >= 2 else None
                ),
                "all_seed_deltas_positive": bool(len(deltas) and np.all(deltas > 0)),
            }
        )
    return {
        "scales": scales,
        "adjacent_seed_deltas": adjacent_seed_deltas,
        "minimum_three_seed_gate": all(item["minimum_three_seed_gate"] for item in scales),
    }


def summarize_physical_validation_endpoints(
    endpoint_reports: list[str | Path],
    scale_subset_report: str | Path,
    output: str | Path,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Summarize controlled scale endpoints with paired injection-level comparisons."""
    if not endpoint_reports:
        raise ValueError("At least one physical validation endpoint report is required")
    with Path(scale_subset_report).open("r", encoding="utf-8") as handle:
        scale_plan = json.load(handle)
    scale_by_train_hash = {
        str(item["manifest_sha256"]): int(item["scale"])
        for item in scale_plan.get("scales", [])
    }
    expected_validation = str(scale_plan["validation_manifest_sha256"])
    records = []
    scored_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    controlled: dict[str, set[str]] = {
        "scoring_code_commit": set(),
        "scoring_config_sha256": set(),
        "training_code_commit": set(),
        "background_manifest_sha256": set(),
        "validation_manifest_sha256": set(),
        "maximum_validation_false_alarms": set(),
        "live_time_seconds": set(),
        "checkpoint_selection": set(),
    }
    for endpoint_path_value in endpoint_reports:
        endpoint_path = Path(endpoint_path_value)
        with endpoint_path.open("r", encoding="utf-8") as handle:
            endpoint = json.load(handle)
        if endpoint.get("status") != "validation_only_exposure_limited_physical_endpoint":
            raise ValueError(f"Invalid physical validation endpoint status: {endpoint_path}")
        if endpoint.get("scientific_claim_allowed") is not False:
            raise ValueError(f"Endpoint lacks exposure-limited claim guard: {endpoint_path}")
        training = endpoint["training"]
        train_hash = str(training["train_manifest_sha256"])
        if train_hash not in scale_by_train_hash:
            raise ValueError(f"Endpoint is not from the frozen scale plan: {endpoint_path}")
        if str(training["validation_manifest_sha256"]) != expected_validation:
            raise ValueError(f"Endpoint uses a different validation manifest: {endpoint_path}")
        scale = scale_by_train_hash[train_hash]
        run_seed = int(training["seed"])
        injection_report_path = Path(endpoint["injection_score_report_path"])
        background_report_path = Path(endpoint["background_score_report_path"])
        if file_sha256(injection_report_path) != endpoint["injection_score_report_sha256"]:
            raise ValueError(f"Endpoint injection score report hash mismatch: {endpoint_path}")
        if file_sha256(background_report_path) != endpoint["background_score_report_sha256"]:
            raise ValueError(f"Endpoint background score report hash mismatch: {endpoint_path}")
        injection_report, injection_rows = _verified_score_artifact(
            injection_report_path, "injection"
        )
        background_report, _ = _verified_score_artifact(background_report_path, "background")
        if str(injection_report["manifest_sha256"]) != expected_validation:
            raise ValueError(f"Scored injection manifest differs from frozen validation: {endpoint_path}")
        key = (scale, run_seed)
        scored_rows[key] = injection_rows
        overall = endpoint["injections"]["overall"]
        records.append(
            {
                "scale": scale,
                "seed": run_seed,
                "weighted_efficiency": float(overall["weighted_efficiency"]),
                "weighted_efficiency_bootstrap_95": overall[
                    "weighted_efficiency_bootstrap_95"
                ],
                "recovered_vt": float(overall["recovered_vt"]),
                "threshold": float(endpoint["calibration"]["threshold"]),
                "checkpoint_sha256": str(endpoint["checkpoint_sha256"]),
                "endpoint_report_path": str(endpoint_path),
                "endpoint_report_sha256": file_sha256(endpoint_path),
            }
        )
        controlled["scoring_code_commit"].add(str(endpoint["code_commit"]))
        controlled["scoring_config_sha256"].add(str(endpoint["config_sha256"]))
        controlled["training_code_commit"].add(str(training["code_commit"]))
        controlled["background_manifest_sha256"].add(
            str(background_report["manifest_sha256"])
        )
        controlled["validation_manifest_sha256"].add(
            str(injection_report["manifest_sha256"])
        )
        controlled["maximum_validation_false_alarms"].add(
            str(endpoint["calibration"]["maximum_validation_false_alarms"])
        )
        controlled["live_time_seconds"].add(str(endpoint["background"]["live_time_seconds"]))
        controlled["checkpoint_selection"].add(str(training.get("checkpoint_selection")))
    disagreements = {field: sorted(values) for field, values in controlled.items() if len(values) != 1}
    if disagreements:
        raise ValueError(f"Physical validation endpoints disagree on controls: {disagreements}")
    aggregate = aggregate_physical_endpoint_records(records)
    paired = []
    scales = [item["scale"] for item in aggregate["scales"]]
    for pair_index, (lower, upper) in enumerate(zip(scales, scales[1:])):
        for run_seed in sorted(
            {key[1] for key in scored_rows if key[0] == lower}
            & {key[1] for key in scored_rows if key[0] == upper}
        ):
            lower_rows = {str(row["injection_id"]): row for row in scored_rows[(lower, run_seed)]}
            upper_rows = {str(row["injection_id"]): row for row in scored_rows[(upper, run_seed)]}
            if set(lower_rows) != set(upper_rows):
                raise ValueError("Adjacent endpoint scales use different physical injection IDs")
            joined = []
            for injection_id in sorted(lower_rows):
                row_a = lower_rows[injection_id]
                row_b = upper_rows[injection_id]
                if (
                    str(row_a["waveform_id"]) != str(row_b["waveform_id"])
                    or float(row_a["vt_weight"]) != float(row_b["vt_weight"])
                ):
                    raise ValueError("Adjacent endpoint scales disagree on injection provenance")
                joined.append(
                    {
                        **row_a,
                        "lower_score": row_a["ranking_score"],
                        "upper_score": row_b["ranking_score"],
                    }
                )
            lower_record = next(
                row for row in records if row["scale"] == lower and row["seed"] == run_seed
            )
            upper_record = next(
                row for row in records if row["scale"] == upper and row["seed"] == run_seed
            )
            comparison = paired_vt_comparison(
                joined,
                lower_record["threshold"],
                upper_record["threshold"],
                "lower_score",
                "upper_score",
                bootstrap_replicates,
                seed + pair_index * 100 + run_seed,
            )
            paired.append(
                {
                    "lower_scale": lower,
                    "upper_scale": upper,
                    "seed": run_seed,
                    **comparison,
                }
            )
    checkpoint_selection = next(iter(controlled["checkpoint_selection"]))
    control_protocol = {
        "final_update": "fixed_update",
        "best_validation": "fixed_epoch",
    }.get(checkpoint_selection, "unknown")
    result = {
        "status": "physical_validation_endpoint_scale_summary",
        "control_protocol": control_protocol,
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "promotion_blockers": [
            "fixed-update and fixed-epoch controls require joint adjudication",
            "O4a window exposure is insufficient for astrophysical FAR/IFAR",
            "locked test remains unopened",
        ],
        "controls": {field: next(iter(values)) for field, values in controlled.items()},
        "scale_subset_report_path": str(scale_subset_report),
        "scale_subset_report_sha256": file_sha256(scale_subset_report),
        **aggregate,
        "paired_injection_comparisons": paired,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "test_evaluation": None,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def score_physical_training_series(
    training_series_dir: str | Path,
    background_manifest: str | Path,
    injection_manifest: str | Path,
    config_path: str | Path,
    scale_subset_report: str | Path,
    output_dir: str | Path,
    maximum_validation_false_alarms: int,
    context_duration: float = 64.0,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Resumably score every completed scale/seed checkpoint on one frozen endpoint."""
    from .injection_score import score_materialized_injections
    from .trigger import score_background_manifest

    if maximum_validation_false_alarms < 0:
        raise ValueError("maximum_validation_false_alarms must be non-negative")
    config = load_yaml(config_path)
    settings = config["physical_training"]
    model_ifos = tuple(str(item) for item in settings["model_ifos"])
    q_values = tuple(float(item) for item in settings["q_values"])
    target_sample_rate = int(settings["target_sample_rate"])
    root = Path(training_series_dir)
    report_paths = sorted(root.glob("scale-*/seed-*/physical_finetune_report.json"))
    if not report_paths:
        raise ValueError("training series contains no completed physical reports")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    endpoint_paths = []
    run_records = []
    seen_keys = set()
    for report_path in report_paths:
        with report_path.open("r", encoding="utf-8") as handle:
            training = json.load(handle)
        scale_name = report_path.parent.parent.name
        seed_name = report_path.parent.name
        if not scale_name.startswith("scale-") or not seed_name.startswith("seed-"):
            raise ValueError(f"unexpected physical series path: {report_path}")
        scale = int(scale_name.removeprefix("scale-"))
        run_seed = int(seed_name.removeprefix("seed-"))
        key = (scale, run_seed)
        if key in seen_keys:
            raise ValueError(f"duplicate physical series run: {key}")
        seen_keys.add(key)
        if training.get("test_evaluation") is not None:
            raise ValueError(f"training series report accessed test data: {report_path}")
        if canonical_hash(config) != str(training["config_hash"]):
            raise ValueError(f"training series config differs from scorer config: {report_path}")
        checkpoint_path = Path(training["checkpoint_path"])
        if file_sha256(checkpoint_path) != str(training["checkpoint_sha256"]):
            raise ValueError(f"training series checkpoint hash mismatch: {report_path}")
        run_output = output / scale_name / seed_name
        background_output = run_output / "background"
        injection_output = run_output / "injections"
        background_report = score_background_manifest(
            background_manifest,
            checkpoint_path,
            config_path,
            background_output,
            model_ifos,
            q_values,
            target_sample_rate,
            context_duration,
            False,
            "val",
        )
        injection_report = score_materialized_injections(
            injection_manifest,
            checkpoint_path,
            config_path,
            injection_output,
            model_ifos,
            q_values,
            target_sample_rate,
            False,
            "val",
        )
        endpoint_path = run_output / "physical_validation_endpoint.json"
        endpoint = run_physical_validation_endpoint(
            report_path,
            background_output / "trigger_score_report.json",
            injection_output / "injection_score_report.json",
            maximum_validation_false_alarms,
            endpoint_path,
            bootstrap_replicates,
            seed + len(endpoint_paths),
        )
        endpoint_paths.append(endpoint_path)
        run_records.append(
            {
                "scale": scale,
                "seed": run_seed,
                "training_report_path": str(report_path),
                "training_report_sha256": file_sha256(report_path),
                "checkpoint_sha256": training["checkpoint_sha256"],
                "background_score_report_sha256": file_sha256(
                    background_output / "trigger_score_report.json"
                ),
                "injection_score_report_sha256": file_sha256(
                    injection_output / "injection_score_report.json"
                ),
                "endpoint_sha256": file_sha256(endpoint_path),
                "weighted_efficiency": endpoint["injections"]["overall"][
                    "weighted_efficiency"
                ],
                "scored_background_windows": background_report["scored_windows"],
                "scored_injections": injection_report["scored_injections"],
            }
        )
    summary_path = output / "physical_validation_scale_summary.json"
    summary = summarize_physical_validation_endpoints(
        endpoint_paths,
        scale_subset_report,
        summary_path,
        bootstrap_replicates,
        seed,
    )
    result = {
        "status": "complete_physical_validation_endpoint_series",
        "scientific_claim_allowed": False,
        "training_series_dir": str(root),
        "training_report_count": len(report_paths),
        "background_manifest_sha256": file_sha256(background_manifest),
        "injection_manifest_sha256": file_sha256(injection_manifest),
        "config_hash": canonical_hash(config),
        "config_file_sha256": file_sha256(config_path),
        "scale_subset_report_sha256": file_sha256(scale_subset_report),
        "maximum_validation_false_alarms": maximum_validation_false_alarms,
        "runs": run_records,
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "control_protocol": summary["control_protocol"],
        "test_evaluation": None,
        **execution_provenance(),
    }
    atomic_write_json(output / "physical_endpoint_series_report.json", result)
    return result


def run_search_comparison(
    validation_background: str | Path,
    test_background: str | Path,
    test_injections: str | Path,
    validation_live_time_years: float,
    test_live_time_years: float,
    target_far_per_year: float,
    score_field_a: str,
    score_field_b: str,
    output: str | Path,
    bootstrap_replicates: int = 2000,
    seed: int = 20260719,
) -> dict[str, Any]:
    result = compare_search_methods(
        load_jsonl(validation_background),
        load_jsonl(test_background),
        load_jsonl(test_injections),
        validation_live_time_years,
        test_live_time_years,
        target_far_per_year,
        score_field_a,
        score_field_b,
        bootstrap_replicates,
        seed,
    )
    atomic_write_json(output, result)
    return result
