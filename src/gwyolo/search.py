from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .background import SECONDS_PER_YEAR, _union_duration
from .io import atomic_write_json, file_sha256
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
    zero_count_threshold = math.nextafter(scores[0], math.inf) if scores else 0.0
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
    if identities["checkpoint_sha256"] != str(training["checkpoint_sha256"]):
        raise ValueError("Scored checkpoint differs from training report checkpoint")
    if identities["config_sha256"] != file_sha256(training["config_path"]):
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
    result = {
        "status": "physical_fixed_update_validation_endpoint_summary",
        "scientific_claim_allowed": False,
        "promotion_allowed": False,
        "promotion_blockers": [
            "equal-epoch checkpoints still require the same frozen background/injection endpoint",
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
