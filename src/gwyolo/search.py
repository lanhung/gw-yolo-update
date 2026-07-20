from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, file_sha256
from .metrics import wilson_interval


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
