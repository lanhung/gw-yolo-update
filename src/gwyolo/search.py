from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .io import atomic_write_json


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
            "total_vt_weight": total_weight,
            "recovered_vt": recovered_weight,
            "weighted_efficiency": recovered_weight / total_weight,
        },
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
