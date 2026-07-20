from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import atomic_write_json, file_sha256
from .metrics import wilson_interval
from .runtime import execution_provenance


def calibrate_known_only_abstention(
    known_scores: Iterable[float],
    maximum_known_abstention_rate: float,
) -> dict[str, Any]:
    """Freeze an OOD threshold using known validation artifacts only."""
    if not 0 <= maximum_known_abstention_rate < 1:
        raise ValueError("maximum known abstention rate must be in [0, 1)")
    scores = np.asarray(list(known_scores), dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("known validation OOD scores must be non-empty and finite")
    maximum_count = int(math.floor(maximum_known_abstention_rate * scores.size))
    candidates = [math.nextafter(float(scores.max()), math.inf), *sorted(set(scores), reverse=True)]
    allowed = []
    for threshold in candidates:
        count = int(np.count_nonzero(scores >= threshold))
        if count <= maximum_count:
            allowed.append((float(threshold), count))
    if not allowed:
        raise AssertionError("zero-count OOD threshold must always satisfy calibration")
    threshold, count = min(allowed, key=lambda item: item[0])
    return {
        "threshold": threshold,
        "known_validation_rows": int(scores.size),
        "maximum_known_abstention_rate": maximum_known_abstention_rate,
        "maximum_known_abstentions": maximum_count,
        "observed_known_abstentions": count,
        "observed_known_abstention_rate": count / scores.size,
        "selection_data": "known_validation_only",
        "unknown_scores_used_for_selection": False,
        "tie_safe": True,
    }


def ood_auc(rows: list[dict[str, Any]], score_field: str = "ood_score") -> float:
    """Pair-count AUROC where larger scores indicate unknown artifacts."""
    known = [float(row[score_field]) for row in rows if not bool(row["is_unknown"])]
    unknown = [float(row[score_field]) for row in rows if bool(row["is_unknown"])]
    if not known or not unknown:
        raise ValueError("OOD AUROC requires known and unknown evaluation rows")
    wins = 0.0
    for unknown_score in unknown:
        for known_score in known:
            wins += float(unknown_score > known_score) + 0.5 * float(
                unknown_score == known_score
            )
    return wins / (len(known) * len(unknown))


def _rate(successes: int, total: int) -> dict[str, Any]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("OOD rate requires a valid non-empty binomial count")
    return {
        "count": successes,
        "total": total,
        "rate": successes / total,
        "wilson_95": list(wilson_interval(successes, total)),
    }


def evaluate_frozen_ood_threshold(
    calibration_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    maximum_known_abstention_rate: float = 0.05,
    score_field: str = "ood_score",
) -> dict[str, Any]:
    if not calibration_rows or not evaluation_rows:
        raise ValueError("OOD calibration and evaluation rows must be non-empty")
    required = {"glitch_id", "gps_block", "glitch_family", "observing_run", score_field}
    for label, rows in (("calibration", calibration_rows), ("evaluation", evaluation_rows)):
        missing = [index for index, row in enumerate(rows) if required - set(row)]
        if missing:
            raise ValueError(f"OOD {label} rows lack required fields at {missing[:10]}")
        scores = np.asarray([float(row[score_field]) for row in rows])
        if not np.isfinite(scores).all():
            raise ValueError(f"OOD {label} scores must be finite")
    if any(bool(row.get("is_unknown", False)) for row in calibration_rows):
        raise ValueError("OOD threshold calibration cannot contain unknown artifacts")
    if any(str(row.get("split")) != "val" for row in calibration_rows):
        raise ValueError("OOD threshold calibration must be validation-only")
    if any("is_unknown" not in row for row in evaluation_rows):
        raise ValueError("OOD evaluation rows require explicit is_unknown labels")
    overlaps = {}
    for field in ("glitch_id", "gps_block"):
        calibration_ids = {str(row[field]) for row in calibration_rows}
        evaluation_ids = {str(row[field]) for row in evaluation_rows}
        overlaps[field] = sorted(calibration_ids & evaluation_ids)
    if any(overlaps.values()):
        raise ValueError(f"OOD calibration/evaluation group leakage: {overlaps}")
    calibration = calibrate_known_only_abstention(
        (float(row[score_field]) for row in calibration_rows),
        maximum_known_abstention_rate,
    )
    threshold = float(calibration["threshold"])
    evaluated = [
        {
            **row,
            "abstained": float(row[score_field]) >= threshold,
        }
        for row in evaluation_rows
    ]
    known = [row for row in evaluated if not bool(row["is_unknown"])]
    unknown = [row for row in evaluated if bool(row["is_unknown"])]
    if not known or not unknown:
        raise ValueError("OOD evaluation requires both known and unknown rows")
    known_false_abstention = _rate(sum(row["abstained"] for row in known), len(known))
    unknown_true_abstention = _rate(sum(row["abstained"] for row in unknown), len(unknown))
    unknown_false_acceptance = _rate(sum(not row["abstained"] for row in unknown), len(unknown))

    def strata(field: str) -> dict[str, Any]:
        output = {}
        for value in sorted({str(row[field]) for row in evaluated}):
            selected = [row for row in evaluated if str(row[field]) == value]
            selected_unknown = [row for row in selected if bool(row["is_unknown"])]
            selected_known = [row for row in selected if not bool(row["is_unknown"])]
            output[value] = {
                "rows": len(selected),
                "unknown_rows": len(selected_unknown),
                "known_rows": len(selected_known),
                "unknown_true_abstention": (
                    _rate(sum(row["abstained"] for row in selected_unknown), len(selected_unknown))
                    if selected_unknown
                    else None
                ),
                "known_false_abstention": (
                    _rate(sum(row["abstained"] for row in selected_known), len(selected_known))
                    if selected_known
                    else None
                ),
            }
        return output

    return {
        "status": "frozen_known_only_ood_abstention_evaluation",
        "scientific_claim_allowed": False,
        "protocol": (
            "threshold frozen from known validation artifacts only; held-out families and runs "
            "are evaluated without threshold adjustment"
        ),
        "score_field": score_field,
        "higher_score_means": "more_unknown",
        "calibration": calibration,
        "split_audit": {"passed": True, "cross_split_overlaps": overlaps},
        "evaluation_rows": len(evaluated),
        "known_rows": len(known),
        "unknown_rows": len(unknown),
        "known_false_abstention": known_false_abstention,
        "unknown_true_abstention": unknown_true_abstention,
        "unknown_false_acceptance": unknown_false_acceptance,
        "auroc_diagnostic": ood_auc(evaluated, score_field),
        "family_strata": strata("glitch_family"),
        "observing_run_strata": strata("observing_run"),
        "unknown_family_counts": dict(
            sorted(Counter(str(row["glitch_family"]) for row in unknown).items())
        ),
    }


def run_ood_abstention_evaluation(
    calibration_manifest: str | Path,
    evaluation_manifest: str | Path,
    output: str | Path,
    maximum_known_abstention_rate: float = 0.05,
    score_field: str = "ood_score",
) -> dict[str, Any]:
    def load(path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    result = evaluate_frozen_ood_threshold(
        load(calibration_manifest),
        load(evaluation_manifest),
        maximum_known_abstention_rate,
        score_field,
    )
    result.update(
        {
            "calibration_manifest_path": str(calibration_manifest),
            "calibration_manifest_sha256": file_sha256(calibration_manifest),
            "evaluation_manifest_path": str(evaluation_manifest),
            "evaluation_manifest_sha256": file_sha256(evaluation_manifest),
            **execution_provenance(),
        }
    )
    atomic_write_json(output, result)
    return result
