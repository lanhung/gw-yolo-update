from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .background import SECONDS_PER_YEAR, _union_duration
from .io import atomic_write_json, file_sha256


def _ifos(row: dict[str, Any]) -> set[str]:
    values = row.get("valid_ifos", row.get("ifos"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"background window {row.get('window_id')} lacks detector availability")
    result = {str(value) for value in values}
    if len(result) != len(values):
        raise ValueError("background detector availability repeats an IFO")
    return result


def plan_candidate_background_exposure(
    background_windows: Iterable[dict[str, Any]],
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    target_far_per_year: float,
    zero_count_confidence: float = 0.90,
) -> dict[str, Any]:
    """Calculate exact candidate-slide exposure before expensive model scoring."""

    if reference_ifo == shifted_ifo:
        raise ValueError("exposure planning requires two different detectors")
    if slide_count <= 0 or step_seconds <= 0 or target_far_per_year <= 0:
        raise ValueError("slide count, step and target FAR must be positive")
    if not 0 < zero_count_confidence < 1:
        raise ValueError("zero-count confidence must be between zero and one")
    rows = [row for row in background_windows if str(row["split"]) == split]
    if not rows:
        raise ValueError(f"no background windows for split {split}")
    durations = {float(row["gps_end"]) - float(row["gps_start"]) for row in rows}
    if len(durations) != 1:
        raise ValueError("candidate exposure planner requires one window duration")
    duration = next(iter(durations))
    if step_seconds < duration:
        raise ValueError("candidate exposure slide step must span at least one window")
    by_start = {int(round(float(row["gps_start"]) * 1e9)): row for row in rows}
    if len(by_start) != len(rows):
        raise ValueError("candidate exposure manifest repeats GPS starts")
    availability = {str(row["window_id"]): _ifos(row) for row in rows}
    per_slide = []
    for index in range(1, slide_count + 1):
        offset = index * step_seconds
        offset_key = int(round(offset * 1e9))
        intervals = []
        for reference in rows:
            shifted = by_start.get(
                int(round(float(reference["gps_start"]) * 1e9)) + offset_key
            )
            if shifted is None:
                continue
            if (
                reference_ifo not in availability[str(reference["window_id"])]
                or shifted_ifo not in availability[str(shifted["window_id"])]
            ):
                continue
            intervals.append((float(reference["gps_start"]), float(reference["gps_end"])))
        exposure = _union_duration(intervals)
        per_slide.append(
            {
                "slide_index": index,
                "offset_seconds": offset,
                "paired_windows": len(intervals),
                "live_time_seconds": exposure,
            }
        )
    nonzero_slides = [row for row in per_slide if row["live_time_seconds"] > 0]
    equivalent_seconds = sum(row["live_time_seconds"] for row in nonzero_slides)
    equivalent_years = equivalent_seconds / SECONDS_PER_YEAR
    required_years_one_count = 1.0 / target_far_per_year
    required_years_zero_upper = -math.log(1.0 - zero_count_confidence) / target_far_per_year
    required_seconds_zero_upper = required_years_zero_upper * SECONDS_PER_YEAR
    minimum_windows_best_case = math.ceil(
        (1.0 + math.sqrt(1.0 + 8.0 * required_seconds_zero_upper / duration)) / 2.0
    )
    zero_lag_seconds = _union_duration(
        (float(row["gps_start"]), float(row["gps_end"])) for row in rows
    )
    available_reference_so_far = 0
    all_observed_pairs = 0
    for row in sorted(rows, key=lambda item: float(item["gps_start"])):
        row_ifos = availability[str(row["window_id"])]
        if shifted_ifo in row_ifos:
            all_observed_pairs += available_reference_so_far
        if reference_ifo in row_ifos:
            available_reference_so_far += 1
    all_observed_exposure_seconds = all_observed_pairs * duration
    all_observed_exposure_years = all_observed_exposure_seconds / SECONDS_PER_YEAR
    return {
        "status": "pre_scoring_candidate_background_exposure_plan",
        "scientific_claim_allowed": False,
        "split": split,
        "reference_ifo": reference_ifo,
        "shifted_ifo": shifted_ifo,
        "windows": len(rows),
        "gps_blocks": len({str(row["gps_block"]) for row in rows}),
        "window_duration_seconds": duration,
        "zero_lag_live_time_seconds": zero_lag_seconds,
        "zero_lag_live_time_days": zero_lag_seconds / 86400.0,
        "slide_count": slide_count,
        "step_seconds": step_seconds,
        "evaluated_offsets": len(per_slide),
        "offsets_with_nonzero_exposure": len(nonzero_slides),
        "offsets_with_zero_exposure": len(per_slide) - len(nonzero_slides),
        "nonzero_slide_exposure": nonzero_slides,
        "equivalent_live_time_seconds": equivalent_seconds,
        "equivalent_live_time_years": equivalent_years,
        "target_far_per_year": target_far_per_year,
        "target_ifar_years": 1.0 / target_far_per_year,
        "far_resolution_one_count_per_year": (
            1.0 / equivalent_years if equivalent_years > 0 else None
        ),
        "zero_count_confidence": zero_count_confidence,
        "zero_count_far_upper_per_year": (
            -math.log(1.0 - zero_count_confidence) / equivalent_years
            if equivalent_years > 0
            else None
        ),
        "required_equivalent_years_for_one_expected_count": required_years_one_count,
        "required_equivalent_years_for_zero_count_upper": required_years_zero_upper,
        "zero_count_target_exposure_fraction": equivalent_years / required_years_zero_upper,
        "zero_count_target_exposure_shortfall_factor": (
            required_years_zero_upper / equivalent_years
            if equivalent_years > 0
            else None
        ),
        "minimum_windows_best_case_all_unordered_pairs": minimum_windows_best_case,
        "minimum_zero_lag_days_best_case_all_unordered_pairs": (
            minimum_windows_best_case * duration / 86400.0
        ),
        "best_case_assumption": (
            "every unordered window pair is detector-valid, usable once as a non-cyclic shift, "
            "and no maximum-lag or segment-boundary restriction removes a pair"
        ),
        "all_observed_positive_lag_pairs": all_observed_pairs,
        "all_observed_positive_lag_exposure_seconds": all_observed_exposure_seconds,
        "all_observed_positive_lag_exposure_years": all_observed_exposure_years,
        "all_observed_positive_lag_zero_count_far_upper_per_year": (
            -math.log(1.0 - zero_count_confidence) / all_observed_exposure_years
            if all_observed_exposure_years > 0
            else None
        ),
        "all_observed_positive_lag_target_exposure_fraction": (
            all_observed_exposure_years / required_years_zero_upper
        ),
        "target_zero_count_upper_reached": equivalent_years >= required_years_zero_upper,
    }


def run_candidate_background_exposure_plan(
    background_manifest: str | Path,
    output: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    target_far_per_year: float,
    zero_count_confidence: float = 0.90,
) -> dict[str, Any]:
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = {
        **plan_candidate_background_exposure(
            rows,
            split,
            reference_ifo,
            shifted_ifo,
            slide_count,
            step_seconds,
            target_far_per_year,
            zero_count_confidence,
        ),
        "background_manifest_path": str(background_manifest),
        "background_manifest_sha256": file_sha256(background_manifest),
    }
    atomic_write_json(output, result)
    return result
