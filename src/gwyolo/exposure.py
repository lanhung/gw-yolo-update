from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .background import SECONDS_PER_YEAR, _union_duration
from .io import atomic_write_json, canonical_hash, file_sha256
from .runtime import execution_provenance


CANDIDATE_BLOCK_PERMUTATION_METHOD = (
    "circular_gps_block_relative_window_permutation_v1"
)
CANDIDATE_BLOCK_SELECTION_DATA = (
    "background_gps_blocks_and_detector_availability_only"
)


def _ifos(row: dict[str, Any]) -> set[str]:
    values = row.get("valid_ifos", row.get("ifos"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"background window {row.get('window_id')} lacks detector availability")
    result = {str(value) for value in values}
    if len(result) != len(values):
        raise ValueError("background detector availability repeats an IFO")
    return result


def normalize_candidate_slide_indices(
    slide_count: int,
    slide_start_index: int = 1,
    slide_indices: Iterable[int] | None = None,
) -> list[int]:
    if slide_count <= 0:
        raise ValueError("slide count must be positive")
    if slide_indices is None:
        if slide_start_index <= 0:
            raise ValueError("slide start index must be positive")
        return list(range(slide_start_index, slide_start_index + slide_count))
    indices = [int(value) for value in slide_indices]
    if len(indices) != slide_count:
        raise ValueError("explicit slide-index count differs from slide_count")
    if any(value <= 0 for value in indices):
        raise ValueError("slide indices must be positive")
    if len(set(indices)) != len(indices):
        raise ValueError("slide indices must be unique")
    if indices != sorted(indices):
        raise ValueError("slide indices must be strictly increasing")
    return indices


def candidate_slide_schedule_identity(schedule: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable fields covered by a frozen slide schedule ID."""

    identity = {
        field: schedule.get(field)
        for field in (
            "schema_version",
            "selection_rule",
            "split",
            "reference_ifo",
            "shifted_ifo",
            "step_seconds",
            "slide_indices",
            "background_manifest_sha256",
            "target_far_per_year",
            "zero_count_confidence",
        )
    }
    if int(schedule.get("schema_version", 1)) >= 2:
        identity["selection_metadata"] = schedule.get("selection_metadata")
    return identity


def candidate_block_schedule_identity(schedule: dict[str, Any]) -> dict[str, Any]:
    """Return immutable fields covered by a GPS-block permutation schedule ID."""

    return {
        field: schedule.get(field)
        for field in (
            "schema_version",
            "method",
            "background_manifest_sha256",
            "split",
            "reference_ifo",
            "shifted_ifo",
            "window_duration_seconds",
            "ordered_gps_blocks",
            "shift_indices",
            "target_far_per_year",
            "zero_count_confidence",
        )
    }


def plan_candidate_background_exposure(
    background_windows: Iterable[dict[str, Any]],
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    slide_count: int,
    step_seconds: float,
    target_far_per_year: float,
    zero_count_confidence: float = 0.90,
    slide_start_index: int = 1,
    slide_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Calculate exact candidate-slide exposure before expensive model scoring."""

    if reference_ifo == shifted_ifo:
        raise ValueError("exposure planning requires two different detectors")
    indices = normalize_candidate_slide_indices(
        slide_count, slide_start_index, slide_indices
    )
    if step_seconds <= 0 or target_far_per_year <= 0:
        raise ValueError("slide step and target FAR must be positive")
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
    for index in indices:
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
        "slide_start_index": min(indices),
        "slide_stop_index_exclusive": max(indices) + 1,
        "slide_indices_contiguous": indices
        == list(range(min(indices), max(indices) + 1)),
        "slide_indices_sha256": canonical_hash(indices, 64),
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
    slide_start_index: int = 1,
    slide_indices: Iterable[int] | None = None,
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
            slide_start_index,
            slide_indices,
        ),
        "background_manifest_path": str(background_manifest),
        "background_manifest_sha256": file_sha256(background_manifest),
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    return result


def freeze_candidate_time_slide_schedule(
    background_manifest: str | Path,
    output: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    step_seconds: float,
    slide_indices: Iterable[int],
    target_far_per_year: float,
    zero_count_confidence: float = 0.90,
    *,
    selection_rule: str = "explicit_nonzero_absolute_indices_v1",
    selection_metadata: dict[str, Any] | None = None,
    schema_version: int = 1,
) -> dict[str, Any]:
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("candidate time-slide schedules are immutable")
    indices = sorted(int(value) for value in slide_indices)
    if not indices:
        raise ValueError("candidate time-slide schedule requires explicit indices")
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    exposure = plan_candidate_background_exposure(
        rows,
        split,
        reference_ifo,
        shifted_ifo,
        len(indices),
        step_seconds,
        target_far_per_year,
        zero_count_confidence,
        min(indices),
        indices,
    )
    observed = {
        int(row["slide_index"]) for row in exposure["nonzero_slide_exposure"]
    }
    zero_exposure = [value for value in indices if value not in observed]
    if zero_exposure:
        raise ValueError(
            f"candidate time-slide schedule contains zero-exposure indices: {zero_exposure}"
        )
    background_hash = file_sha256(background_manifest)
    schedule_fields = {
        "schema_version": schema_version,
        "selection_rule": selection_rule,
        "split": split,
        "reference_ifo": reference_ifo,
        "shifted_ifo": shifted_ifo,
        "step_seconds": step_seconds,
        "slide_indices": indices,
        "background_manifest_sha256": background_hash,
        "target_far_per_year": target_far_per_year,
        "zero_count_confidence": zero_count_confidence,
    }
    if schema_version >= 2:
        if not isinstance(selection_metadata, dict) or not selection_metadata:
            raise ValueError("schema-v2 slide schedule requires selection metadata")
        schedule_fields["selection_metadata"] = selection_metadata
    identity = candidate_slide_schedule_identity(schedule_fields)
    result = {
        **schedule_fields,
        "status": "frozen_candidate_time_slide_schedule",
        "scientific_claim_allowed": False,
        "selection_data": "background_gps_and_detector_availability_only",
        "candidate_scores_inspected": False,
        "schedule_id": canonical_hash(identity, 32),
        "slide_count": len(indices),
        "slide_indices_sha256": canonical_hash(indices, 64),
        "background_manifest_path": str(Path(background_manifest).resolve()),
        "exposure_plan": exposure,
        "schedule_exposure_target_reached": exposure[
            "target_zero_count_upper_reached"
        ],
        "scientific_blocker": (
            "frozen validation-background execution schedule only; locked-test search claims "
            "remain unavailable"
            if exposure["target_zero_count_upper_reached"]
            else "the frozen schedule does not reach the predeclared zero-count FAR exposure"
        ),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result


def freeze_candidate_time_slide_range_schedule(
    background_manifest: str | Path,
    output: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    step_seconds: float,
    slide_start_index: int,
    slide_stop_index_exclusive: int,
    target_far_per_year: float,
    zero_count_confidence: float = 0.90,
) -> dict[str, Any]:
    """Freeze the shortest nonzero-offset prefix meeting a predeclared FAR exposure."""

    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("candidate time-slide schedules are immutable")
    if slide_start_index <= 0 or slide_stop_index_exclusive <= slide_start_index:
        raise ValueError("slide range must be a non-empty positive half-open interval")
    indices = list(range(slide_start_index, slide_stop_index_exclusive))
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    scanned = plan_candidate_background_exposure(
        rows,
        split,
        reference_ifo,
        shifted_ifo,
        len(indices),
        step_seconds,
        target_far_per_year,
        zero_count_confidence,
        slide_start_index,
        indices,
    )
    nonzero = list(scanned["nonzero_slide_exposure"])
    if not nonzero:
        raise ValueError("slide range contains no nonzero detector-coincident exposure")
    required_seconds = (
        float(scanned["required_equivalent_years_for_zero_count_upper"])
        * SECONDS_PER_YEAR
    )
    selected: list[int] = []
    selected_seconds = 0.0
    for row in nonzero:
        selected.append(int(row["slide_index"]))
        selected_seconds += float(row["live_time_seconds"])
        if selected_seconds >= required_seconds:
            break
    selection_metadata = {
        "slide_start_index": slide_start_index,
        "slide_stop_index_exclusive": slide_stop_index_exclusive,
        "evaluated_offsets": len(indices),
        "nonzero_offsets_available": len(nonzero),
        "available_equivalent_live_time_seconds": float(
            scanned["equivalent_live_time_seconds"]
        ),
        "required_equivalent_live_time_seconds": required_seconds,
        "available_range_reaches_target": bool(
            scanned["target_zero_count_upper_reached"]
        ),
        "selected_nonzero_prefix_count": len(selected),
        "selected_equivalent_live_time_seconds": selected_seconds,
        "candidate_scores_inspected": False,
    }
    return freeze_candidate_time_slide_schedule(
        background_manifest,
        target,
        split,
        reference_ifo,
        shifted_ifo,
        step_seconds,
        selected,
        target_far_per_year,
        zero_count_confidence,
        selection_rule="nonzero_prefix_to_zero_count_target_within_range_v1",
        selection_metadata=selection_metadata,
        schema_version=2,
    )


def freeze_candidate_block_permutation_schedule(
    background_manifest: str | Path,
    output: str | Path,
    split: str,
    reference_ifo: str,
    shifted_ifo: str,
    target_far_per_year: float,
    zero_count_confidence: float = 0.90,
    maximum_shifts: int | None = None,
) -> dict[str, Any]:
    """Freeze circular GPS-block permutations using relative window positions."""

    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("candidate block-permutation schedules are immutable")
    if reference_ifo == shifted_ifo or target_far_per_year <= 0:
        raise ValueError(
            "block permutation requires two IFOs and a positive FAR target"
        )
    if not 0 < zero_count_confidence < 1:
        raise ValueError("zero-count confidence must be between zero and one")
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if str(row.get("split")) == split]
    if not rows:
        raise ValueError(f"no background windows for split {split}")
    durations = {float(row["gps_end"]) - float(row["gps_start"]) for row in rows}
    if len(durations) != 1:
        raise ValueError("block permutation requires one common window duration")
    window_duration = next(iter(durations))
    if window_duration <= 0:
        raise ValueError("block permutation window duration must be positive")

    blocks: dict[str, dict[str, Any]] = {}
    for row in rows:
        block_id = str(row["gps_block"])
        parts = block_id.split(":")
        if len(parts) != 3 or parts[0] != "gps":
            raise ValueError(f"unsupported GPS block identity: {block_id}")
        block_start = float(parts[1])
        block_duration = float(parts[2])
        offset = (float(row["gps_start"]) - block_start) / window_duration
        slot = int(round(offset))
        if not math.isclose(offset, slot, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("background window is not aligned to its GPS block")
        if slot < 0 or (slot + 1) * window_duration > block_duration + 1e-9:
            raise ValueError("background window lies outside its GPS block")
        record = blocks.setdefault(
            block_id,
            {
                "gps_start": block_start,
                "duration": block_duration,
                "slots_by_ifo": {},
            },
        )
        if record["gps_start"] != block_start or record["duration"] != block_duration:
            raise ValueError("GPS block metadata is inconsistent")
        for ifo in _ifos(row):
            slots = record["slots_by_ifo"].setdefault(ifo, set())
            if slot in slots:
                raise ValueError(
                    f"GPS block {block_id} repeats detector slot {ifo}:{slot}"
                )
            slots.add(slot)
    ordered = sorted(blocks, key=lambda value: (blocks[value]["gps_start"], value))
    if len(ordered) < 2:
        raise ValueError("block permutation requires at least two GPS blocks")
    available_shifts = len(ordered) - 1
    if maximum_shifts is None:
        maximum_shifts = available_shifts
    if maximum_shifts <= 0 or maximum_shifts > available_shifts:
        raise ValueError("maximum block shifts exceeds the nonzero circular range")
    required_seconds = (
        -math.log(1.0 - zero_count_confidence) / target_far_per_year * SECONDS_PER_YEAR
    )
    selected = []
    total_seconds = 0.0
    for shift in range(1, maximum_shifts + 1):
        paired_windows = 0
        paired_blocks = 0
        for index, reference_block in enumerate(ordered):
            shifted_block = ordered[(index + shift) % len(ordered)]
            reference_slots = blocks[reference_block]["slots_by_ifo"].get(
                reference_ifo, set()
            )
            shifted_slots = blocks[shifted_block]["slots_by_ifo"].get(
                shifted_ifo, set()
            )
            overlap = len(reference_slots & shifted_slots)
            if overlap:
                paired_blocks += 1
                paired_windows += overlap
        exposure = paired_windows * window_duration
        if exposure <= 0:
            continue
        selected.append(
            {
                "shift_index": shift,
                "paired_blocks": paired_blocks,
                "paired_windows": paired_windows,
                "live_time_seconds": exposure,
            }
        )
        total_seconds += exposure
        if total_seconds >= required_seconds:
            break
    shift_indices = [row["shift_index"] for row in selected]
    reached = total_seconds >= required_seconds
    schedule_fields = {
        "schema_version": 1,
        "method": CANDIDATE_BLOCK_PERMUTATION_METHOD,
        "background_manifest_sha256": file_sha256(background_manifest),
        "split": split,
        "reference_ifo": reference_ifo,
        "shifted_ifo": shifted_ifo,
        "window_duration_seconds": window_duration,
        "ordered_gps_blocks": ordered,
        "shift_indices": shift_indices,
        "target_far_per_year": target_far_per_year,
        "zero_count_confidence": zero_count_confidence,
    }
    result = {
        **schedule_fields,
        "status": "frozen_candidate_block_permutation_schedule",
        "scientific_claim_allowed": False,
        "selection_data": CANDIDATE_BLOCK_SELECTION_DATA,
        "candidate_scores_inspected": False,
        "schedule_id": canonical_hash(
            candidate_block_schedule_identity(schedule_fields), 32
        ),
        "gps_blocks": len(ordered),
        "available_nonzero_circular_shifts": available_shifts,
        "maximum_shifts_scanned": maximum_shifts,
        "selected_shifts": selected,
        "selected_shift_count": len(selected),
        "shift_indices_sha256": canonical_hash(shift_indices, 64),
        "selected_equivalent_live_time_seconds": total_seconds,
        "selected_equivalent_live_time_years": total_seconds / SECONDS_PER_YEAR,
        "required_equivalent_live_time_seconds": required_seconds,
        "required_equivalent_live_time_years": required_seconds / SECONDS_PER_YEAR,
        "schedule_exposure_target_reached": reached,
        "far_resolution_one_count_per_year": (
            SECONDS_PER_YEAR / total_seconds if total_seconds > 0 else None
        ),
        "scientific_blocker": (
            "validation-only schedule; execution and a separate locked test remain required"
            if reached
            else "available GPS-block permutations do not reach the frozen FAR exposure target"
        ),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result


def forecast_candidate_block_permutation_capacity(
    pilot_schedule_path: str | Path,
    pilot_background_report_path: str | Path,
    planned_parent_plan_path: str | Path,
    safety_factor: float = 1.5,
) -> dict[str, Any]:
    """Forecast score-blind block-permutation capacity from a DQ-verified pilot."""

    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("candidate background capacity safety factor must be at least one")
    with Path(pilot_schedule_path).open("r", encoding="utf-8") as handle:
        schedule = json.load(handle)
    with Path(pilot_background_report_path).open("r", encoding="utf-8") as handle:
        background = json.load(handle)
    with Path(planned_parent_plan_path).open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if (
        schedule.get("status") != "frozen_candidate_block_permutation_schedule"
        or schedule.get("selection_data") != CANDIDATE_BLOCK_SELECTION_DATA
        or schedule.get("candidate_scores_inspected") is not False
        or schedule.get("split") != "val"
    ):
        raise ValueError("pilot block schedule is not a score-blind validation schedule")
    if (
        background.get("status") != "verified_multi_segment_development_background"
        or background.get("scientific_claim_allowed") is not False
    ):
        raise ValueError("pilot background report is not a verified development background")
    manifest = Path(str(background.get("manifest_path", ""))).resolve()
    if (
        not manifest.is_file()
        or file_sha256(manifest) != background.get("manifest_sha256")
        or schedule.get("background_manifest_sha256") != background.get("manifest_sha256")
    ):
        raise ValueError("pilot schedule and background manifest hashes differ")
    if (
        plan.get("status") != "development_acquisition_plan"
        or plan.get("locked_evaluation_data") is not False
        or not isinstance(plan.get("pairs"), list)
        or int(plan.get("selected_pairs", -1)) != len(plan["pairs"])
    ):
        raise ValueError("planned parent acquisition is not a complete development plan")
    pair_ids = [str(row.get("pair_id", "")) for row in plan["pairs"]]
    if any(not pair_id for pair_id in pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("planned parent acquisition pair IDs must be nonempty and unique")
    source_pairs = int(background.get("source_pairs", 0))
    pilot_blocks = int(schedule.get("gps_blocks", 0))
    selected_shifts = int(schedule.get("selected_shift_count", 0))
    pilot_seconds = float(schedule.get("selected_equivalent_live_time_seconds", 0))
    required_seconds = float(schedule.get("required_equivalent_live_time_seconds", 0))
    if (
        source_pairs <= 0
        or pilot_blocks < 2
        or selected_shifts <= 0
        or pilot_seconds <= 0
        or required_seconds <= 0
    ):
        raise ValueError("pilot schedule lacks positive pair, block, shift or exposure counts")

    blocks_per_source_pair = pilot_blocks / source_pairs
    seconds_per_block_shift = pilot_seconds / (pilot_blocks * selected_shifts)
    planned_pairs = len(plan["pairs"])
    projected_blocks = math.floor(blocks_per_source_pair * planned_pairs)
    projected_shifts = max(0, projected_blocks - 1)
    projected_seconds = (
        seconds_per_block_shift * projected_blocks * projected_shifts
    )
    safety_required_seconds = required_seconds * safety_factor
    block_discriminant = 1 + 4 * safety_required_seconds / seconds_per_block_shift
    recommended_blocks = math.ceil((1 + math.sqrt(block_discriminant)) / 2)
    recommended_pairs = math.ceil(recommended_blocks / blocks_per_source_pair)
    aligned_available = int(plan.get("aligned_pairs_available", planned_pairs))
    safe = projected_seconds >= safety_required_seconds
    feasible = recommended_pairs <= aligned_available
    if safe:
        blocker = (
            "forecast passed; exact post-DQ schedule and locked test are still required"
        )
    elif feasible:
        blocker = "planned acquisition lacks the predeclared forecast safety margin"
    else:
        blocker = (
            "available aligned source pairs cannot meet the predeclared forecast "
            "safety margin"
        )
    return {
        "status": "score_blind_candidate_block_capacity_forecast",
        "scientific_claim_allowed": False,
        "forecast_only": True,
        "candidate_scores_inspected": False,
        "selection_data": CANDIDATE_BLOCK_SELECTION_DATA,
        "projection_assumption": (
            "linear validation-block yield per selected source pair and constant "
            "pilot-average seconds per block-shift"
        ),
        "pilot_schedule_path": str(Path(pilot_schedule_path).resolve()),
        "pilot_schedule_sha256": file_sha256(pilot_schedule_path),
        "pilot_background_report_path": str(
            Path(pilot_background_report_path).resolve()
        ),
        "pilot_background_report_sha256": file_sha256(
            pilot_background_report_path
        ),
        "planned_parent_plan_path": str(Path(planned_parent_plan_path).resolve()),
        "planned_parent_plan_sha256": file_sha256(planned_parent_plan_path),
        "safety_factor": safety_factor,
        "target_far_per_year": float(schedule["target_far_per_year"]),
        "zero_count_confidence": float(schedule["zero_count_confidence"]),
        "pilot_source_pairs": source_pairs,
        "pilot_gps_blocks": pilot_blocks,
        "pilot_selected_shifts": selected_shifts,
        "pilot_equivalent_live_time_seconds": pilot_seconds,
        "observed_blocks_per_source_pair": blocks_per_source_pair,
        "observed_seconds_per_block_shift": seconds_per_block_shift,
        "planned_source_pairs": planned_pairs,
        "aligned_pairs_available": aligned_available,
        "projected_gps_blocks": projected_blocks,
        "projected_available_nonzero_shifts": projected_shifts,
        "projected_maximum_equivalent_live_time_seconds": projected_seconds,
        "required_equivalent_live_time_seconds": required_seconds,
        "safety_required_equivalent_live_time_seconds": safety_required_seconds,
        "projected_to_required_ratio": projected_seconds / required_seconds,
        "projected_to_safety_required_ratio": (
            projected_seconds / safety_required_seconds
        ),
        "recommended_minimum_gps_blocks": recommended_blocks,
        "recommended_minimum_source_pairs": recommended_pairs,
        "recommendation_fits_available_pairs": feasible,
        "planned_pairs_satisfy_safety_forecast": safe,
        "scientific_blocker": blocker,
        **execution_provenance(),
    }


def run_candidate_block_permutation_capacity_forecast(
    pilot_schedule_path: str | Path,
    pilot_background_report_path: str | Path,
    planned_parent_plan_path: str | Path,
    output_path: str | Path,
    safety_factor: float = 1.5,
    allow_insufficient: bool = False,
) -> dict[str, Any]:
    report = forecast_candidate_block_permutation_capacity(
        pilot_schedule_path,
        pilot_background_report_path,
        planned_parent_plan_path,
        safety_factor,
    )
    atomic_write_json(output_path, report)
    if not report["planned_pairs_satisfy_safety_forecast"] and not allow_insufficient:
        raise RuntimeError(
            f"candidate block capacity forecast failed; inspect {output_path}"
        )
    return report


def freeze_candidate_block_capacity_extension_decision(
    base_forecast_path: str | Path,
    extended_plan_path: str | Path,
    extended_forecast_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze why a score-blind parent plan was extended before candidate scoring."""

    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("candidate background capacity decisions are immutable")
    inputs = {}
    for name, path_value in (
        ("base_forecast", base_forecast_path),
        ("extended_plan", extended_plan_path),
        ("extended_forecast", extended_forecast_path),
    ):
        path = Path(path_value).resolve()
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        inputs[name] = (path, value)
    base_path, base = inputs["base_forecast"]
    plan_path, plan = inputs["extended_plan"]
    extended_path, extended = inputs["extended_forecast"]
    if (
        base.get("status") != "score_blind_candidate_block_capacity_forecast"
        or base.get("candidate_scores_inspected") is not False
        or base.get("planned_pairs_satisfy_safety_forecast") is not False
    ):
        raise ValueError("base capacity forecast is not a score-blind failed safety gate")
    if (
        plan.get("status") != "development_acquisition_plan"
        or plan.get("locked_evaluation_data") is not False
        or plan.get("selection_rule") != "frozen_prefix_stratified_complement_v1"
        or plan.get("candidate_scores_inspected") is not False
    ):
        raise ValueError("extended plan is not a score-blind frozen-prefix extension")
    pairs = list(plan.get("pairs", []))
    pair_ids = [str(row.get("pair_id", "")) for row in pairs]
    base_count = int(plan.get("base_selected_pairs", 0))
    if (
        not pairs
        or len(pair_ids) != len(set(pair_ids))
        or any(not pair_id for pair_id in pair_ids)
        or int(plan.get("selected_pairs", -1)) != len(pairs)
        or int(plan.get("extension_pairs", -1)) != len(pairs) - base_count
        or canonical_hash(pair_ids[:base_count], 64)
        != str(plan.get("base_pair_ids_hash", ""))
        or str(plan.get("base_parent_plan_sha256", ""))
        != str(base.get("planned_parent_plan_sha256", ""))
        or base_count != int(base.get("planned_source_pairs", -1))
    ):
        raise ValueError("extended plan does not preserve the failed forecast parent exactly")
    if (
        extended.get("status") != "score_blind_candidate_block_capacity_forecast"
        or extended.get("candidate_scores_inspected") is not False
        or extended.get("planned_pairs_satisfy_safety_forecast") is not True
        or str(extended.get("planned_parent_plan_sha256", ""))
        != file_sha256(plan_path)
        or int(extended.get("planned_source_pairs", -1)) != len(pairs)
    ):
        raise ValueError("extended capacity forecast does not pass for the exact extended plan")
    common_fields = (
        "pilot_schedule_sha256",
        "pilot_background_report_sha256",
        "safety_factor",
        "target_far_per_year",
        "zero_count_confidence",
        "required_equivalent_live_time_seconds",
    )
    if any(base.get(field) != extended.get(field) for field in common_fields):
        raise ValueError("base and extended capacity forecasts changed the frozen target")
    recommended_pairs = int(base.get("recommended_minimum_source_pairs", 0))
    if recommended_pairs <= base_count or recommended_pairs != len(pairs):
        raise ValueError("extended plan is not the failed forecast's exact minimum recommendation")
    result = {
        "status": "frozen_score_blind_background_capacity_extension_decision",
        "scientific_claim_allowed": False,
        "test_data_opened": False,
        "candidate_scores_inspected": False,
        "selection_data": CANDIDATE_BLOCK_SELECTION_DATA,
        "decision": "freeze_extended_parent_for_validation_background",
        "base_forecast_path": str(base_path),
        "base_forecast_sha256": file_sha256(base_path),
        "base_parent_plan_sha256": str(base["planned_parent_plan_sha256"]),
        "base_source_pairs": base_count,
        "extended_plan_path": str(plan_path),
        "extended_plan_sha256": file_sha256(plan_path),
        "extended_source_pairs": len(pairs),
        "extension_source_pairs": len(pairs) - base_count,
        "extended_forecast_path": str(extended_path),
        "extended_forecast_sha256": file_sha256(extended_path),
        "recommended_minimum_source_pairs": recommended_pairs,
        "safety_factor": float(extended["safety_factor"]),
        "target_far_per_year": float(extended["target_far_per_year"]),
        "zero_count_confidence": float(extended["zero_count_confidence"]),
        "projected_to_safety_required_ratio": float(
            extended["projected_to_safety_required_ratio"]
        ),
        "scientific_blocker": (
            "exact post-DQ schedule, frozen validation threshold and locked test remain required"
        ),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result
