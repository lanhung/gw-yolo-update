from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .background import SECONDS_PER_YEAR
from .candidates import _available_ifos
from .exposure import (
    CANDIDATE_BLOCK_PERMUTATION_METHOD,
    CANDIDATE_BLOCK_SELECTION_DATA,
    DETECTOR_SET_BLOCK_PERMUTATION_METHOD,
    DETECTOR_SET_BLOCK_SELECTION_DATA,
    candidate_block_schedule_identity,
    detector_set_block_schedule_identity,
)
from .io import canonical_hash, file_sha256


def _effective_units(weights: np.ndarray) -> float:
    total = float(weights.sum())
    squared = float(np.square(weights).sum())
    return total * total / squared if squared > 0 else 0.0


def _cluster_bootstrap_rates(
    exposure: np.ndarray,
    exceedances: np.ndarray,
    shifted_indices: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Three-way pigeonhole bootstrap over both physical blocks and shifts."""

    block_count, shift_count = exposure.shape
    rng = np.random.default_rng(seed)
    probabilities_blocks = np.full(block_count, 1.0 / block_count)
    probabilities_shifts = np.full(shift_count, 1.0 / shift_count)
    values: list[np.ndarray] = []
    remaining = replicates
    attempts = 0
    while remaining:
        attempts += 1
        if attempts > 1000:
            break
        batch = min(64, max(remaining, 8))
        block_weights = rng.multinomial(
            block_count, probabilities_blocks, size=batch
        ).astype(np.float64)
        shift_weights = rng.multinomial(
            shift_count, probabilities_shifts, size=batch
        ).astype(np.float64)
        cell_weights = (
            block_weights[:, :, None]
            * block_weights[:, shifted_indices]
            * shift_weights[:, None, :]
        )
        denominators = np.sum(cell_weights * exposure[None, :, :], axis=(1, 2))
        numerators = np.sum(cell_weights * exceedances[None, :, :], axis=(1, 2))
        valid = denominators > 0
        sampled = numerators[valid] / denominators[valid] * SECONDS_PER_YEAR
        if len(sampled):
            accepted = sampled[:remaining]
            values.append(accepted)
            remaining -= len(accepted)
    return np.concatenate(values) if values else np.asarray([], dtype=np.float64)


def _multiway_cluster_bootstrap_rates(
    exposure: np.ndarray,
    exceedances: np.ndarray,
    dependency_block_indices: np.ndarray,
    shift_positions: np.ndarray,
    block_count: int,
    shift_count: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Pigeonhole bootstrap over up to three source blocks and one shift."""

    rng = np.random.default_rng(seed)
    block_probabilities = np.full(block_count, 1.0 / block_count)
    shift_probabilities = np.full(shift_count, 1.0 / shift_count)
    values: list[np.ndarray] = []
    remaining = replicates
    attempts = 0
    while remaining:
        attempts += 1
        if attempts > 1000:
            break
        batch = min(32, max(remaining, 8))
        block_weights = rng.multinomial(
            block_count,
            block_probabilities,
            size=batch,
        ).astype(np.float64)
        shift_weights = rng.multinomial(
            shift_count,
            shift_probabilities,
            size=batch,
        ).astype(np.float64)
        cell_weights = shift_weights[:, shift_positions].copy()
        for column in range(dependency_block_indices.shape[1]):
            indices = dependency_block_indices[:, column]
            valid = indices >= 0
            if np.any(valid):
                cell_weights[:, valid] *= block_weights[
                    :,
                    indices[valid],
                ]
        denominators = np.sum(
            cell_weights * exposure[None, :],
            axis=1,
        )
        numerators = np.sum(
            cell_weights * exceedances[None, :],
            axis=1,
        )
        valid = denominators > 0
        sampled = (
            numerators[valid]
            / denominators[valid]
            * SECONDS_PER_YEAR
        )
        if len(sampled):
            accepted = sampled[:remaining]
            values.append(accepted)
            remaining -= len(accepted)
    return (
        np.concatenate(values)
        if values
        else np.asarray([], dtype=np.float64)
    )


def audit_candidate_background_dependence(
    time_slide_report_path: str | Path,
    background_manifest_path: str | Path,
    threshold: float,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260722,
    minimum_gps_blocks: int = 25,
    minimum_shifts: int = 25,
    minimum_effective_gps_blocks: float = 20.0,
    minimum_effective_shifts: float = 20.0,
    maximum_exposure_fraction_per_gps_block: float = 0.10,
    maximum_exposure_fraction_per_shift: float = 0.10,
) -> dict[str, Any]:
    """Audit FAR dependence using physical blocks and frozen permutation offsets.

    The equivalent live time and point FAR retain the standard time-slide
    interpretation.  The bootstrap is a dependence sensitivity analysis: it
    resamples the reference-block, shifted-block and offset clusters together,
    rather than treating candidate rows as independent observations.
    """

    numeric_values = (
        threshold,
        minimum_effective_gps_blocks,
        minimum_effective_shifts,
        maximum_exposure_fraction_per_gps_block,
        maximum_exposure_fraction_per_shift,
    )
    if any(not math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("background dependence settings must be finite")
    if bootstrap_replicates < 1 or minimum_gps_blocks < 2 or minimum_shifts < 1:
        raise ValueError("background dependence count settings are invalid")
    if min(minimum_effective_gps_blocks, minimum_effective_shifts) <= 0:
        raise ValueError("background dependence effective-unit minima must be positive")
    if not 0 < maximum_exposure_fraction_per_gps_block <= 1:
        raise ValueError("GPS-block exposure fraction limit must lie in (0, 1]")
    if not 0 < maximum_exposure_fraction_per_shift <= 1:
        raise ValueError("shift exposure fraction limit must lie in (0, 1]")

    report_path = Path(time_slide_report_path).resolve()
    background_path = Path(background_manifest_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate_path = Path(str(report.get("manifest_path", ""))).resolve()
    schedule_path = Path(str(report.get("slide_schedule_path", ""))).resolve()
    if (
        report.get("status") != "subwindow_clustered_time_slide_integration_only"
        or report.get("background_pairing_method")
        != CANDIDATE_BLOCK_PERMUTATION_METHOD
        or not candidate_path.is_file()
        or report.get("manifest_sha256") != file_sha256(candidate_path)
        or not schedule_path.is_file()
        or report.get("slide_schedule_sha256") != file_sha256(schedule_path)
        or not background_path.is_file()
        or report.get("background_manifest_sha256") != file_sha256(background_path)
    ):
        raise ValueError("background dependence audit requires a verified block report")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if (
        schedule.get("status") != "frozen_candidate_block_permutation_schedule"
        or schedule.get("method") != CANDIDATE_BLOCK_PERMUTATION_METHOD
        or schedule.get("selection_data") != CANDIDATE_BLOCK_SELECTION_DATA
        or schedule.get("candidate_scores_inspected") is not False
        or schedule.get("background_manifest_sha256") != file_sha256(background_path)
        or canonical_hash(candidate_block_schedule_identity(schedule), 32)
        != schedule.get("schedule_id")
        or schedule.get("schedule_id") != report.get("slide_schedule_id")
    ):
        raise ValueError("background dependence frozen schedule failed replay")

    split = str(report.get("split"))
    reference_ifo = str(report.get("reference_ifo"))
    shifted_ifo = str(report.get("shifted_ifo"))
    ordered = [str(value) for value in schedule.get("ordered_gps_blocks", [])]
    shifts = [int(value) for value in schedule.get("shift_indices", [])]
    if (
        len(ordered) < 2
        or len(set(ordered)) != len(ordered)
        or not shifts
        or shifts != sorted(set(shifts))
        or report.get("input_gps_blocks") != ordered
        or [int(value) for value in report.get("slide_indices", [])] != shifts
    ):
        raise ValueError("background dependence schedule inventory differs")

    window_duration = float(schedule["window_duration_seconds"])
    slots: dict[str, dict[str, set[int]]] = {block: {} for block in ordered}
    with background_path.open("r", encoding="utf-8") as handle:
        background_rows = [json.loads(line) for line in handle if line.strip()]
    for row in background_rows:
        if str(row.get("split")) != split:
            continue
        block = str(row.get("gps_block"))
        if block not in slots:
            raise ValueError("background manifest contains an unscheduled GPS block")
        parts = block.split(":")
        if len(parts) != 3 or parts[0] != "gps":
            raise ValueError("background dependence requires canonical GPS blocks")
        offset = (float(row["gps_start"]) - float(parts[1])) / window_duration
        slot = int(round(offset))
        if not np.isclose(offset, slot, rtol=0.0, atol=1e-6):
            raise ValueError("background window is not aligned to its GPS block")
        for ifo in _available_ifos(row):
            if slot in slots[block].setdefault(ifo, set()):
                raise ValueError("background manifest repeats an IFO block slot")
            slots[block][ifo].add(slot)

    block_count = len(ordered)
    shift_count = len(shifts)
    shifted_indices = np.empty((block_count, shift_count), dtype=np.int64)
    exposure = np.zeros((block_count, shift_count), dtype=np.float64)
    for block_index, reference_block in enumerate(ordered):
        for shift_position, shift in enumerate(shifts):
            shifted_index = (block_index + shift) % block_count
            shifted_indices[block_index, shift_position] = shifted_index
            shifted_block = ordered[shifted_index]
            common = slots[reference_block].get(reference_ifo, set()) & slots[
                shifted_block
            ].get(shifted_ifo, set())
            exposure[block_index, shift_position] = len(common) * window_duration

    expected_by_shift = {
        int(row["slide_index"]): float(row["live_time_seconds"])
        for row in report.get("slide_exposure", [])
    }
    if set(expected_by_shift) != set(shifts):
        raise ValueError("background dependence report has incomplete shift exposure")
    observed_by_shift = exposure.sum(axis=0)
    if any(
        not np.isclose(
            observed_by_shift[position], expected_by_shift[shift], rtol=0.0, atol=1e-9
        )
        for position, shift in enumerate(shifts)
    ):
        raise ValueError("reconstructed block exposure differs from the executed report")

    exceedances = np.zeros_like(exposure)
    candidate_rows = 0
    false_alarms = 0
    shift_positions = {shift: position for position, shift in enumerate(shifts)}
    block_positions = {block: position for position, block in enumerate(ordered)}
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_rows += 1
            shift = int(row.get("slide_index", -1))
            if shift not in shift_positions:
                raise ValueError("background candidate uses an unscheduled shift")
            sources = row.get("source_gps_blocks")
            if not isinstance(sources, dict):
                raise ValueError("background candidate lacks source GPS blocks")
            reference_block = str(sources.get(reference_ifo))
            shifted_block = str(sources.get(shifted_ifo))
            if reference_block not in block_positions:
                raise ValueError("background candidate uses an unknown reference block")
            block_position = block_positions[reference_block]
            shift_position = shift_positions[shift]
            expected_shifted = ordered[shifted_indices[block_position, shift_position]]
            if shifted_block != expected_shifted:
                raise ValueError("background candidate block pair differs from its shift")
            score = float(row["ranking_score"])
            if not math.isfinite(score):
                raise ValueError("background candidate score must be finite")
            if score >= threshold:
                exceedances[block_position, shift_position] += 1
                false_alarms += 1
    if candidate_rows != int(report.get("background_rows", -1)):
        raise ValueError("background dependence candidate count differs from report")

    total_seconds = float(exposure.sum())
    report_seconds = float(report.get("equivalent_live_time_seconds", -1))
    if total_seconds <= 0 or not np.isclose(
        total_seconds, report_seconds, rtol=0.0, atol=1e-9
    ):
        raise ValueError("background dependence total exposure differs from report")
    live_time_years = total_seconds / SECONDS_PER_YEAR
    far_per_year = false_alarms / live_time_years
    zero_upper = (
        -math.log(0.1) / live_time_years if false_alarms == 0 else None
    )

    shift_exposure = exposure.sum(axis=0)
    block_exposure = np.zeros(block_count, dtype=np.float64)
    for block_position in range(block_count):
        block_exposure[block_position] += exposure[block_position].sum() / 2
    for block_position in range(block_count):
        block_exposure[block_position] += (
            exposure[shifted_indices == block_position].sum() / 2
        )
    effective_blocks = _effective_units(block_exposure)
    effective_shifts = _effective_units(shift_exposure)
    block_fraction = float(block_exposure.max() / block_exposure.sum())
    shift_fraction = float(shift_exposure.max() / shift_exposure.sum())

    bootstrap = _cluster_bootstrap_rates(
        exposure,
        exceedances,
        shifted_indices,
        bootstrap_replicates,
        seed,
    )
    bootstrap_valid = len(bootstrap) == bootstrap_replicates and bool(
        np.all(np.isfinite(bootstrap))
    )
    bootstrap_interval = (
        [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))]
        if bootstrap_valid
        else None
    )

    shift_leave_one_out = []
    for position, shift in enumerate(shifts):
        seconds = total_seconds - float(exposure[:, position].sum())
        count = false_alarms - int(exceedances[:, position].sum())
        shift_leave_one_out.append(
            {"shift_index": shift, "far_per_year": count / (seconds / SECONDS_PER_YEAR)}
        )
    block_leave_one_out = []
    for block_position, block in enumerate(ordered):
        keep = np.ones_like(exposure, dtype=bool)
        keep[block_position, :] = False
        keep[shifted_indices == block_position] = False
        seconds = float(exposure[keep].sum())
        count = int(exceedances[keep].sum())
        block_leave_one_out.append(
            {"gps_block": block, "far_per_year": count / (seconds / SECONDS_PER_YEAR)}
        )

    gates = {
        "frozen_score_blind_schedule_replayed": True,
        "candidate_rows_assigned_to_physical_clusters": True,
        "equivalent_exposure_reconstructed": True,
        "minimum_unique_gps_blocks": block_count >= minimum_gps_blocks,
        "minimum_unique_shifts": shift_count >= minimum_shifts,
        "minimum_effective_gps_blocks": effective_blocks >= minimum_effective_gps_blocks,
        "minimum_effective_shifts": effective_shifts >= minimum_effective_shifts,
        "gps_block_exposure_not_dominated": (
            block_fraction <= maximum_exposure_fraction_per_gps_block
        ),
        "shift_exposure_not_dominated": shift_fraction <= maximum_exposure_fraction_per_shift,
        "cluster_bootstrap_complete": bootstrap_valid,
    }
    return {
        "status": "candidate_background_dependence_audit_v1",
        "passed": all(gates.values()),
        "scientific_claim_allowed": False,
        "interpretation": (
            "standard equivalent time-slide exposure with a three-way cluster dependence "
            "sensitivity analysis; candidate rows are never bootstrapped as IID"
        ),
        "split": split,
        "threshold": threshold,
        "time_slide_report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "background_manifest": {
            "path": str(background_path),
            "sha256": file_sha256(background_path),
        },
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": file_sha256(schedule_path),
            "schedule_id": schedule["schedule_id"],
        },
        "unique_gps_blocks": block_count,
        "unique_shifts": shift_count,
        "effective_gps_blocks": effective_blocks,
        "effective_shifts": effective_shifts,
        "maximum_exposure_fraction_per_gps_block": block_fraction,
        "maximum_exposure_fraction_per_shift": shift_fraction,
        "live_time_seconds": total_seconds,
        "live_time_years": live_time_years,
        "candidate_rows": candidate_rows,
        "false_alarms": false_alarms,
        "far_per_year": far_per_year,
        "ifar_years": 1.0 / far_per_year if far_per_year > 0 else None,
        "far_90_upper_limit_if_zero": zero_upper,
        "three_way_cluster_bootstrap": {
            "method": "physical_reference_block_x_shifted_block_x_offset_pigeonhole_v1",
            "replicates": bootstrap_replicates,
            "seed": seed,
            "far_per_year_95": bootstrap_interval,
            "informative_for_rate_variation": false_alarms > 0,
            "zero_count_policy": (
                "report Poisson 90% upper limit; zero-only resampling cannot estimate event rate"
                if false_alarms == 0
                else None
            ),
        },
        "leave_one_shift_out_far_range": [
            min(row["far_per_year"] for row in shift_leave_one_out),
            max(row["far_per_year"] for row in shift_leave_one_out),
        ],
        "leave_one_gps_block_out_far_range": [
            min(row["far_per_year"] for row in block_leave_one_out),
            max(row["far_per_year"] for row in block_leave_one_out),
        ],
        "gates": gates,
    }


def audit_detector_set_candidate_background_dependence(
    time_slide_report_path: str | Path,
    background_manifest_path: str | Path,
    threshold: float,
    bootstrap_replicates: int = 10_000,
    seed: int = 20260722,
    minimum_gps_blocks: int = 25,
    minimum_shifts: int = 25,
    minimum_effective_gps_blocks: float = 20.0,
    minimum_effective_shifts: float = 20.0,
    maximum_exposure_fraction_per_gps_block: float = 0.10,
    maximum_exposure_fraction_per_shift: float = 0.10,
) -> dict[str, Any]:
    """Audit variable-detector FAR dependence over source blocks and shifts."""

    numeric_values = (
        threshold,
        minimum_effective_gps_blocks,
        minimum_effective_shifts,
        maximum_exposure_fraction_per_gps_block,
        maximum_exposure_fraction_per_shift,
    )
    if any(not math.isfinite(float(value)) for value in numeric_values):
        raise ValueError("detector-set dependence settings must be finite")
    if (
        bootstrap_replicates < 1
        or minimum_gps_blocks < 3
        or minimum_shifts < 1
        or min(
            minimum_effective_gps_blocks,
            minimum_effective_shifts,
        )
        <= 0
        or not 0 < maximum_exposure_fraction_per_gps_block <= 1
        or not 0 < maximum_exposure_fraction_per_shift <= 1
    ):
        raise ValueError("detector-set dependence gate settings are invalid")

    report_path = Path(time_slide_report_path).resolve()
    background_path = Path(background_manifest_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate_path = Path(str(report.get("manifest_path", ""))).resolve()
    schedule_path = Path(
        str(report.get("slide_schedule_path", ""))
    ).resolve()
    if (
        report.get("status")
        != "variable_detector_set_block_permutation_background"
        or report.get("background_pairing_method")
        != DETECTOR_SET_BLOCK_PERMUTATION_METHOD
        or not candidate_path.is_file()
        or report.get("manifest_sha256") != file_sha256(candidate_path)
        or not schedule_path.is_file()
        or report.get("slide_schedule_sha256")
        != file_sha256(schedule_path)
        or not background_path.is_file()
        or report.get("background_manifest_sha256")
        != file_sha256(background_path)
    ):
        raise ValueError(
            "detector-set dependence audit requires a verified block report"
        )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if (
        schedule.get("status")
        != "frozen_detector_set_block_permutation_schedule"
        or schedule.get("method")
        != DETECTOR_SET_BLOCK_PERMUTATION_METHOD
        or schedule.get("selection_data")
        != DETECTOR_SET_BLOCK_SELECTION_DATA
        or schedule.get("candidate_scores_inspected") is not False
        or schedule.get("background_manifest_sha256")
        != file_sha256(background_path)
        or canonical_hash(
            detector_set_block_schedule_identity(schedule),
            32,
        )
        != schedule.get("schedule_id")
        or schedule.get("schedule_id")
        != report.get("slide_schedule_id")
    ):
        raise ValueError(
            "detector-set dependence frozen schedule failed replay"
        )

    split = str(report["split"])
    ordered = [str(value) for value in schedule["ordered_gps_blocks"]]
    block_positions = {
        block: index for index, block in enumerate(ordered)
    }
    permutations = schedule["permutations"]
    permutation_indices = [
        int(row["permutation_index"]) for row in permutations
    ]
    if (
        len(block_positions) != len(ordered)
        or len(ordered) < 3
        or len(set(permutation_indices)) != len(permutation_indices)
        or permutation_indices
        != [int(value) for value in report.get("slide_indices", [])]
    ):
        raise ValueError(
            "detector-set dependence schedule inventory differs"
        )
    subsets = [
        tuple(str(value) for value in subset)
        for subset in schedule["detector_subsets"]
    ]
    detectors = [str(value) for value in schedule["detectors"]]
    duration = float(schedule["window_duration_seconds"])
    slots: dict[str, dict[str, set[int]]] = {
        block: {} for block in ordered
    }
    with background_path.open("r", encoding="utf-8") as handle:
        background_rows = [
            json.loads(line) for line in handle if line.strip()
        ]
    all_slots: set[int] = set()
    for row in background_rows:
        if str(row.get("split")) != split:
            continue
        block = str(row.get("gps_block"))
        if block not in slots:
            raise ValueError(
                "background manifest contains an unscheduled GPS block"
            )
        parts = block.split(":")
        if len(parts) != 3 or parts[0] != "gps":
            raise ValueError(
                "detector-set dependence requires canonical GPS blocks"
            )
        offset = (float(row["gps_start"]) - float(parts[1])) / duration
        slot = int(round(offset))
        if not np.isclose(offset, slot, rtol=0.0, atol=1e-6):
            raise ValueError(
                "detector-set background window is not block aligned"
            )
        for ifo in _available_ifos(row):
            if slot in slots[block].setdefault(ifo, set()):
                raise ValueError(
                    "detector-set background repeats an IFO block slot"
                )
            slots[block][ifo].add(slot)
        all_slots.add(slot)

    cell_keys = []
    cell_exposure = []
    cell_dependency_blocks = []
    cell_shift_positions = []
    eligible_slots_by_cell: dict[
        tuple[int, str],
        set[int],
    ] = {}
    expected_sources_by_cell: dict[
        tuple[int, str],
        dict[str, str],
    ] = {}
    expected_subsets_by_cell_slot: dict[
        tuple[int, str, int],
        set[str],
    ] = {}
    for shift_position, permutation in enumerate(permutations):
        permutation_index = int(permutation["permutation_index"])
        shift_by_ifo = {
            str(key): int(value)
            for key, value in permutation["shift_by_ifo"].items()
        }
        observed_subset_counts: dict[str, int] = {
            "+".join(subset): 0 for subset in subsets
        }
        observed_eligible_blocks = 0
        observed_eligible_windows = 0
        for base_position, base_block in enumerate(ordered):
            source_blocks = {
                ifo: ordered[
                    (base_position + shift_by_ifo[ifo]) % len(ordered)
                ]
                for ifo in detectors
            }
            eligible_slots: set[int] = set()
            dependency_ifos: set[str] = set()
            for slot in all_slots:
                eligible_names = set()
                for subset in subsets:
                    if all(
                        slot in slots[source_blocks[ifo]].get(ifo, set())
                        for ifo in subset
                    ):
                        name = "+".join(subset)
                        eligible_names.add(name)
                        observed_subset_counts[name] += 1
                        dependency_ifos.update(subset)
                if eligible_names:
                    eligible_slots.add(slot)
                    expected_subsets_by_cell_slot[
                        (permutation_index, base_block, slot)
                    ] = eligible_names
            if not eligible_slots:
                continue
            observed_eligible_blocks += 1
            observed_eligible_windows += len(eligible_slots)
            key = (permutation_index, base_block)
            dependency_blocks = tuple(
                sorted({source_blocks[ifo] for ifo in dependency_ifos})
            )
            cell_keys.append(key)
            cell_exposure.append(len(eligible_slots) * duration)
            cell_dependency_blocks.append(dependency_blocks)
            cell_shift_positions.append(shift_position)
            eligible_slots_by_cell[key] = eligible_slots
            expected_sources_by_cell[key] = source_blocks
        if (
            observed_eligible_blocks != int(
                permutation["eligible_blocks"]
            )
            or observed_eligible_windows != int(
                permutation["eligible_windows"]
            )
            or observed_subset_counts
            != {
                str(key): int(value)
                for key, value in permutation[
                    "eligible_windows_by_detector_subset"
                ].items()
            }
        ):
            raise ValueError(
                "detector-set dependence exposure differs from schedule"
            )
    if not cell_keys:
        raise ValueError("detector-set dependence has no exposure cells")
    cell_positions = {
        key: index for index, key in enumerate(cell_keys)
    }
    exposure = np.asarray(cell_exposure, dtype=np.float64)
    exceedances = np.zeros(len(cell_keys), dtype=np.float64)
    max_dependencies = max(len(value) for value in cell_dependency_blocks)
    dependency_indices = np.full(
        (len(cell_keys), max_dependencies),
        -1,
        dtype=np.int64,
    )
    for row_index, dependencies in enumerate(cell_dependency_blocks):
        for column, block in enumerate(dependencies):
            dependency_indices[row_index, column] = block_positions[block]
    shift_positions = np.asarray(
        cell_shift_positions,
        dtype=np.int64,
    )

    candidate_rows = 0
    false_alarms = 0
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_rows += 1
            permutation_index = int(
                row.get(
                    "permutation_index",
                    row.get("slide_index", -1),
                )
            )
            base_block = str(row.get("base_gps_block"))
            base_slot = int(row.get("base_slot", -1))
            key = (permutation_index, base_block)
            if (
                key not in cell_positions
                or base_slot not in eligible_slots_by_cell[key]
            ):
                raise ValueError(
                    "detector-set candidate uses an unexposed block/slot"
                )
            subset_name = str(row.get("detector_subset"))
            if subset_name not in expected_subsets_by_cell_slot[
                (permutation_index, base_block, base_slot)
            ]:
                raise ValueError(
                    "detector-set candidate uses an ineligible subset"
                )
            source_blocks = row.get("source_gps_blocks")
            subset_ifos = subset_name.split("+")
            expected_sources = expected_sources_by_cell[key]
            if (
                not isinstance(source_blocks, dict)
                or set(source_blocks) != set(subset_ifos)
                or any(
                    str(source_blocks[ifo]) != expected_sources[ifo]
                    for ifo in subset_ifos
                )
            ):
                raise ValueError(
                    "detector-set candidate source blocks differ from schedule"
                )
            score = float(row["ranking_score"])
            if not math.isfinite(score):
                raise ValueError(
                    "detector-set candidate score must be finite"
                )
            if score >= threshold:
                exceedances[cell_positions[key]] += 1
                false_alarms += 1
    if candidate_rows != int(report.get("background_rows", -1)):
        raise ValueError(
            "detector-set candidate count differs from report"
        )
    total_seconds = float(exposure.sum())
    if total_seconds <= 0 or not np.isclose(
        total_seconds,
        float(report.get("equivalent_live_time_seconds", -1)),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "detector-set reconstructed exposure differs from report"
        )
    live_time_years = total_seconds / SECONDS_PER_YEAR
    far_per_year = false_alarms / live_time_years
    zero_upper = (
        -math.log(0.1) / live_time_years
        if false_alarms == 0
        else None
    )

    shift_exposure = np.bincount(
        shift_positions,
        weights=exposure,
        minlength=len(permutations),
    )
    block_exposure = np.zeros(len(ordered), dtype=np.float64)
    for cell_index, dependencies in enumerate(cell_dependency_blocks):
        allocation = exposure[cell_index] / len(dependencies)
        for block in dependencies:
            block_exposure[block_positions[block]] += allocation
    effective_blocks = _effective_units(block_exposure)
    effective_shifts = _effective_units(shift_exposure)
    block_fraction = float(
        block_exposure.max() / block_exposure.sum()
    )
    shift_fraction = float(
        shift_exposure.max() / shift_exposure.sum()
    )
    bootstrap = _multiway_cluster_bootstrap_rates(
        exposure,
        exceedances,
        dependency_indices,
        shift_positions,
        len(ordered),
        len(permutations),
        bootstrap_replicates,
        seed,
    )
    bootstrap_valid = (
        len(bootstrap) == bootstrap_replicates
        and bool(np.all(np.isfinite(bootstrap)))
    )
    bootstrap_interval = (
        [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ]
        if bootstrap_valid
        else None
    )
    shift_leave_one_out = []
    for position, permutation in enumerate(permutations):
        keep = shift_positions != position
        seconds = float(exposure[keep].sum())
        count = int(exceedances[keep].sum())
        shift_leave_one_out.append(
            {
                "permutation_index": int(
                    permutation["permutation_index"]
                ),
                "far_per_year": (
                    count / (seconds / SECONDS_PER_YEAR)
                    if seconds > 0
                    else None
                ),
            }
        )
    block_leave_one_out = []
    for block_position, block in enumerate(ordered):
        keep = np.all(
            dependency_indices != block_position,
            axis=1,
        )
        seconds = float(exposure[keep].sum())
        count = int(exceedances[keep].sum())
        block_leave_one_out.append(
            {
                "gps_block": block,
                "far_per_year": (
                    count / (seconds / SECONDS_PER_YEAR)
                    if seconds > 0
                    else None
                ),
            }
        )
    gates = {
        "frozen_score_blind_schedule_replayed": True,
        "candidate_rows_assigned_to_multiway_physical_clusters": True,
        "equivalent_exposure_reconstructed": True,
        "minimum_unique_gps_blocks": len(ordered) >= minimum_gps_blocks,
        "minimum_unique_shifts": (
            len(permutations) >= minimum_shifts
        ),
        "minimum_effective_gps_blocks": (
            effective_blocks >= minimum_effective_gps_blocks
        ),
        "minimum_effective_shifts": (
            effective_shifts >= minimum_effective_shifts
        ),
        "gps_block_exposure_not_dominated": (
            block_fraction
            <= maximum_exposure_fraction_per_gps_block
        ),
        "shift_exposure_not_dominated": (
            shift_fraction <= maximum_exposure_fraction_per_shift
        ),
        "cluster_bootstrap_complete": bootstrap_valid,
    }
    shift_values = [
        row["far_per_year"]
        for row in shift_leave_one_out
        if row["far_per_year"] is not None
    ]
    block_values = [
        row["far_per_year"]
        for row in block_leave_one_out
        if row["far_per_year"] is not None
    ]
    return {
        "status": (
            "detector_set_candidate_background_dependence_audit_v1"
        ),
        "passed": all(gates.values()),
        "scientific_claim_allowed": False,
        "interpretation": (
            "standard equivalent block-permutation exposure with a "
            "multiway source-block x offset pigeonhole dependence "
            "sensitivity analysis; candidates are never IID units"
        ),
        "split": split,
        "threshold": threshold,
        "time_slide_report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "background_manifest": {
            "path": str(background_path),
            "sha256": file_sha256(background_path),
        },
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": file_sha256(schedule_path),
            "schedule_id": schedule["schedule_id"],
        },
        "unique_gps_blocks": len(ordered),
        "unique_shifts": len(permutations),
        "effective_gps_blocks": effective_blocks,
        "effective_shifts": effective_shifts,
        "maximum_exposure_fraction_per_gps_block": block_fraction,
        "maximum_exposure_fraction_per_shift": shift_fraction,
        "live_time_seconds": total_seconds,
        "live_time_years": live_time_years,
        "candidate_rows": candidate_rows,
        "false_alarms": false_alarms,
        "far_per_year": far_per_year,
        "ifar_years": 1.0 / far_per_year if far_per_year > 0 else None,
        "far_90_upper_limit_if_zero": zero_upper,
        "multiway_cluster_bootstrap": {
            "method": (
                "physical_source_blocks_x_offset_pigeonhole_v1"
            ),
            "maximum_source_blocks_per_cell": max_dependencies,
            "replicates": bootstrap_replicates,
            "seed": seed,
            "far_per_year_95": bootstrap_interval,
            "informative_for_rate_variation": false_alarms > 0,
            "zero_count_policy": (
                "report Poisson 90% upper limit; zero-only resampling "
                "cannot estimate event-rate variation"
                if false_alarms == 0
                else None
            ),
        },
        "leave_one_shift_out_far_range": (
            [min(shift_values), max(shift_values)]
            if shift_values
            else None
        ),
        "leave_one_gps_block_out_far_range": (
            [min(block_values), max(block_values)]
            if block_values
            else None
        ),
        "gates": gates,
    }
