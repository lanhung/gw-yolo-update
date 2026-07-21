from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from gwyolo.background import SECONDS_PER_YEAR
from gwyolo.exposure import (
    candidate_slide_schedule_identity,
    freeze_candidate_time_slide_schedule,
    freeze_candidate_time_slide_range_schedule,
    freeze_candidate_block_permutation_schedule,
    plan_candidate_background_exposure,
)
from gwyolo.io import canonical_hash


def test_candidate_exposure_plan_counts_every_valid_noncyclic_pair_once() -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": "g",
            "ifos": ["H1", "L1"],
        }
        for index in range(4)
    ]
    report = plan_candidate_background_exposure(
        windows, "val", "H1", "L1", 3, 8, target_far_per_year=1.0
    )
    assert [row["paired_windows"] for row in report["nonzero_slide_exposure"]] == [
        3,
        2,
        1,
    ]
    assert report["equivalent_live_time_seconds"] == 48
    assert report["all_observed_positive_lag_pairs"] == 6
    assert report["all_observed_positive_lag_exposure_seconds"] == 48
    assert report["far_resolution_one_count_per_year"] == SECONDS_PER_YEAR / 48
    assert report["required_equivalent_years_for_zero_count_upper"] == math.log(10)
    assert report["target_zero_count_upper_reached"] is False


def test_block_permutation_schedule_reaches_target_without_scores(tmp_path) -> None:
    manifest = tmp_path / "background.jsonl"
    rows = []
    for block in range(3):
        block_start = 1000 + block * 256
        for slot in range(2):
            rows.append(
                {
                    "window_id": f"w-{block}-{slot}",
                    "split": "val",
                    "gps_start": block_start + slot * 8,
                    "gps_end": block_start + (slot + 1) * 8,
                    "gps_block": f"gps:{block_start}:256",
                    "ifos": ["H1", "L1"],
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = freeze_candidate_block_permutation_schedule(
        manifest,
        tmp_path / "schedule.json",
        "val",
        "H1",
        "L1",
        target_far_per_year=1_000_000,
        maximum_shifts=2,
    )
    assert result["schedule_exposure_target_reached"]
    assert result["selected_shift_count"] == 2
    assert result["selected_equivalent_live_time_seconds"] == 96
    assert [row["paired_windows"] for row in result["selected_shifts"]] == [6, 6]
    assert result["candidate_scores_inspected"] is False
    assert result["selection_data"] == (
        "background_gps_blocks_and_detector_availability_only"
    )


def test_candidate_exposure_plan_excludes_missing_shifted_detector() -> None:
    windows = [
        {
            "window_id": "w0",
            "split": "test",
            "gps_start": 0,
            "gps_end": 8,
            "gps_block": "g0",
            "ifos": ["H1", "L1"],
        },
        {
            "window_id": "w1",
            "split": "test",
            "gps_start": 8,
            "gps_end": 16,
            "gps_block": "g1",
            "ifos": ["H1"],
        },
    ]
    report = plan_candidate_background_exposure(
        windows, "test", "H1", "L1", 1, 8, target_far_per_year=10.0
    )
    assert report["equivalent_live_time_seconds"] == 0
    assert report["zero_count_far_upper_per_year"] is None


def test_candidate_exposure_plan_uses_absolute_slide_range() -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": "g",
            "ifos": ["H1", "L1"],
        }
        for index in range(5)
    ]
    report = plan_candidate_background_exposure(
        windows,
        "val",
        "H1",
        "L1",
        2,
        8,
        target_far_per_year=1.0,
        slide_start_index=3,
    )
    assert report["slide_start_index"] == 3
    assert report["slide_stop_index_exclusive"] == 5
    assert [row["slide_index"] for row in report["nonzero_slide_exposure"]] == [3, 4]
    assert [row["paired_windows"] for row in report["nonzero_slide_exposure"]] == [2, 1]


def test_candidate_time_slide_schedule_freezes_only_nonzero_offsets(
    tmp_path: Path,
) -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": "g",
            "ifos": ["H1", "L1"],
        }
        for index in range(5)
    ]
    manifest = tmp_path / "background.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    output = tmp_path / "schedule.json"
    report = freeze_candidate_time_slide_schedule(
        manifest, output, "val", "H1", "L1", 8, [1, 3], 1.0
    )
    assert report["status"] == "frozen_candidate_time_slide_schedule"
    assert report["slide_indices"] == [1, 3]
    assert report["exposure_plan"]["slide_indices_contiguous"] is False
    assert report["candidate_scores_inspected"] is False
    assert len(report["schedule_id"]) == 32

    with pytest.raises(FileExistsError, match="immutable"):
        freeze_candidate_time_slide_schedule(
            manifest, output, "val", "H1", "L1", 8, [1], 1.0
        )
    with pytest.raises(ValueError, match="zero-exposure"):
        freeze_candidate_time_slide_schedule(
            manifest, tmp_path / "zero.json", "val", "H1", "L1", 8, [9], 1.0
        )


def test_range_schedule_selects_shortest_nonzero_prefix_for_target(
    tmp_path: Path,
) -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": "g",
            "ifos": ["H1", "L1"],
        }
        for index in range(5)
    ]
    manifest = tmp_path / "background.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in windows), encoding="utf-8"
    )
    required_seconds = 40.0
    target_far = -math.log(0.1) * SECONDS_PER_YEAR / required_seconds
    output = tmp_path / "range-schedule.json"
    report = freeze_candidate_time_slide_range_schedule(
        manifest,
        output,
        "val",
        "H1",
        "L1",
        8.0,
        1,
        8,
        target_far,
        0.90,
    )
    assert report["schema_version"] == 2
    assert report["slide_indices"] == [1, 2]
    assert report["exposure_plan"]["equivalent_live_time_seconds"] == 56
    assert report["schedule_exposure_target_reached"] is True
    metadata = report["selection_metadata"]
    assert metadata["evaluated_offsets"] == 7
    assert metadata["nonzero_offsets_available"] == 4
    assert metadata["available_equivalent_live_time_seconds"] == 80
    assert metadata["selected_equivalent_live_time_seconds"] == 56
    assert report["schedule_id"] == canonical_hash(
        candidate_slide_schedule_identity(report), 32
    )
