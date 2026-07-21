from __future__ import annotations

import math

from gwyolo.background import SECONDS_PER_YEAR
from gwyolo.exposure import plan_candidate_background_exposure


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
