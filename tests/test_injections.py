from __future__ import annotations

import math

from gwyolo.injections import plan_injection_recipes


def test_injection_vt_weights_integrate_sampled_volume_time() -> None:
    rows = [
        {
            "window_id": f"w-{index}",
            "split": "test",
            "gps_block": f"b-{index // 2}",
            "gps_start": 1000 + 8 * index,
            "gps_end": 1008 + 8 * index,
            "ifos": ["H1", "L1"],
        }
        for index in range(4)
    ]
    population = {"BBH": {"fraction": 1.0, "maximum_distance_mpc": 100.0}}
    recipes, report = plan_injection_recipes(
        rows,
        {"test": 2.0},
        {"test": 4},
        population,
        seed=3,
    )
    expected_vt = 2.0 * 4.0 * math.pi / 3.0 * 100.0**3
    assert math.isclose(sum(row["vt_weight"] for row in recipes), expected_vt)
    assert math.isclose(report["total_vt_weight_by_split"]["test"], expected_vt)
    assert report["unique_injection_ids"] == 4
    assert all(not values for values in report["cross_split_injection_overlaps"].values())


def test_injection_splits_inherit_disjoint_background_blocks() -> None:
    rows = [
        {
            "window_id": "v",
            "split": "val",
            "gps_block": "val-block",
            "gps_start": 1000,
            "gps_end": 1008,
            "ifos": ["H1"],
        },
        {
            "window_id": "t",
            "split": "test",
            "gps_block": "test-block",
            "gps_start": 2000,
            "gps_end": 2008,
            "ifos": ["H1"],
        },
    ]
    recipes, _ = plan_injection_recipes(
        rows,
        {"val": 1.0, "test": 1.0},
        {"val": 2, "test": 2},
        {"BNS": {"fraction": 1.0, "maximum_distance_mpc": 10.0}},
    )
    assert {row["gps_block"] for row in recipes if row["split"] == "val"} == {"val-block"}
    assert {row["gps_block"] for row in recipes if row["split"] == "test"} == {"test-block"}
