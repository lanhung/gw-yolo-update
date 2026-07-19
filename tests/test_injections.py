from __future__ import annotations

import math

import numpy as np
import pytest

from gwyolo.cosmology import FlatLambdaCDMGrid
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
    cosmology = FlatLambdaCDMGrid()
    maximum_redshift = float(cosmology.redshift_at_luminosity_distance(100.0))
    maximum_comoving = float(cosmology.distances_at_redshift(maximum_redshift)[0])
    base_weight = 2.0 * 4.0 * math.pi / 3.0 * maximum_comoving**3 / 4
    expected_vt = sum(base_weight / (1.0 + row["redshift"]) for row in recipes)
    assert math.isclose(sum(row["vt_weight"] for row in recipes), expected_vt)
    assert math.isclose(report["total_vt_weight_by_split"]["test"], expected_vt)
    assert all(row["mass_1_detector_msun"] > row["mass_1_msun"] for row in recipes)
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


def test_flat_cosmology_distance_round_trip_and_monotonicity() -> None:
    cosmology = FlatLambdaCDMGrid(points=10_001)
    redshifts = [0.01, 0.1, 0.5, 1.0]
    comoving, luminosity = cosmology.distances_at_redshift(redshifts)
    assert all(left < right for left, right in zip(luminosity, luminosity[1:]))
    recovered = cosmology.redshift_at_luminosity_distance(luminosity)
    assert recovered == pytest.approx(redshifts, abs=1e-8)
    assert luminosity == pytest.approx((1 + np.asarray(redshifts)) * comoving)


def test_nsbh_detector_frame_neutron_star_mass_stays_in_approximant_domain() -> None:
    rows = [
        {
            "window_id": "v",
            "split": "val",
            "gps_block": "val-block",
            "gps_start": 1000,
            "gps_end": 1008,
            "ifos": ["H1", "L1"],
        }
    ]
    population = {
        "NSBH": {
            "fraction": 1.0,
            "maximum_distance_mpc": 1500.0,
            "approximant": "IMRPhenomNSBH",
        }
    }
    recipes, report = plan_injection_recipes(
        rows, {"val": 1.0}, {"val": 1000}, population, seed=11
    )
    assert max(row["mass_2_detector_msun"] for row in recipes) <= 3.0
    assert report["approximant_domain_audit"]["passed"]
