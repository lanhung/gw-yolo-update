from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.cosmology import FlatLambdaCDMGrid
from gwyolo.injections import (
    plan_injection_recipes,
    run_injection_plan,
    run_nested_injection_scale_plan,
)
from gwyolo.io import file_sha256


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


def test_run_injection_plan_can_create_train_val_without_test(tmp_path) -> None:
    import json

    rows = [
        {
            "window_id": f"{split}-window",
            "split": split,
            "gps_block": f"{split}-block",
            "gps_start": gps,
            "gps_end": gps + 8,
            "ifos": ["H1", "L1"],
        }
        for split, gps in (("train", 1000), ("val", 2000))
    ]
    manifest = tmp_path / "background.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    background_report = tmp_path / "background_report.json"
    background_report.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"live_time_years": 1.0},
                    "val": {"live_time_years": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    report = run_injection_plan(
        manifest,
        background_report,
        tmp_path / "planned",
        validation_count=2,
        test_count=0,
        training_count=3,
    )
    assert report["counts_by_split"] == {"train": 3, "val": 2}
    assert report["requested_counts_by_split"]["test"] == 0
    assert report["background_manifest_sha256"] == file_sha256(manifest)


def test_nested_injection_scale_plan_reuses_core_and_frozen_validation(tmp_path) -> None:
    rows = [
        {
            "window_id": f"{split}-window",
            "split": split,
            "gps_block": f"{split}-block",
            "gps_start": gps,
            "gps_end": gps + 8,
            "ifos": ["H1", "L1"],
        }
        for split, gps in (("train", 1000), ("val", 2000))
    ]
    background = tmp_path / "background.jsonl"
    background.write_text("".join(json.dumps(row) + "\n" for row in rows))
    exposure = tmp_path / "background_report.json"
    exposure.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"live_time_years": 1.0},
                    "val": {"live_time_years": 1.0},
                }
            }
        )
    )
    base_dir = tmp_path / "base"
    base = run_injection_plan(
        background,
        exposure,
        base_dir,
        validation_count=3,
        test_count=0,
        training_count=10,
        seed=7,
    )
    report = run_nested_injection_scale_plan(
        base["manifest_path"],
        background,
        exposure,
        tmp_path / "scales",
        scales=(10, 20, 30),
        supplement_seed=8,
    )
    assert report["strictly_nested"]
    assert report["gps_diversity_saturated"]
    assert report["test_recipes_read"] == 0
    assert [item["increment_rows"] for item in report["scales"]] == [10, 10, 10]
    manifests = [
        {
            json.loads(line)["injection_id"]
            for line in Path(item["manifest_path"]).read_text().splitlines()
        }
        for item in report["scales"]
    ]
    assert manifests[0] < manifests[1] < manifests[2]
    assert report["validation"]["rows"] == 3
