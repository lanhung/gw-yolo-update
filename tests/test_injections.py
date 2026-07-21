from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.cosmology import FlatLambdaCDMGrid
from gwyolo.injections import (
    audit_paired_data_domain_manifests,
    plan_injection_recipes,
    run_paired_background_remap,
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


def test_paired_background_remap_preserves_sources_and_replaces_gps_groups(
    tmp_path,
) -> None:
    source = tmp_path / "source.jsonl"
    source_rows = [
        {
            "split": "train",
            "injection_id": f"i{index}",
            "waveform_id": f"w{index}",
            "source_family": "BBH",
            "mass_1_msun": 30.0 + index,
            "background_window_id": f"old-window-{index}",
            "gps_block": f"old-block-{index}",
            "gps_time": 1001.0 + index,
            "ifos": ["H1", "L1"],
            "vt_weight": 2.0 + index,
        }
        for index in range(4)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    background = tmp_path / "background.jsonl"
    background_rows = [
        {
            "split": "train",
            "window_id": f"new-window-{index}",
            "gps_block": f"new-block-{index % 2}",
            "gps_start": 2000.0 + 8 * index,
            "gps_end": 2008.0 + 8 * index,
            "ifos": ["H1", "L1", "V1"] if index == 0 else ["H1", "L1"],
        }
        for index in range(4)
    ]
    background.write_text(
        "".join(json.dumps(row) + "\n" for row in background_rows),
        encoding="utf-8",
    )
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        json.dumps(
            {
                "split": "val",
                "injection_id": "vi",
                "waveform_id": "vw",
                "gps_block": "validation-block",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = run_paired_background_remap(
        source, background, validation, tmp_path / "output", seed=7
    )
    remapped = [
        json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()
    ]
    assert report["source_parameters_preserved"] is True
    assert report["source_physics_hash"] == report["remapped_physics_hash"]
    assert report["remapped_unique_gps_blocks"] == 2
    assert {row["injection_id"] for row in remapped} == {"i0", "i1", "i2", "i3"}
    assert {row["mass_1_msun"] for row in remapped} == {30.0, 31.0, 32.0, 33.0}
    assert {row["gps_block"] for row in remapped} == {"new-block-0", "new-block-1"}
    assert all(row["background_remap"]["source_gps_block"].startswith("old-") for row in remapped)


def test_paired_background_remap_rejects_validation_gps_overlap(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "train",
                "injection_id": "i",
                "waveform_id": "w",
                "background_window_id": "old-window",
                "gps_block": "old-block",
                "gps_time": 1001.0,
                "ifos": ["H1", "L1"],
            }
        )
        + "\n"
    )
    background = tmp_path / "background.jsonl"
    background.write_text(
        json.dumps(
            {
                "split": "train",
                "window_id": "new-window",
                "gps_block": "shared-block",
                "gps_start": 2000.0,
                "gps_end": 2008.0,
                "ifos": ["H1", "L1"],
            }
        )
        + "\n"
    )
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        json.dumps(
            {
                "split": "val",
                "injection_id": "vi",
                "waveform_id": "vw",
                "gps_block": "shared-block",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="no new split-safe GPS windows"):
        run_paired_background_remap(
            source, background, validation, tmp_path / "output", seed=7
        )


def test_paired_data_domain_audit_requires_same_population_and_disjoint_gps(
    tmp_path,
) -> None:
    source = {
        "split": "train",
        "injection_id": "i",
        "waveform_id": "w",
        "source_family": "BBH",
        "waveform_backend": "validated",
        "waveform_approximant": "A",
        "f_lower_hz": 20.0,
        "mass_1_msun": 30.0,
        "mass_2_msun": 20.0,
        "mass_1_detector_msun": 33.0,
        "mass_2_detector_msun": 22.0,
        "spin_1z": 0.1,
        "spin_2z": 0.2,
        "lambda_1": 0.0,
        "lambda_2": 0.0,
        "inclination": 1.0,
        "right_ascension": 2.0,
        "declination": 0.3,
        "polarization": 0.4,
        "coalescence_phase": 0.5,
        "luminosity_distance_mpc": 1000.0,
        "comoving_distance_mpc": 900.0,
        "redshift": 0.1,
        "maximum_distance_mpc": 5000.0,
        "vt_weight": 3.0,
        "vt_weight_unit": "Mpc^3 yr",
        "vt_measure": "measure",
        "gps_block": "old",
        "ifos": ["H1", "L1"],
    }
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(json.dumps(source) + "\n")
    diverse = tmp_path / "diverse.jsonl"
    diverse.write_text(json.dumps({**source, "gps_block": "new"}) + "\n")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        json.dumps(
            {
                **source,
                "split": "val",
                "injection_id": "vi",
                "waveform_id": "vw",
                "gps_block": "validation",
            }
        )
        + "\n"
    )
    report = audit_paired_data_domain_manifests(
        baseline, diverse, validation, tmp_path / "audit.json"
    )
    assert report["source_parameters_identical"] is True
    assert report["cross_arm_gps_block_overlap"] == 0
    assert report["validation_group_overlaps"]["independent_gps"] == {
        "injection_id": 0,
        "waveform_id": 0,
        "gps_block": 0,
    }
    diverse.write_text(json.dumps({**source, "gps_block": "new", "mass_1_msun": 31.0}) + "\n")
    with pytest.raises(ValueError, match="source population changed"):
        audit_paired_data_domain_manifests(
            baseline, diverse, validation, tmp_path / "bad-audit.json"
        )


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
