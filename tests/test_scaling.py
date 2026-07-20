import csv
import json

import pytest

from gwyolo.scaling import (
    analyze_manifest,
    fit_power_law_curve,
    make_scaling_plan,
    summarize_physical_scale_reports,
)


def _write_manifest(path, rows):
    fieldnames = ["group_id", "split", "class_0", "class_1", "sample_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_scale_audit_counts_physical_groups_not_rendered_images(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"group_id": "a", "split": "train", "class_0": 1, "class_1": 0, "sample_id": "a1"},
            {"group_id": "a", "split": "train", "class_0": 1, "class_1": 0, "sample_id": "a2"},
            {"group_id": "b", "split": "train", "class_0": 1, "class_1": 1, "sample_id": "b1"},
            {"group_id": "c", "split": "val", "class_0": 0, "class_1": 1, "sample_id": "c1"},
            {"group_id": "d", "split": "test", "class_0": 0, "class_1": 0, "sample_id": "d1"},
        ],
    )
    audit = analyze_manifest(manifest)
    assert audit["images"] == 5
    assert audit["physical_groups"] == 4
    assert audit["images_in_multi_image_groups"] == 2
    assert audit["splits"]["train"]["physical_groups"] == 2
    assert audit["splits"]["train"]["group_composition"] == {
        "chirp+noise": 1,
        "chirp_only": 1,
    }


def test_scale_plan_reports_target_gaps(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"group_id": "a", "split": "train", "class_0": 1, "class_1": 0, "sample_id": "a"},
            {"group_id": "b", "split": "train", "class_0": 1, "class_1": 1, "sample_id": "b"},
            {"group_id": "c", "split": "val", "class_0": 0, "class_1": 1, "sample_id": "c"},
            {"group_id": "d", "split": "test", "class_0": 0, "class_1": 0, "sample_id": "d"},
        ],
    )
    plan = make_scaling_plan(analyze_manifest(manifest), baseline_target=100, research_target=200)
    assert plan["baseline_target"]["target_composition"] == {
        "chirp_only": 25,
        "noise_only": 25,
        "chirp+noise": 40,
        "empty": 10,
    }
    assert plan["baseline_target"]["gap_by_composition"]["chirp+noise"] == 39
    assert plan["promotion_blockers"]["evaluation_set_too_small"] is True


def test_scale_audit_rejects_cross_split_group(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {"group_id": "a", "split": "train", "class_0": 1, "class_1": 0, "sample_id": "a1"},
            {"group_id": "a", "split": "test", "class_0": 1, "class_1": 0, "sample_id": "a2"},
        ],
    )
    with pytest.raises(ValueError, match="crosses splits"):
        analyze_manifest(manifest)


def test_power_law_fit_recovers_hand_generated_curve():
    points = [
        {"physical_groups": groups, "metric": 0.9 - 0.6 * groups**-0.5}
        for groups in (100, 200, 400, 800, 1600)
    ]
    result = fit_power_law_curve(points)
    assert result["parameters"]["alpha"] == pytest.approx(0.5, abs=0.002)
    assert result["parameters"]["asymptote"] == pytest.approx(0.9, abs=0.001)
    assert result["parameters"]["amplitude"] == pytest.approx(0.6, abs=0.01)
    assert result["r_squared"] == pytest.approx(1.0)


def test_physical_scale_summary_enforces_controls_and_calculates_seed_spread(tmp_path):
    plan = tmp_path / "scale-plan.json"
    plan.write_text(
        json.dumps(
            {
                "validation_manifest_sha256": "val-hash",
                "scales": [{"scale": 2000, "manifest_sha256": "train-hash"}],
            }
        )
    )
    reports = []
    for seed, metric in ((1, 0.2), (2, 0.4)):
        path = tmp_path / f"run-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "train_manifest_sha256": "train-hash",
                    "validation_manifest_sha256": "val-hash",
                    "test_evaluation": None,
                    "checkpoint_selection": "final_update",
                    "training_budget_reached": True,
                    "seed": seed,
                    "training_selection": {"selected_rows": 2000},
                    "calibrated_validation": {"chirp_iou": metric},
                    "selected_chirp_threshold": 0.5,
                    "optimizer_updates": 3750,
                    "optimizer_examples": 60000,
                    "pretrained_checkpoint_sha256": "pretrained-hash",
                    "config_hash": "config-hash",
                    "checkpoint_sha256": f"checkpoint-{seed}",
                }
            )
        )
        reports.append(path)
    result = summarize_physical_scale_reports(
        reports, plan, tmp_path / "summary.json"
    )
    scale = result["scales"][0]
    assert scale["validation_chirp_iou_mean"] == pytest.approx(0.3)
    assert scale["validation_chirp_iou_sample_std"] == pytest.approx(2**-0.5 * 0.2)
    assert not scale["minimum_three_seed_gate"]
    assert not result["scientific_claim_allowed"]
