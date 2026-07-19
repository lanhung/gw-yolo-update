import csv

import pytest

from gwyolo.scaling import analyze_manifest, make_scaling_plan


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
