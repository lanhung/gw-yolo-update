from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.mask_audit import (
    binary_mask_iou,
    evaluate_gravityspy_mask_audit,
    materialize_gravityspy_mask_consensus,
    plan_gravityspy_mask_audit,
)


def test_binary_mask_iou_is_hand_calculated() -> None:
    left = np.asarray([1, 1, 0, 0])
    right = np.asarray([1, 0, 1, 0])
    assert binary_mask_iou(left, right) == pytest.approx(1 / 3)


def test_mask_audit_plan_requires_three_blinded_annotators(tmp_path) -> None:
    sample = tmp_path / "sample.npz"
    np.savez(
        sample,
        features=np.asarray([[0.1, 0.8], [0.2, 0.0]], dtype=np.float32),
        glitch_mask=np.asarray([[0, 1], [0, 0]], dtype=np.uint8),
    )
    manifest = tmp_path / "val.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "glitch_id": "g1",
                "ml_label": "Blip",
                "ifo": "H1",
                "observing_run": "O3a",
                "network_gps_block": "O3a:1",
                "path": str(sample),
                "sha256": file_sha256(sample),
            }
        )
        + "\n"
    )
    report = plan_gravityspy_mask_audit(manifest, tmp_path / "audit", per_label=1)
    task = json.loads(Path(report["task_manifest_path"]).read_text())
    annotation_task = json.loads(
        Path(report["annotation_task_manifest_path"]).read_text()
    )
    assert task["required_independent_annotators"] == 3
    assert "three independent" in report["scientific_blocker"]
    assert "exact_command" in report and "environment" in report
    assert "numeric_sample_path" not in annotation_task
    assert "weak_mask_key" not in annotation_task
    assert task["annotation_task_hash"] == annotation_task["annotation_task_hash"]
    assert report["mask_targets_exposed_to_annotators"] is False
    with np.load(annotation_task["blinded_input_path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"features"}
    assert binary_mask_iou(np.zeros(3), np.zeros(3)) == 1.0


def test_mask_audit_requires_blinded_independent_annotations(tmp_path) -> None:
    weak = tmp_path / "weak.npz"
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    third = tmp_path / "third.npz"
    np.savez(weak, glitch_mask=np.asarray([1, 1, 0, 0]))
    blind = tmp_path / "blind.npz"
    np.savez(blind, features=np.asarray([0.4, 0.2, 0.8, 0.1]))
    np.savez(first, mask=np.asarray([1, 0, 1, 0]))
    np.savez(second, mask=np.asarray([1, 0, 1, 0]))
    np.savez(third, mask=np.asarray([1, 0, 1, 0]))
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "audit_id": "a1",
                "glitch_id": "g1",
                "ml_label": "Blip",
                "numeric_sample_path": str(weak),
                "numeric_sample_sha256": file_sha256(weak),
                "weak_mask_key": "glitch_mask",
                "blinded_input_path": str(blind),
                "blinded_input_sha256": file_sha256(blind),
                "blinded_input_keys": ["features"],
                "mask_shape": [4],
                "required_independent_annotators": 3,
                "required_annotation_key": "mask",
            }
        )
        + "\n"
    )
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        "".join(
            json.dumps(
                {
                    "audit_id": "a1",
                    "annotator_id": annotator,
                    "mask_path": str(path),
                    "mask_sha256": file_sha256(path),
                    "blinded_to_weak_mask": True,
                    "protocol_version": "v1",
                    "annotation_task_hash": "task-hash-v1",
                }
            )
            + "\n"
            for annotator, path in (("one", first), ("two", second), ("three", third))
        )
    )
    task_row = json.loads(tasks.read_text())
    task_row["annotation_task_hash"] = "task-hash-v1"
    tasks.write_text(json.dumps(task_row) + "\n")
    report = evaluate_gravityspy_mask_audit(
        tasks, annotations, tmp_path / "report.json"
    )
    assert report["overall"]["mean_interannotator_iou"] == 1.0
    assert report["overall"]["weak_consensus_iou_mean"] == pytest.approx(1 / 3)
    consensus = materialize_gravityspy_mask_consensus(
        tasks, annotations, tmp_path / "report.json", tmp_path / "consensus"
    )
    row = json.loads(Path(consensus["manifest_path"]).read_text())
    with np.load(row["path"], allow_pickle=False) as arrays:
        assert arrays["mask"].tolist() == [1, 0, 1, 0]
    assert consensus["training_allowed"] is False
    assert row["human_pixel_mask"] is True
    assert row["training_allowed"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        materialize_gravityspy_mask_consensus(
            tasks, annotations, tmp_path / "report.json", tmp_path / "consensus"
        )

    tampered_report = json.loads((tmp_path / "report.json").read_text())
    tampered_report["evaluated_tasks"][0]["weak_consensus_iou"] = 1.0
    tampered_path = tmp_path / "tampered-report.json"
    tampered_path.write_text(json.dumps(tampered_report))
    with pytest.raises(ValueError, match="metrics differ"):
        materialize_gravityspy_mask_consensus(
            tasks, annotations, tampered_path, tmp_path / "tampered-consensus"
        )

    annotation_rows = [json.loads(line) for line in annotations.read_text().splitlines()]
    annotation_rows[-1]["protocol_version"] = "v2"
    mixed = tmp_path / "mixed-protocol.jsonl"
    mixed.write_text("".join(json.dumps(row) + "\n" for row in annotation_rows))
    with pytest.raises(ValueError, match="mix protocol versions"):
        evaluate_gravityspy_mask_audit(
            tasks, mixed, tmp_path / "mixed-protocol-report.json"
        )

    annotation_rows[-1]["protocol_version"] = "v1"
    annotation_rows[-1]["annotator_id"] = "two"
    repeated = tmp_path / "repeated-annotator.jsonl"
    repeated.write_text("".join(json.dumps(row) + "\n" for row in annotation_rows))
    with pytest.raises(ValueError, match="lacks independent annotators"):
        evaluate_gravityspy_mask_audit(
            tasks, repeated, tmp_path / "repeated-annotator-report.json"
        )
