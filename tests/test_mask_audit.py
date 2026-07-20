from __future__ import annotations

import json

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.mask_audit import binary_mask_iou, evaluate_gravityspy_mask_audit


def test_binary_mask_iou_is_hand_calculated() -> None:
    left = np.asarray([1, 1, 0, 0])
    right = np.asarray([1, 0, 1, 0])
    assert binary_mask_iou(left, right) == pytest.approx(1 / 3)
    assert binary_mask_iou(np.zeros(3), np.zeros(3)) == 1.0


def test_mask_audit_requires_blinded_independent_annotations(tmp_path) -> None:
    weak = tmp_path / "weak.npz"
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    third = tmp_path / "third.npz"
    np.savez(weak, glitch_mask=np.asarray([1, 1, 0, 0]))
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
                "weak_mask_key": "glitch_mask",
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
                }
            )
            + "\n"
            for annotator, path in (("one", first), ("two", second), ("three", third))
        )
    )
    report = evaluate_gravityspy_mask_audit(
        tasks, annotations, tmp_path / "report.json"
    )
    assert report["overall"]["mean_interannotator_iou"] == 1.0
    assert report["overall"]["weak_consensus_iou_mean"] == pytest.approx(1 / 3)
