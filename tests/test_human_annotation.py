from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from gwyolo.human_annotation import (
    HumanMaskAnnotationSession,
    merge_human_mask_annotation_manifests,
    serve_human_mask_annotation,
)
from gwyolo.io import file_sha256


def _blinded_tasks(tmp_path: Path, count: int = 2) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        blind = tmp_path / f"blind-{index}.npz"
        np.savez(
            blind,
            features=np.asarray(
                [
                    [[[0.0, 1.0], [2.0, 3.0]]],
                    [[[3.0, 2.0], [1.0, 0.0]]],
                ],
                dtype=np.float32,
            ),
            ifos=np.asarray(["H1", "L1"]),
            q_values=np.asarray([4.0], dtype=np.float32),
            sample_rate=np.asarray(2048, dtype=np.int32),
        )
        rows.append(
            {
                "audit_id": f"audit-{index}",
                "blinded_input_path": str(blind),
                "blinded_input_sha256": file_sha256(blind),
                "blinded_input_keys": ["features", "ifos", "q_values", "sample_rate"],
                "mask_shape": [2, 1, 2, 2],
                "required_independent_annotators": 3,
                "required_annotation_key": "mask",
                "blinding_requirement": "target-free numeric features only",
                "annotation_status": "pending",
                "annotation_task_hash": f"task-hash-{index}",
            }
        )
    manifest = tmp_path / "annotation-tasks.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def _complete_session(
    tasks: Path, output: Path, annotator: str
) -> HumanMaskAnnotationSession:
    session = HumanMaskAnnotationSession(tasks, annotator, output)
    for index in range(2):
        session.save_mask(index, [1, 0, 0, 1, 0, 1, 1, 0])
    session.finalize()
    return session


def test_blinded_annotation_session_roundtrips_every_plane(tmp_path: Path) -> None:
    tasks = _blinded_tasks(tmp_path)
    session = HumanMaskAnnotationSession(tasks, "reviewer-a", tmp_path / "reviewer-a")
    payload = session.task_payload(0)
    assert payload["shape"] == [2, 1, 2, 2]
    assert payload["plane_labels"] == ["H1 / Q=4", "L1 / Q=4"]
    assert len(payload["features_u8"]) == 8
    assert payload["target_fields_exposed"] is False
    assert not {
        "glitch_id",
        "ml_label",
        "numeric_sample_path",
        "weak_mask_key",
    } & set(payload)
    first_mask = [1, 0, 0, 1, 0, 1, 1, 0]
    progress = session.save_mask(0, first_mask)
    assert progress["completed"] == 1
    assert session.task_payload(0)["mask"] == first_mask
    with pytest.raises(ValueError, match="incomplete"):
        session.finalize()
    session.save_mask(1, [0] * 8)
    final = session.finalize()
    assert final["tasks"] == 2
    assert final["complete_frozen_task_coverage"] is True
    assert final["target_fields_exposed"] is False
    with pytest.raises(FileExistsError, match="immutable"):
        session.save_mask(0, first_mask)


def test_annotation_session_rejects_internal_or_target_inputs(tmp_path: Path) -> None:
    tasks = _blinded_tasks(tmp_path, count=1)
    row = json.loads(tasks.read_text())
    row["weak_mask_key"] = "glitch_mask"
    tasks.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="internal target metadata"):
        HumanMaskAnnotationSession(tasks, "reviewer-a", tmp_path / "bad")

    clean = _blinded_tasks(tmp_path / "second", count=1)
    with pytest.raises(ValueError, match="localhost"):
        serve_human_mask_annotation(
            clean, "reviewer-a", tmp_path / "serve", host="0.0.0.0", port=8765
        )


def test_three_complete_independent_manifests_merge_hand_calculably(
    tmp_path: Path,
) -> None:
    tasks = _blinded_tasks(tmp_path)
    sessions = [
        _complete_session(tasks, tmp_path / annotator, annotator)
        for annotator in ("reviewer-a", "reviewer-b", "reviewer-c")
    ]
    output = tmp_path / "completed-human-annotations.jsonl"
    report = merge_human_mask_annotation_manifests(
        tasks, [session.final_manifest for session in sessions], output
    )
    assert report["tasks"] == 2
    assert report["annotations"] == 6
    assert report["annotators"] == ["reviewer-a", "reviewer-b", "reviewer-c"]
    assert report["independent_annotator_ids"] is True
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 6
    assert [row["audit_id"] for row in rows] == [
        "audit-0",
        "audit-0",
        "audit-0",
        "audit-1",
        "audit-1",
        "audit-1",
    ]
    with pytest.raises(ValueError, match="independent annotator"):
        merge_human_mask_annotation_manifests(
            tasks,
            [sessions[0].final_manifest] * 3,
            tmp_path / "duplicate.jsonl",
        )


def test_local_annotation_http_roundtrip_including_empty_finalize_post(
    tmp_path: Path,
) -> None:
    tasks = _blinded_tasks(tmp_path, count=1)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    thread = threading.Thread(
        target=serve_human_mask_annotation,
        args=(tasks, "reviewer-http", tmp_path / "http", "127.0.0.1", port),
        daemon=True,
    )
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{base}/api/state") as response:
                state = json.load(response)
            break
        except OSError:
            time.sleep(0.02)
    else:
        raise AssertionError("annotation server did not start")
    assert state["target_fields_exposed"] is False
    with urllib.request.urlopen(f"{base}/api/task?index=0") as response:
        task = json.load(response)
    assert task["plane_labels"] == ["H1 / Q=4", "L1 / Q=4"]
    save = urllib.request.Request(
        f"{base}/api/task?index=0",
        data=json.dumps({"mask": [1, 0, 0, 1, 0, 1, 1, 0]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(save) as response:
        saved = json.load(response)
    assert saved["completed"] == 1
    finalize = urllib.request.Request(f"{base}/api/finalize", data=b"", method="POST")
    with urllib.request.urlopen(finalize) as response:
        final = json.load(response)
    assert final["status"] == "completed_independent_blinded_human_mask_annotation"
