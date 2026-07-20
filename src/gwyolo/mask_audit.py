from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .metrics import wilson_interval


def binary_mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left, dtype=bool)
    right_mask = np.asarray(right, dtype=bool)
    if left_mask.shape != right_mask.shape or left_mask.size == 0:
        raise ValueError("binary masks must be non-empty and aligned")
    union = int(np.count_nonzero(left_mask | right_mask))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(left_mask & right_mask)) / union


def _load_npz_mask(path: str | Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise ValueError(f"mask file {path} lacks key {key}")
        mask = np.asarray(arrays[key])
    if mask.size == 0 or not np.isfinite(mask).all():
        raise ValueError(f"mask file {path} is empty or non-finite")
    if np.any((mask < 0) | (mask > 1)):
        raise ValueError(f"mask file {path} lies outside [0,1]")
    return mask >= 0.5


def plan_gravityspy_mask_audit(
    numeric_manifest: str | Path,
    output_dir: str | Path,
    per_label: int = 5,
    seed: int = 20260720,
) -> dict[str, Any]:
    if per_label <= 0:
        raise ValueError("mask audit per-label target must be positive")
    with Path(numeric_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("split") != "val" for row in rows):
        raise ValueError("mask audit planning accepts a non-empty validation-only manifest")
    if len({str(row["glitch_id"]) for row in rows}) != len(rows):
        raise ValueError("mask audit source contains duplicate glitch IDs")
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["ml_label"])].append(row)
    selected = []
    underfilled = {}
    for label, label_rows in sorted(by_label.items()):
        ordered = sorted(
            label_rows,
            key=lambda row: canonical_hash(
                {"seed": seed, "glitch_id": row["glitch_id"]}, 64
            ),
        )
        selected.extend(ordered[:per_label])
        if len(ordered) < per_label:
            underfilled[label] = per_label - len(ordered)
    tasks = []
    for row in sorted(selected, key=lambda item: str(item["glitch_id"])):
        if file_sha256(row["path"]) != str(row["sha256"]):
            raise ValueError(f"numeric sample hash mismatch: {row['glitch_id']}")
        weak_mask = _load_npz_mask(row["path"], "glitch_mask")
        tasks.append(
            {
                "audit_id": f"mask-audit-{canonical_hash(row['glitch_id'], 24)}",
                "glitch_id": str(row["glitch_id"]),
                "ml_label": str(row["ml_label"]),
                "ifo": str(row["ifo"]),
                "observing_run": str(row["observing_run"]),
                "network_gps_block": str(row["network_gps_block"]),
                "numeric_sample_path": str(row["path"]),
                "numeric_sample_sha256": str(row["sha256"]),
                "weak_mask_key": "glitch_mask",
                "mask_shape": list(weak_mask.shape),
                "required_independent_annotators": 3,
                "required_annotation_key": "mask",
                "blinding_requirement": "annotator must not inspect the weak mask",
                "annotation_status": "pending",
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "gravityspy_mask_audit_tasks.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
    )
    report = {
        "status": "frozen_gravityspy_human_mask_audit_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "tasks require three independent blinded human masks before weak-mask quality is known"
        ),
        "seed": seed,
        "per_label_target": per_label,
        "target_met": not underfilled,
        "underfilled_label_deficits": dict(sorted(underfilled.items())),
        "source_manifest_path": str(numeric_manifest),
        "source_manifest_sha256": file_sha256(numeric_manifest),
        "tasks": len(tasks),
        "label_counts": dict(sorted(Counter(task["ml_label"] for task in tasks).items())),
        "unique_glitches": len({task["glitch_id"] for task in tasks}),
        "unique_network_gps_blocks": len(
            {task["network_gps_block"] for task in tasks}
        ),
        "task_manifest_path": str(target),
        "task_manifest_sha256": file_sha256(target),
    }
    atomic_write_json(output / "gravityspy_mask_audit_plan_report.json", report)
    return report


def _audit_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weak_ious = np.asarray([row["weak_consensus_iou"] for row in rows])
    inter_ious = np.asarray([row["mean_interannotator_iou"] for row in rows])
    successes = int(np.count_nonzero(weak_ious >= 0.5))
    interval = wilson_interval(successes, len(rows))
    return {
        "tasks": len(rows),
        "weak_consensus_iou_mean": float(weak_ious.mean()),
        "weak_consensus_iou_median": float(np.median(weak_ious)),
        "mean_interannotator_iou": float(inter_ious.mean()),
        "weak_consensus_iou_ge_0_5": successes,
        "weak_consensus_iou_ge_0_5_fraction": successes / len(rows),
        "weak_consensus_iou_ge_0_5_wilson_95": list(interval),
    }


def evaluate_gravityspy_mask_audit(
    task_manifest: str | Path,
    annotation_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    with Path(task_manifest).open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    with Path(annotation_manifest).open("r", encoding="utf-8") as handle:
        annotations = [json.loads(line) for line in handle if line.strip()]
    if not tasks or not annotations:
        raise ValueError("mask audit tasks and annotations must be non-empty")
    task_by_id = {str(task["audit_id"]): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("mask audit task IDs are not unique")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        audit_id = str(row["audit_id"])
        if audit_id not in task_by_id:
            raise ValueError(f"annotation references unknown mask audit task: {audit_id}")
        if row.get("blinded_to_weak_mask") is not True:
            raise ValueError(f"annotation is not declared weak-mask-blinded: {audit_id}")
        if not str(row.get("protocol_version", "")):
            raise ValueError(f"annotation lacks protocol version: {audit_id}")
        if file_sha256(row["mask_path"]) != str(row["mask_sha256"]):
            raise ValueError(f"human mask hash mismatch: {audit_id}")
        by_task[audit_id].append(row)
    missing = sorted(set(task_by_id) - set(by_task))
    if missing:
        raise ValueError(f"mask audit tasks lack annotations: {missing[:10]}")
    evaluated = []
    for audit_id, task in sorted(task_by_id.items()):
        task_annotations = by_task[audit_id]
        annotators = [str(row["annotator_id"]) for row in task_annotations]
        required_annotators = int(task["required_independent_annotators"])
        if required_annotators < 3 or required_annotators % 2 == 0:
            raise ValueError(f"mask audit requires an odd consensus panel >=3: {audit_id}")
        if (
            len(set(annotators)) != len(annotators)
            or len(annotators) < required_annotators
            or len(annotators) % 2 == 0
        ):
            raise ValueError(f"mask audit lacks independent annotators: {audit_id}")
        human_masks = [
            _load_npz_mask(row["mask_path"], task["required_annotation_key"])
            for row in task_annotations
        ]
        expected_shape = tuple(int(value) for value in task["mask_shape"])
        if any(mask.shape != expected_shape for mask in human_masks):
            raise ValueError(f"human mask shape mismatch: {audit_id}")
        weak_mask = _load_npz_mask(task["numeric_sample_path"], task["weak_mask_key"])
        consensus = np.mean(np.stack(human_masks), axis=0) >= 0.5
        pairwise = [
            binary_mask_iou(left, right)
            for left, right in itertools.combinations(human_masks, 2)
        ]
        evaluated.append(
            {
                "audit_id": audit_id,
                "glitch_id": task["glitch_id"],
                "ml_label": task["ml_label"],
                "annotators": sorted(annotators),
                "mean_interannotator_iou": float(np.mean(pairwise)),
                "minimum_interannotator_iou": float(np.min(pairwise)),
                "weak_consensus_iou": binary_mask_iou(weak_mask, consensus),
                "weak_positive_fraction": float(np.mean(weak_mask)),
                "consensus_positive_fraction": float(np.mean(consensus)),
            }
        )
    by_label = {
        label: _audit_group_summary(
            [row for row in evaluated if row["ml_label"] == label]
        )
        for label in sorted({row["ml_label"] for row in evaluated})
    }
    result = {
        "status": "completed_blinded_gravityspy_human_mask_audit",
        "scientific_claim_allowed": True,
        "claim_scope": (
            "weak-mask agreement on this frozen validation audit only; model segmentation and "
            "search/deglitch benefit require separate locked evaluations"
        ),
        "task_manifest_path": str(task_manifest),
        "task_manifest_sha256": file_sha256(task_manifest),
        "annotation_manifest_path": str(annotation_manifest),
        "annotation_manifest_sha256": file_sha256(annotation_manifest),
        "tasks": len(evaluated),
        "overall": _audit_group_summary(evaluated),
        "by_label": by_label,
        "evaluated_tasks": evaluated,
    }
    atomic_write_json(output_path, result)
    return result
