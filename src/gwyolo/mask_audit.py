from __future__ import annotations

import itertools
import json
import os
import platform
import shlex
import sys
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


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def plan_gravityspy_mask_audit(
    numeric_manifest: str | Path,
    output_dir: str | Path,
    per_label: int = 5,
    seed: int = 20260720,
) -> dict[str, Any]:
    if per_label <= 0:
        raise ValueError("mask audit per-label target must be positive")
    output = Path(output_dir).resolve()
    target = output / "gravityspy_mask_audit_tasks.jsonl"
    annotation_target = output / "gravityspy_mask_annotation_tasks.jsonl"
    report_target = output / "gravityspy_mask_audit_plan_report.json"
    if target.exists() or annotation_target.exists() or report_target.exists():
        raise FileExistsError("Gravity Spy human-mask audit plans are immutable")
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
    annotation_tasks = []
    for row in sorted(selected, key=lambda item: str(item["glitch_id"])):
        if file_sha256(row["path"]) != str(row["sha256"]):
            raise ValueError(f"numeric sample hash mismatch: {row['glitch_id']}")
        weak_mask = _load_npz_mask(row["path"], "glitch_mask")
        audit_id = f"mask-audit-{canonical_hash(row['glitch_id'], 24)}"
        with np.load(row["path"], allow_pickle=False) as arrays:
            if "features" not in arrays:
                raise ValueError(f"numeric sample lacks features: {row['glitch_id']}")
            features = np.asarray(arrays["features"])
            if features.shape != weak_mask.shape or not np.isfinite(features).all():
                raise ValueError(f"numeric feature/mask shape mismatch: {row['glitch_id']}")
            blind_arrays = {
                key: np.asarray(arrays[key])
                for key in ("features", "ifos", "q_values", "sample_rate")
                if key in arrays
            }
        blind_path = output / "blinded_inputs" / f"{audit_id}.npz"
        _atomic_save_npz(blind_path, blind_arrays)
        annotation_task = {
            "audit_id": audit_id,
            "blinded_input_path": str(blind_path),
            "blinded_input_sha256": file_sha256(blind_path),
            "blinded_input_keys": sorted(blind_arrays),
            "mask_shape": list(weak_mask.shape),
            "required_independent_annotators": 3,
            "required_annotation_key": "mask",
            "blinding_requirement": (
                "annotator may access only blinded_input_path; all mask targets are excluded"
            ),
            "annotation_status": "pending",
        }
        annotation_task_hash = canonical_hash(annotation_task, 64)
        annotation_task["annotation_task_hash"] = annotation_task_hash
        annotation_tasks.append(annotation_task)
        tasks.append(
            {
                "audit_id": audit_id,
                "glitch_id": str(row["glitch_id"]),
                "ml_label": str(row["ml_label"]),
                "ifo": str(row["ifo"]),
                "observing_run": str(row["observing_run"]),
                "network_gps_block": str(row["network_gps_block"]),
                "numeric_sample_path": str(row["path"]),
                "numeric_sample_sha256": str(row["sha256"]),
                "weak_mask_key": "glitch_mask",
                **annotation_task,
                "annotation_task_hash": annotation_task_hash,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        target, "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
    )
    atomic_write_text(
        annotation_target,
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in annotation_tasks),
    )
    report = {
        "status": "frozen_gravityspy_human_mask_audit_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "tasks require three independent blinded human masks before weak-mask quality is known"
        ),
        "seed": seed,
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "config_hash": None,
        "model_hash": None,
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
        "annotation_task_manifest_path": str(annotation_target),
        "annotation_task_manifest_sha256": file_sha256(annotation_target),
        "blinded_inputs": len(annotation_tasks),
        "mask_targets_exposed_to_annotators": False,
    }
    atomic_write_json(report_target, report)
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


def _validated_mask_audit_inputs(
    task_manifest: str | Path,
    annotation_manifest: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    with Path(task_manifest).open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    with Path(annotation_manifest).open("r", encoding="utf-8") as handle:
        annotations = [json.loads(line) for line in handle if line.strip()]
    if not tasks or not annotations:
        raise ValueError("mask audit tasks and annotations must be non-empty")
    task_by_id = {str(task["audit_id"]): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("mask audit task IDs are not unique")
    forbidden = {"mask", "glitch_mask", "chirp_mask"}
    for audit_id, task in task_by_id.items():
        blind_path = Path(str(task.get("blinded_input_path", "")))
        if (
            not blind_path.is_file()
            or file_sha256(blind_path) != str(task.get("blinded_input_sha256", ""))
        ):
            raise ValueError(f"mask audit blinded input hash mismatch: {audit_id}")
        with np.load(blind_path, allow_pickle=False) as arrays:
            keys = set(arrays.files)
            if "features" not in keys or keys & forbidden:
                raise ValueError(f"mask audit blinded input exposes a target: {audit_id}")
            if list(np.asarray(arrays["features"]).shape) != list(task["mask_shape"]):
                raise ValueError(f"mask audit blinded input shape mismatch: {audit_id}")
        if sorted(keys) != sorted(task.get("blinded_input_keys", [])):
            raise ValueError(f"mask audit blinded input key inventory mismatch: {audit_id}")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    protocol_versions = set()
    for row in annotations:
        audit_id = str(row["audit_id"])
        if audit_id not in task_by_id:
            raise ValueError(f"annotation references unknown mask audit task: {audit_id}")
        if row.get("blinded_to_weak_mask") is not True:
            raise ValueError(f"annotation is not declared weak-mask-blinded: {audit_id}")
        if not str(row.get("protocol_version", "")):
            raise ValueError(f"annotation lacks protocol version: {audit_id}")
        protocol_versions.add(str(row["protocol_version"]))
        if str(row.get("annotation_task_hash", "")) != str(
            task_by_id[audit_id].get("annotation_task_hash", "")
        ):
            raise ValueError(f"annotation task hash mismatch: {audit_id}")
        if file_sha256(row["mask_path"]) != str(row["mask_sha256"]):
            raise ValueError(f"human mask hash mismatch: {audit_id}")
        by_task[audit_id].append(row)
    missing = sorted(set(task_by_id) - set(by_task))
    if missing:
        raise ValueError(f"mask audit tasks lack annotations: {missing[:10]}")
    if len(protocol_versions) != 1:
        raise ValueError("mask audit annotations mix protocol versions")
    return task_by_id, dict(by_task)


def _mask_audit_consensus_records(
    task_by_id: dict[str, dict[str, Any]],
    by_task: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    evaluated = []
    consensus_by_id = {}
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
        source_path = Path(task["numeric_sample_path"])
        if file_sha256(source_path) != str(task.get("numeric_sample_sha256", "")):
            raise ValueError(f"numeric weak-mask sample hash mismatch: {audit_id}")
        weak_mask = _load_npz_mask(source_path, task["weak_mask_key"])
        consensus = np.mean(np.stack(human_masks), axis=0) >= 0.5
        consensus_by_id[audit_id] = consensus
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
                "protocol_version": str(task_annotations[0]["protocol_version"]),
                "mean_interannotator_iou": float(np.mean(pairwise)),
                "minimum_interannotator_iou": float(np.min(pairwise)),
                "weak_consensus_iou": binary_mask_iou(weak_mask, consensus),
                "weak_positive_fraction": float(np.mean(weak_mask)),
                "consensus_positive_fraction": float(np.mean(consensus)),
            }
        )
    return evaluated, consensus_by_id


def evaluate_gravityspy_mask_audit(
    task_manifest: str | Path,
    annotation_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("Gravity Spy human-mask audit reports are immutable")
    task_by_id, by_task = _validated_mask_audit_inputs(
        task_manifest, annotation_manifest
    )
    evaluated, _ = _mask_audit_consensus_records(task_by_id, by_task)
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
    atomic_write_json(target, result)
    return result


def materialize_gravityspy_mask_consensus(
    task_manifest: str | Path,
    annotation_manifest: str | Path,
    audit_report_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize a validation-only human-consensus mask bank for locked metrics."""

    output = Path(output_dir).resolve()
    report_path = output / "gravityspy_human_consensus_mask_report.json"
    manifest_path = output / "gravityspy_human_consensus_masks.jsonl"
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError("Gravity Spy human-consensus mask banks are immutable")
    audit_path = Path(audit_report_path).resolve()
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if (
        audit.get("status") != "completed_blinded_gravityspy_human_mask_audit"
        or audit.get("task_manifest_sha256") != file_sha256(task_manifest)
        or audit.get("annotation_manifest_sha256") != file_sha256(annotation_manifest)
    ):
        raise ValueError("human-consensus materialization requires the exact completed audit")
    task_by_id, by_task = _validated_mask_audit_inputs(
        task_manifest, annotation_manifest
    )
    evaluated, consensus_by_id = _mask_audit_consensus_records(task_by_id, by_task)
    if audit.get("evaluated_tasks") != evaluated or int(audit.get("tasks", -1)) != len(
        evaluated
    ):
        raise ValueError("human-consensus metrics differ from the completed audit")

    rows = []
    mask_root = output / "masks"
    for record in evaluated:
        audit_id = str(record["audit_id"])
        task = task_by_id[audit_id]
        target = mask_root / f"{audit_id}.npz"
        _atomic_save_npz(
            target, {"mask": np.asarray(consensus_by_id[audit_id], dtype=np.uint8)}
        )
        rows.append(
            {
                "audit_id": audit_id,
                "glitch_id": str(task["glitch_id"]),
                "ml_label": str(task["ml_label"]),
                "split": "val",
                "training_allowed": False,
                "human_pixel_mask": True,
                "mask_key": "mask",
                "mask_shape": list(consensus_by_id[audit_id].shape),
                "path": str(target),
                "sha256": file_sha256(target),
                "annotators": record["annotators"],
                "protocol_version": record["protocol_version"],
                "annotation_task_hash": str(task["annotation_task_hash"]),
                "numeric_sample_path": str(task["numeric_sample_path"]),
                "numeric_sample_sha256": str(task["numeric_sample_sha256"]),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    result = {
        "status": "verified_gravityspy_human_consensus_mask_bank",
        "scientific_claim_allowed": False,
        "training_allowed": False,
        "claim_scope": "validation-only human-consensus segmentation reference",
        "scientific_blocker": (
            "model segmentation, deglitch and search benefit require separate frozen evaluations"
        ),
        "task_manifest_path": str(Path(task_manifest).resolve()),
        "task_manifest_sha256": file_sha256(task_manifest),
        "annotation_manifest_path": str(Path(annotation_manifest).resolve()),
        "annotation_manifest_sha256": file_sha256(annotation_manifest),
        "audit_report_path": str(audit_path),
        "audit_report_sha256": file_sha256(audit_path),
        "tasks": len(rows),
        "unique_glitches": len({row["glitch_id"] for row in rows}),
        "labels": dict(sorted(Counter(row["ml_label"] for row in rows).items())),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    atomic_write_json(report_path, result)
    return result
