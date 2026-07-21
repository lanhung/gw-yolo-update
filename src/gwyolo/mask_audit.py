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


def binary_mask_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Return hand-auditable per-mask counts and overlap metrics."""

    target = np.asarray(expected, dtype=bool)
    estimate = np.asarray(predicted, dtype=bool)
    if target.shape != estimate.shape or target.size == 0:
        raise ValueError("binary masks must be non-empty and aligned")
    true_positive = int(np.count_nonzero(target & estimate))
    false_positive = int(np.count_nonzero(~target & estimate))
    false_negative = int(np.count_nonzero(target & ~estimate))
    true_negative = int(np.count_nonzero(~target & ~estimate))
    predicted_positive = true_positive + false_positive
    target_positive = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    precision = (
        true_positive / predicted_positive
        if predicted_positive
        else (1.0 if target_positive == 0 else 0.0)
    )
    recall = true_positive / target_positive if target_positive else 1.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "iou": true_positive / union if union else 1.0,
        "dice": 2 * true_positive / dice_denominator if dice_denominator else 1.0,
    }


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


def _load_npz_probability(path: str | Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise ValueError(f"prediction file {path} lacks key {key}")
        probability = np.asarray(arrays[key], dtype=np.float64)
    if probability.size == 0 or not np.isfinite(probability).all():
        raise ValueError(f"prediction file {path} is empty or non-finite")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"prediction file {path} lies outside [0,1]")
    return probability


def _mask_segmentation_summary(
    rows: list[dict[str, Any]], bootstrap_replicates: int, seed: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("mask segmentation summary requires rows")
    metric_names = ("precision", "recall", "iou", "dice")
    macro = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    pooled_counts = {
        name: sum(int(row[name]) for row in rows)
        for name in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
        )
    }
    tp = pooled_counts["true_positive"]
    fp = pooled_counts["false_positive"]
    fn = pooled_counts["false_negative"]
    pooled = {
        **pooled_counts,
        "precision": tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0),
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    }
    intervals: dict[str, list[float | None]] = {}
    if len(rows) < 2:
        intervals = {name: [None, None] for name in metric_names}
    else:
        values = np.asarray(
            [[float(row[name]) for name in metric_names] for row in rows],
            dtype=np.float64,
        )
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(rows), size=(bootstrap_replicates, len(rows)))
        estimates = values[indices].mean(axis=1)
        for index, name in enumerate(metric_names):
            intervals[name] = [
                float(np.percentile(estimates[:, index], 2.5)),
                float(np.percentile(estimates[:, index], 97.5)),
            ]
    successes = sum(float(row["iou"]) >= 0.5 for row in rows)
    return {
        "tasks": len(rows),
        "macro": macro,
        "macro_paired_task_bootstrap_95": intervals,
        "pooled_pixels": pooled,
        "iou_ge_0_5": successes,
        "iou_ge_0_5_fraction": successes / len(rows),
        "iou_ge_0_5_wilson_95": list(wilson_interval(successes, len(rows))),
    }


def evaluate_gravityspy_mask_segmentation(
    gold_report_path: str | Path,
    prediction_manifest_path: str | Path,
    output_path: str | Path,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    """Evaluate validation-only model masks against a blinded human-consensus bank."""

    if bootstrap_replicates < 100 or bootstrap_seed < 0:
        raise ValueError("mask segmentation evaluation needs >=100 bootstrap replicates and a seed")
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("Gravity Spy mask segmentation reports are immutable")
    gold_report_path = Path(gold_report_path).resolve()
    gold_report = json.loads(gold_report_path.read_text(encoding="utf-8"))
    gold_manifest = Path(str(gold_report.get("manifest_path", "")))
    if (
        gold_report.get("status") != "verified_gravityspy_human_consensus_mask_bank"
        or gold_report.get("training_allowed") is not False
        or not gold_manifest.is_file()
        or file_sha256(gold_manifest) != str(gold_report.get("manifest_sha256", ""))
    ):
        raise ValueError("mask segmentation requires an exact validation-only consensus bank")
    with gold_manifest.open("r", encoding="utf-8") as handle:
        gold_rows = [json.loads(line) for line in handle if line.strip()]
    prediction_manifest = Path(prediction_manifest_path).resolve()
    with prediction_manifest.open("r", encoding="utf-8") as handle:
        predictions = [json.loads(line) for line in handle if line.strip()]
    if not gold_rows or not predictions:
        raise ValueError("gold and prediction mask manifests must be non-empty")
    if int(gold_report.get("tasks", -1)) != len(gold_rows):
        raise ValueError("human-consensus report and manifest row counts differ")
    gold_by_id = {str(row["audit_id"]): row for row in gold_rows}
    prediction_by_id = {str(row["audit_id"]): row for row in predictions}
    if len(gold_by_id) != len(gold_rows) or len(prediction_by_id) != len(predictions):
        raise ValueError("gold or prediction audit IDs are not unique")
    if set(gold_by_id) != set(prediction_by_id):
        raise ValueError("prediction manifest does not exactly cover the human gold bank")
    if len({str(row.get("path")) for row in gold_rows}) != len(gold_rows) or len(
        {str(row.get("path")) for row in predictions}
    ) != len(predictions):
        raise ValueError("gold or prediction mask paths are not unique")

    required_artifacts = (
        ("model_path", "model_sha256"),
        ("config_path", "config_sha256"),
        ("checkpoint_selection_report_path", "checkpoint_selection_report_sha256"),
    )
    identities = set()
    artifact_digests: dict[Path, str] = {}
    inspected_selection_reports: set[Path] = set()
    evaluated = []
    for audit_id, gold in sorted(gold_by_id.items()):
        prediction = prediction_by_id[audit_id]
        checkpoint_split = str(prediction.get("checkpoint_selection_split", ""))
        threshold_split = str(prediction.get("threshold_selection_split", ""))
        if (
            gold.get("split") != "val"
            or gold.get("training_allowed") is not False
            or gold.get("human_pixel_mask") is not True
            or prediction.get("split") != "val"
            or checkpoint_split not in {"calibration", "val"}
            or threshold_split not in {"calibration", "val"}
        ):
            raise ValueError("human-consensus segmentation evaluation is validation-only")
        if str(gold["glitch_id"]) != str(prediction.get("glitch_id")):
            raise ValueError(f"prediction glitch identity mismatch: {audit_id}")
        for path_field, hash_field in required_artifacts:
            artifact = Path(str(prediction.get(path_field, "")))
            if not artifact.is_file():
                raise ValueError(f"prediction artifact hash mismatch: {path_field}")
            if artifact not in artifact_digests:
                artifact_digests[artifact] = file_sha256(artifact)
            if artifact_digests[artifact] != str(prediction.get(hash_field, "")):
                raise ValueError(f"prediction artifact hash mismatch: {path_field}")
            if (
                path_field == "checkpoint_selection_report_path"
                and artifact not in inspected_selection_reports
            ):
                selection_report = json.loads(artifact.read_text(encoding="utf-8"))
                forbidden_test_selection = (
                    selection_report.get("test_evaluation") is True
                    or selection_report.get("selected_split") == "test"
                    or selection_report.get("selection_split") == "test"
                    or selection_report.get("test_metrics") not in (None, {})
                )
                if forbidden_test_selection:
                    raise ValueError("checkpoint selection report contains test selection")
                inspected_selection_reports.add(artifact)
        threshold = float(prediction.get("threshold", -1))
        if not 0 <= threshold <= 1:
            raise ValueError(f"prediction threshold lies outside [0,1]: {audit_id}")
        prediction_path = Path(str(prediction.get("path", "")))
        if not prediction_path.is_file() or file_sha256(prediction_path) != str(
            prediction.get("sha256", "")
        ):
            raise ValueError(f"prediction mask hash mismatch: {audit_id}")
        gold_path = Path(str(gold.get("path", "")))
        if not gold_path.is_file() or file_sha256(gold_path) != str(gold.get("sha256", "")):
            raise ValueError(f"gold mask hash mismatch: {audit_id}")
        expected = _load_npz_mask(gold_path, str(gold["mask_key"]))
        probability = _load_npz_probability(
            prediction_path, str(prediction.get("mask_key", "mask_probability"))
        )
        if probability.shape != expected.shape or list(expected.shape) != list(
            gold.get("mask_shape", [])
        ):
            raise ValueError(f"prediction/gold mask shape mismatch: {audit_id}")
        identity = {
            field: prediction[field]
            for pair in required_artifacts
            for field in pair
        }
        identity.update(
            {
                "threshold": threshold,
                "prediction_mask_key": str(
                    prediction.get("mask_key", "mask_probability")
                ),
                "checkpoint_selection_split": checkpoint_split,
                "threshold_selection_split": threshold_split,
            }
        )
        identities.add(canonical_hash(identity, 64))
        evaluated.append(
            {
                "audit_id": audit_id,
                "glitch_id": str(gold["glitch_id"]),
                "ml_label": str(gold["ml_label"]),
                **binary_mask_metrics(expected, probability >= threshold),
            }
        )
    if len(identities) != 1:
        raise ValueError("prediction rows do not share one model/config/threshold identity")

    labels = sorted({str(row["ml_label"]) for row in evaluated})
    by_label = {
        label: _mask_segmentation_summary(
            [row for row in evaluated if row["ml_label"] == label],
            bootstrap_replicates,
            bootstrap_seed + index + 1,
        )
        for index, label in enumerate(labels)
    }
    first_prediction = prediction_by_id[sorted(prediction_by_id)[0]]
    result = {
        "status": "completed_validation_human_consensus_mask_segmentation",
        "scientific_claim_allowed": False,
        "promotion_evidence_allowed": True,
        "claim_scope": (
            "validation-only human-consensus glitch-mask segmentation; locked search, deglitch "
            "and test evaluation remain separate"
        ),
        "test_evaluation": False,
        "gold_report_path": str(gold_report_path),
        "gold_report_sha256": file_sha256(gold_report_path),
        "gold_manifest_path": str(gold_manifest),
        "gold_manifest_sha256": file_sha256(gold_manifest),
        "prediction_manifest_path": str(prediction_manifest),
        "prediction_manifest_sha256": file_sha256(prediction_manifest),
        "prediction_identity_hash": next(iter(identities)),
        "model_sha256": str(first_prediction["model_sha256"]),
        "config_sha256": str(first_prediction["config_sha256"]),
        "checkpoint_selection_report_sha256": str(
            first_prediction["checkpoint_selection_report_sha256"]
        ),
        "threshold": float(first_prediction["threshold"]),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "tasks": len(evaluated),
        "overall": _mask_segmentation_summary(
            evaluated, bootstrap_replicates, bootstrap_seed
        ),
        "by_label": by_label,
        "evaluated_tasks": evaluated,
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    atomic_write_json(output, result)
    return result
