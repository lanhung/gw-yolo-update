from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, file_sha256


BUNDLE_SCHEMA = "portable_detector_set_training_inputs_v1"
OBJECT_PREFIX = "object-sha256:"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"training transfer manifest is empty: {path}")
    return rows


def _object_path(root: Path, sha256: str) -> Path:
    return root / "objects" / sha256[:2] / sha256


def _add_object(
    source: str | Path,
    root: Path,
    objects: dict[str, dict[str, Any]],
    expected_sha256: str | None = None,
) -> str:
    if expected_sha256 is not None and expected_sha256 in objects:
        return f"{OBJECT_PREFIX}{expected_sha256}"
    path = Path(source).resolve()
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"training bundle source hash mismatch: {path}")
    target = _object_path(root, actual)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".part")
        try:
            os.link(path, temporary)
        except OSError:
            shutil.copy2(path, temporary)
        os.replace(temporary, target)
    if file_sha256(target) != actual:
        raise ValueError("training bundle object failed post-copy verification")
    objects[actual] = {
        "path": str(target.relative_to(root)),
        "bytes": target.stat().st_size,
    }
    return f"{OBJECT_PREFIX}{actual}"


def _bundle_overlap_manifest(
    path: str | Path,
    label: str,
    root: Path,
    objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = Path(path).resolve()
    rows = _read_jsonl(source)
    for row in rows:
        row["path"] = _add_object(row["path"], root, objects, str(row["sha256"]))
    target = root / "manifests" / f"{label}.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return {
        "kind": "physical_overlap",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "template_path": str(target.relative_to(root)),
        "template_sha256": file_sha256(target),
        "rows": len(rows),
    }


def _bundle_clean_manifest(
    path: str | Path,
    label: str,
    root: Path,
    objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = Path(path).resolve()
    rows = _read_jsonl(source)
    seen_injection_ids = set()
    seen_waveform_ids = set()
    verified_sources: dict[str, str] = {}
    for row in rows:
        injection_id = str(row["injection_id"])
        waveform_id = str(row["waveform_id"])
        if injection_id in seen_injection_ids or waveform_id in seen_waveform_ids:
            raise ValueError("clean training manifest repeats a physical identity")
        seen_injection_ids.add(injection_id)
        seen_waveform_ids.add(waveform_id)
        row["materialized_path"] = _add_object(
            row["materialized_path"],
            root,
            objects,
            str(row["materialized_sha256"]),
        )
        bank = row.get("background_bank")
        if bank:
            bank["path"] = _add_object(
                bank["path"], root, objects, str(bank["sha256"])
            )
        else:
            sources = row.get("background_source_files")
            if not isinstance(sources, dict) or not sources:
                raise ValueError("clean row lacks reconstructable background sources")
            for ifo, source_ref in sources.items():
                source_path = str(Path(source_ref["path"]).resolve())
                expected = str(source_ref["sha256"])
                observed = verified_sources.get(source_path)
                if observed is None:
                    observed = file_sha256(source_path)
                    verified_sources[source_path] = observed
                if observed != expected:
                    raise ValueError(f"clean background source hash mismatch for {ifo}")
                source_ref["path"] = _add_object(
                    source_path, root, objects, expected
                )
        if not injection_id or not waveform_id:
            raise ValueError("clean training identity is empty")
    target = root / "manifests" / f"{label}.jsonl"
    atomic_write_text(
        target, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    return {
        "kind": "clean_physical_injection",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "template_path": str(target.relative_to(root)),
        "template_sha256": file_sha256(target),
        "rows": len(rows),
        "unique_background_source_objects": len(set(verified_sources.values())),
    }


def export_detector_set_training_bundle(
    overlap_receipt_path: str | Path,
    clean_train_manifest_path: str | Path,
    clean_validation_manifest_path: str | Path,
    pretrained_checkpoint_path: str | Path,
    configs: dict[str, str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a content-addressed, path-portable detector-set training bundle."""

    if not configs or any(not label for label in configs):
        raise ValueError("detector-set training bundle requires labeled configs")
    root = Path(output_dir).resolve()
    receipt_target = root / "detector_set_training_input_bundle.json"
    overlap_receipt_path = Path(overlap_receipt_path).resolve()
    overlap_receipt = json.loads(overlap_receipt_path.read_text(encoding="utf-8"))
    if (
        overlap_receipt.get("status")
        != "verified_detector_set_overlap_robustness_corpus"
        or overlap_receipt.get("passed") is not True
        or overlap_receipt.get("test_rows_read") != 0
        or overlap_receipt.get("test_evaluation") is not None
        or overlap_receipt.get("same_distribution_data_scaling_claim_allowed")
        is not False
    ):
        raise ValueError("detector-set overlap receipt failed portable replay")
    artifacts = overlap_receipt.get("artifacts", {})
    required = {
        "train_report",
        "validation_report",
        "joint_group_audit",
        "expansion_readiness_audit",
        "capacity_report",
    }
    if set(artifacts) != required:
        raise ValueError("detector-set overlap receipt has an incomplete artifact graph")
    loaded_artifacts = {}
    for label, identity in artifacts.items():
        path = Path(str(identity["path"])).resolve()
        if file_sha256(path) != identity["sha256"]:
            raise ValueError(f"detector-set overlap receipt artifact drift: {label}")
        loaded_artifacts[label] = json.loads(path.read_text(encoding="utf-8"))
    readiness = loaded_artifacts["expansion_readiness_audit"]
    if (
        readiness.get("status") != "audited_detector_set_signal_bank_readiness"
        or readiness.get("signal_overlap_materialization_authorized") is not True
        or readiness.get("detector_complete_clean_training_authorized") is not False
        or readiness.get("detector_set_robustness_ablation_ready") is not False
    ):
        raise ValueError("training bundle requires the signal-only readiness boundary")
    train_report = loaded_artifacts["train_report"]
    validation_report = loaded_artifacts["validation_report"]
    if (
        train_report.get("split") != "train"
        or validation_report.get("split") != "val"
        or train_report.get("manifest_sha256")
        != file_sha256(train_report["manifest_path"])
        or validation_report.get("manifest_sha256")
        != file_sha256(validation_report["manifest_path"])
    ):
        raise ValueError("overlap report manifests failed portable replay")

    objects: dict[str, dict[str, Any]] = {}
    manifests = {
        "overlap_train": _bundle_overlap_manifest(
            train_report["manifest_path"], "overlap_train", root, objects
        ),
        "overlap_validation": _bundle_overlap_manifest(
            validation_report["manifest_path"],
            "overlap_validation",
            root,
            objects,
        ),
        "clean_train": _bundle_clean_manifest(
            clean_train_manifest_path, "clean_train", root, objects
        ),
        "clean_validation": _bundle_clean_manifest(
            clean_validation_manifest_path, "clean_validation", root, objects
        ),
    }
    checkpoint_path = Path(pretrained_checkpoint_path).resolve()
    checkpoint_sha = file_sha256(checkpoint_path)
    checkpoint_marker = _add_object(
        checkpoint_path, root, objects, checkpoint_sha
    )
    portable_configs = {}
    for label, value in sorted(configs.items()):
        path = Path(value).resolve()
        sha256 = file_sha256(path)
        portable_configs[label] = {
            "source_path": str(path),
            "sha256": sha256,
            "object": _add_object(path, root, objects, sha256),
        }
    provenance_artifacts = {
        "overlap_receipt": {
            "source_path": str(overlap_receipt_path),
            "sha256": file_sha256(overlap_receipt_path),
            "object": _add_object(overlap_receipt_path, root, objects),
        }
    }
    for label, identity in sorted(artifacts.items()):
        provenance_artifacts[label] = {
            "source_path": str(Path(identity["path"]).resolve()),
            "sha256": identity["sha256"],
            "object": _add_object(
                identity["path"], root, objects, str(identity["sha256"])
            ),
        }
    result = {
        "status": "portable_detector_set_training_input_bundle",
        "passed": True,
        "schema": BUNDLE_SCHEMA,
        "scientific_claim_allowed": False,
        "detector_complete_clean_training_authorized": False,
        "scientific_blocker": (
            "bundle supports H1/L1 clean distillation plus variable-detector overlap "
            "training; detector-complete empirical-noise clean training and O4 transfer "
            "remain required"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "manifests": manifests,
        "checkpoint": {
            "source_path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "object": checkpoint_marker,
        },
        "configs": portable_configs,
        "provenance_artifacts": provenance_artifacts,
        "objects": dict(sorted(objects.items())),
        "object_count": len(objects),
        "object_bytes": sum(value["bytes"] for value in objects.values()),
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    atomic_write_json(receipt_target, result)
    return result


def _resolve_markers(value: Any, root: Path) -> Any:
    if isinstance(value, str) and value.startswith(OBJECT_PREFIX):
        sha256 = value[len(OBJECT_PREFIX) :]
        return str(_object_path(root, sha256).resolve())
    if isinstance(value, list):
        return [_resolve_markers(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_markers(item, root) for key, item in value.items()}
    return value


def import_detector_set_training_bundle(
    bundle_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify and project a detector-set training bundle on another machine."""

    receipt_path = Path(bundle_receipt_path).resolve()
    root = receipt_path.parent
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "portable_detector_set_training_input_bundle"
        or receipt.get("passed") is not True
        or receipt.get("schema") != BUNDLE_SCHEMA
        or receipt.get("test_rows_read") != 0
        or receipt.get("test_evaluation") is not None
        or receipt.get("detector_complete_clean_training_authorized") is not False
    ):
        raise ValueError("portable detector-set training receipt failed replay")
    objects = receipt.get("objects", {})
    if len(objects) != int(receipt.get("object_count", -1)):
        raise ValueError("portable detector-set training object inventory is incomplete")
    for sha256, identity in objects.items():
        path = root / str(identity["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(identity["bytes"])
            or file_sha256(path) != sha256
        ):
            raise ValueError(f"portable detector-set training object drift: {sha256}")

    output = Path(output_dir).resolve()
    projected_manifests = {}
    for label, identity in sorted(receipt["manifests"].items()):
        template = root / str(identity["template_path"])
        if file_sha256(template) != identity["template_sha256"]:
            raise ValueError(f"portable training manifest template drift: {label}")
        rows = _read_jsonl(template)
        projected = [_resolve_markers(row, root) for row in rows]
        target = output / "manifests" / f"{label}.jsonl"
        atomic_write_text(
            target,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in projected),
        )
        projected_manifests[label] = {
            "kind": identity["kind"],
            "path": str(target),
            "sha256": file_sha256(target),
            "rows": len(projected),
            "source_sha256": identity["source_sha256"],
        }
    checkpoint = _resolve_markers(receipt["checkpoint"]["object"], root)
    projected_configs = {
        label: _resolve_markers(identity["object"], root)
        for label, identity in sorted(receipt["configs"].items())
    }
    result = {
        "status": "projected_detector_set_training_input_bundle",
        "passed": True,
        "schema": BUNDLE_SCHEMA,
        "scientific_claim_allowed": False,
        "detector_complete_clean_training_authorized": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "bundle_receipt": {
            "path": str(receipt_path),
            "sha256": file_sha256(receipt_path),
        },
        "manifests": projected_manifests,
        "checkpoint": {
            "path": checkpoint,
            "sha256": receipt["checkpoint"]["sha256"],
        },
        "configs": projected_configs,
        "object_count": len(objects),
        "object_bytes": sum(int(value["bytes"]) for value in objects.values()),
        "code_commit": os.environ.get("GWYOLO_CODE_COMMIT"),
        "exact_command": " ".join(shlex.quote(part) for part in sys.argv),
    }
    atomic_write_json(output / "detector_set_training_inputs.json", result)
    return result
