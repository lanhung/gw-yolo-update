from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, file_sha256
from .runtime import execution_provenance


_BACKEND_SPECIFICATIONS = {
    "DINGO": {
        "summary_status": (
            "validation_only_dingo_official_native_paired_smoke_complete"
        ),
        "batch_status": (
            "real_dingo_official_native_paired_robustness_batch_complete"
        ),
    },
    "AMPLFI": {
        "summary_status": (
            "validation_only_amplfi_within_backend_paired_smoke_complete"
        ),
        "batch_status": "real_amplfi_common_batch_complete",
    },
}

_PORTABLE_PATH_HASH_FIELDS = (
    ("posterior_path", "posterior_sha256"),
    ("analysis_input_path", "analysis_input_sha256"),
    ("base_injection_manifest_path", "base_injection_manifest_sha256"),
    ("common_prior_path", "common_prior_sha256"),
    ("native_conditioning_path", "native_conditioning_sha256"),
    ("native_conditioning_config_path", "native_conditioning_config_sha256"),
    ("contamination_manifest_path", "contamination_manifest_sha256"),
    ("mask_artifact_path", "mask_artifact_sha256"),
    ("mask_model_path", "mask_model_sha256"),
    ("mask_policy_path", "mask_policy_sha256"),
)


def _metadata_identity_locations(
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    result = []
    for label, path_field, hash_field in (
        ("model", "model_path", "model_sha256"),
        (
            "initialization_model",
            "initialization_model_path",
            "initialization_model_sha256",
        ),
    ):
        path_value = metadata.get(path_field)
        hash_value = metadata.get(hash_field)
        if path_value in (None, "") and hash_value in (None, ""):
            continue
        if path_value in (None, "") or hash_value in (None, ""):
            raise ValueError(f"PE model metadata has an incomplete {label} identity")
        result.append(
            {
                "location": "top",
                "label": label,
                "path_field": path_field,
                "hash_field": hash_field,
                "path": str(path_value),
                "sha256": str(hash_value),
            }
        )
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("PE model metadata artifacts are not a mapping")
    for label, identity in sorted(artifacts.items()):
        if not isinstance(identity, dict):
            raise ValueError(f"PE model metadata artifact is not a mapping: {label}")
        path_value = identity.get("path")
        hash_value = identity.get("sha256")
        if path_value in (None, "") or hash_value in (None, ""):
            raise ValueError(f"PE model metadata artifact is incomplete: {label}")
        result.append(
            {
                "location": "artifact",
                "label": str(label),
                "path_field": "path",
                "hash_field": "sha256",
                "path": str(path_value),
                "sha256": str(hash_value),
            }
        )
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return value


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _backend_from_summary(summary: dict[str, Any]) -> str:
    matches = [
        backend
        for backend, specification in _BACKEND_SPECIFICATIONS.items()
        if summary.get("status") == specification["summary_status"]
    ]
    if len(matches) != 1:
        raise ValueError("Within-backend summary status is unsupported or ambiguous")
    return matches[0]


def _validate_source_evidence(
    summary_path: Path,
) -> tuple[
    str,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    list[dict[str, Any]],
]:
    summary = _load_json(summary_path)
    backend = _backend_from_summary(summary)
    specification = _BACKEND_SPECIFICATIONS[backend]
    artifacts = summary.get("artifacts", {})
    if (
        summary.get("scientific_claim_allowed") is not False
        or summary.get("cross_backend_absolute_comparison_allowed") is not False
        or int(summary.get("test_rows_read", -1)) != 0
        or not isinstance(artifacts, dict)
        or set(("posterior_batch", "robustness")) - set(artifacts)
    ):
        raise ValueError("Within-backend summary violates the validation-only boundary")

    resolved: dict[str, Path] = {}
    for label, identity in artifacts.items():
        if not isinstance(identity, dict):
            raise ValueError(f"Within-backend artifact is not a mapping: {label}")
        identity = artifacts[label]
        path = Path(str(identity.get("path", ""))).resolve()
        if not path.is_file() or file_sha256(path) != identity.get("sha256"):
            raise ValueError(f"Within-backend source artifact changed: {label}")
        resolved[label] = path

    batch_path = resolved["posterior_batch"]
    robustness_path = resolved["robustness"]
    batch = _load_json(batch_path)
    robustness = _load_json(robustness_path)
    manifest_path = Path(str(batch.get("manifest_path", ""))).resolve()
    if (
        batch.get("status") != specification["batch_status"]
        or batch.get("run_identity", {}).get("required_split") != "val"
        or not manifest_path.is_file()
        or batch.get("manifest_sha256") != file_sha256(manifest_path)
        or robustness.get("status") != "paired_pe_contamination_mask_robustness"
        or robustness.get("comparison_scope") != "strict_within_backend_paired"
        or robustness.get("within_backend_provenance_gate") is not True
        or robustness.get("cross_backend_matched_input_gate") is not False
        or robustness.get("dingo_amplfi_joint_gate") is not False
        or robustness.get("publication_provenance_required") is not True
        or Path(str(robustness.get("manifest_path", ""))).resolve()
        != manifest_path
        or robustness.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        raise ValueError("Within-backend batch or robustness evidence failed replay")
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if (
        not rows
        or len(rows) != int(batch.get("rows", -1))
        or any(str(row.get("backend", "")).upper() != backend for row in rows)
        or any(str(row.get("split", "")) != "val" for row in rows)
    ):
        raise ValueError("Within-backend posterior manifest is not validation-only")
    return (
        backend,
        summary,
        batch_path,
        batch,
        robustness_path,
        robustness,
        manifest_path,
        rows,
    )


def export_within_backend_pe_evidence_bundle(
    summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze portable, content-addressed validation evidence for one PE backend."""

    source_summary = Path(summary_path).resolve()
    (
        backend,
        summary,
        batch_path,
        _batch,
        robustness_path,
        _robustness,
        manifest_path,
        rows,
    ) = _validate_source_evidence(source_summary)
    root = Path(output_dir).resolve()
    receipt_path = root / "within_backend_pe_evidence_bundle.json"
    if receipt_path.is_file():
        existing = _load_json(receipt_path)
        if (
            existing.get("status") != "portable_within_backend_pe_evidence_bundle"
            or existing.get("bundle_schema_version") != 2
            or existing.get("source_summary_sha256") != file_sha256(source_summary)
        ):
            raise ValueError("Existing PE evidence bundle has another identity")
        for identity in existing.get("files", []):
            path = root / str(identity.get("relative_path", ""))
            if not path.is_file() or file_sha256(path) != identity.get("sha256"):
                raise ValueError("Existing PE evidence bundle file changed")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    payload_sources = {
        "summary": source_summary,
        "batch_report": batch_path,
        "robustness_report": robustness_path,
        "posterior_manifest": manifest_path,
    }
    for label, identity in summary["artifacts"].items():
        if label in {"posterior_batch", "robustness"}:
            continue
        payload_sources[f"summary_artifact:{label}"] = Path(
            str(identity["path"])
        ).resolve()
    files: list[dict[str, Any]] = []
    for role, source in payload_sources.items():
        relative = Path("reports") / f"{role}{source.suffix}"
        target = root / relative
        _atomic_copy(source, target)
        files.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "sha256": file_sha256(target),
                "bytes": target.stat().st_size,
            }
        )

    objects_by_hash: dict[str, dict[str, Any]] = {}
    row_bindings: list[dict[str, Any]] = []

    def add_object(source: Path, digest: str) -> str:
        prior = objects_by_hash.get(digest)
        if prior is not None:
            return str(prior["relative_path"])
        suffix = "".join(source.suffixes[-2:]) or ".bin"
        relative = Path("objects") / digest[:2] / f"{digest}{suffix}"
        target = root / relative
        _atomic_copy(source, target)
        objects_by_hash[digest] = {
            "role": "content_object",
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "bytes": target.stat().st_size,
        }
        return relative.as_posix()

    for row_index, row in enumerate(rows):
        for path_field, hash_field in _PORTABLE_PATH_HASH_FIELDS:
            path_value = row.get(path_field)
            hash_value = row.get(hash_field)
            if path_value in (None, "") and hash_value in (None, ""):
                continue
            if path_value in (None, "") or hash_value in (None, ""):
                raise ValueError(
                    f"PE row {row_index} has an incomplete {path_field}/{hash_field} identity"
                )
            source = Path(str(path_value)).resolve()
            if not source.is_file() or file_sha256(source) != str(hash_value):
                raise ValueError(
                    f"PE row {row_index} portable artifact failed hash replay: {path_field}"
                )
            relative = add_object(source, str(hash_value))
            row_bindings.append(
                {
                    "row_index": row_index,
                    "path_field": path_field,
                    "hash_field": hash_field,
                    "sha256": str(hash_value),
                    "relative_path": relative,
                }
            )
    metadata_bindings = []
    metadata_identity = summary["artifacts"].get("model_metadata")
    if metadata_identity is not None:
        metadata_path = Path(str(metadata_identity["path"])).resolve()
        metadata = _load_json(metadata_path)
        for identity in _metadata_identity_locations(metadata):
            source = Path(identity["path"]).resolve()
            if not source.is_file() or file_sha256(source) != identity["sha256"]:
                raise ValueError(
                    f"PE model metadata artifact failed hash replay: {identity['label']}"
                )
            metadata_bindings.append(
                {
                    key: identity[key]
                    for key in (
                        "location",
                        "label",
                        "path_field",
                        "hash_field",
                        "sha256",
                    )
                }
                | {
                    "relative_path": add_object(source, identity["sha256"]),
                }
            )
    files.extend(sorted(objects_by_hash.values(), key=lambda row: row["relative_path"]))
    result = {
        "status": "portable_within_backend_pe_evidence_bundle",
        "bundle_schema_version": 2,
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "this transport bundle preserves validation evidence but does not itself "
            "constitute a paired DINGO/AMPLFI result"
        ),
        "backend": backend,
        "required_split": "val",
        "test_rows_read": 0,
        "source_summary_path": str(source_summary),
        "source_summary_sha256": file_sha256(source_summary),
        "rows": len(rows),
        "row_bindings": row_bindings,
        "model_metadata_bindings": metadata_bindings,
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(int(identity["bytes"]) for identity in files),
        **execution_provenance(),
    }
    atomic_write_json(receipt_path, result)
    return result


def import_within_backend_pe_evidence_bundle(
    bundle_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Replay a transferred PE bundle and create explicit local path projections."""

    source_receipt_path = Path(bundle_receipt_path).resolve()
    source_receipt = _load_json(source_receipt_path)
    bundle_root = source_receipt_path.parent
    backend = str(source_receipt.get("backend", "")).upper()
    if (
        source_receipt.get("status")
        != "portable_within_backend_pe_evidence_bundle"
        or source_receipt.get("bundle_schema_version") != 2
        or source_receipt.get("passed") is not True
        or source_receipt.get("scientific_claim_allowed") is not False
        or source_receipt.get("required_split") != "val"
        or int(source_receipt.get("test_rows_read", -1)) != 0
        or backend not in _BACKEND_SPECIFICATIONS
    ):
        raise ValueError("Transferred PE bundle receipt failed its validation boundary")
    by_role: dict[str, Path] = {}
    for identity in source_receipt.get("files", []):
        relative = Path(str(identity.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("PE evidence bundle contains an unsafe relative path")
        path = bundle_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(identity.get("bytes", -1))
            or file_sha256(path) != identity.get("sha256")
        ):
            raise ValueError("Transferred PE evidence bundle file failed hash replay")
        role = str(identity.get("role", ""))
        if role != "content_object":
            if role in by_role:
                raise ValueError(f"PE evidence bundle repeats report role: {role}")
            by_role[role] = path
    fixed_roles = {
        "summary",
        "batch_report",
        "robustness_report",
        "posterior_manifest",
    }
    if not fixed_roles.issubset(by_role):
        raise ValueError("PE evidence bundle report inventory is incomplete")

    summary = _load_json(by_role["summary"])
    extra_labels = set(summary.get("artifacts", {})) - {
        "posterior_batch",
        "robustness",
    }
    expected_roles = fixed_roles | {
        f"summary_artifact:{label}" for label in extra_labels
    }
    if set(by_role) != expected_roles:
        raise ValueError("PE evidence bundle summary-artifact inventory changed")
    batch = _load_json(by_role["batch_report"])
    robustness = _load_json(by_role["robustness_report"])
    with by_role["posterior_manifest"].open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if (
        _backend_from_summary(summary) != backend
        or len(rows) != int(source_receipt.get("rows", -1))
    ):
        raise ValueError("Transferred PE bundle summary or row count changed")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    projected_rows = [dict(row) for row in rows]
    seen_bindings: set[tuple[int, str]] = set()
    for binding in source_receipt.get("row_bindings", []):
        row_index = int(binding.get("row_index", -1))
        path_field = str(binding.get("path_field", ""))
        hash_field = str(binding.get("hash_field", ""))
        if (
            not 0 <= row_index < len(projected_rows)
            or (path_field, hash_field) not in _PORTABLE_PATH_HASH_FIELDS
            or projected_rows[row_index].get(hash_field) != binding.get("sha256")
            or (row_index, path_field) in seen_bindings
        ):
            raise ValueError("Transferred PE row binding has another identity")
        relative = Path(str(binding.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Transferred PE row binding has an unsafe path")
        object_path = bundle_root / relative
        if (
            not object_path.is_file()
            or file_sha256(object_path) != binding.get("sha256")
        ):
            raise ValueError("Transferred PE row object failed binding replay")
        projected_rows[row_index][path_field] = str(object_path.resolve())
        seen_bindings.add((row_index, path_field))
    expected_bindings = {
        (row_index, path_field)
        for row_index, row in enumerate(rows)
        for path_field, hash_field in _PORTABLE_PATH_HASH_FIELDS
        if row.get(path_field) not in (None, "") or row.get(hash_field) not in (None, "")
    }
    if seen_bindings != expected_bindings:
        raise ValueError("Transferred PE row bindings are incomplete")

    projected_manifest = root / "posterior_manifest.projected.jsonl"
    atomic_write_text(
        projected_manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in projected_rows),
    )
    transport_identity = {
        "status": "path_projection_from_portable_pe_bundle",
        "bundle_receipt_path": str(source_receipt_path),
        "bundle_receipt_sha256": file_sha256(source_receipt_path),
    }
    batch["manifest_path"] = str(projected_manifest)
    batch["manifest_sha256"] = file_sha256(projected_manifest)
    batch["transport_projection"] = transport_identity
    projected_batch = root / "batch_report.projected.json"
    atomic_write_json(projected_batch, batch)

    robustness["manifest_path"] = str(projected_manifest)
    robustness["manifest_sha256"] = file_sha256(projected_manifest)
    robustness["transport_projection"] = transport_identity
    projected_robustness = root / "robustness_report.projected.json"
    atomic_write_json(projected_robustness, robustness)

    projected_summary = dict(summary)
    projected_summary["artifacts"] = dict(summary["artifacts"])
    projected_summary["artifacts"]["posterior_batch"] = {
        "path": str(projected_batch),
        "sha256": file_sha256(projected_batch),
    }
    projected_summary["artifacts"]["robustness"] = {
        "path": str(projected_robustness),
        "sha256": file_sha256(projected_robustness),
    }
    for label in extra_labels - {"model_metadata"}:
        path = by_role[f"summary_artifact:{label}"]
        projected_summary["artifacts"][label] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    if "model_metadata" in extra_labels:
        source_metadata_path = by_role["summary_artifact:model_metadata"]
        metadata = _load_json(source_metadata_path)
        projected_metadata = dict(metadata)
        projected_metadata["artifacts"] = {
            label: dict(identity)
            for label, identity in metadata.get("artifacts", {}).items()
        }
        seen_metadata_bindings = set()
        for binding in source_receipt.get("model_metadata_bindings", []):
            location = str(binding.get("location", ""))
            label = str(binding.get("label", ""))
            path_field = str(binding.get("path_field", ""))
            hash_field = str(binding.get("hash_field", ""))
            key = (location, label, path_field)
            if key in seen_metadata_bindings:
                raise ValueError("Transferred PE model metadata repeats a binding")
            relative = Path(str(binding.get("relative_path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Transferred PE model metadata has an unsafe path")
            object_path = bundle_root / relative
            if (
                not object_path.is_file()
                or file_sha256(object_path) != binding.get("sha256")
            ):
                raise ValueError("Transferred PE model artifact failed hash replay")
            if location == "top":
                if (
                    metadata.get(hash_field) != binding.get("sha256")
                    or metadata.get(path_field) in (None, "")
                ):
                    raise ValueError("Transferred PE top-level model identity changed")
                projected_metadata[path_field] = str(object_path.resolve())
            elif location == "artifact":
                identity = projected_metadata["artifacts"].get(label)
                if (
                    not isinstance(identity, dict)
                    or identity.get(hash_field) != binding.get("sha256")
                    or identity.get(path_field) in (None, "")
                ):
                    raise ValueError("Transferred PE model artifact identity changed")
                identity[path_field] = str(object_path.resolve())
            else:
                raise ValueError("Transferred PE model metadata location is invalid")
            seen_metadata_bindings.add(key)
        expected_metadata_bindings = {
            (identity["location"], identity["label"], identity["path_field"])
            for identity in _metadata_identity_locations(metadata)
        }
        if seen_metadata_bindings != expected_metadata_bindings:
            raise ValueError("Transferred PE model metadata bindings are incomplete")
        projected_metadata["transport_projection"] = transport_identity
        projected_metadata_path = root / "model_metadata.projected.json"
        atomic_write_json(projected_metadata_path, projected_metadata)
        projected_summary["artifacts"]["model_metadata"] = {
            "path": str(projected_metadata_path),
            "sha256": file_sha256(projected_metadata_path),
        }
    projected_summary["transport_projection"] = transport_identity
    projected_summary_path = root / "within_backend_summary.projected.json"
    atomic_write_json(projected_summary_path, projected_summary)

    result = {
        "status": "imported_portable_within_backend_pe_evidence",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "the imported backend evidence must still enter a matched-event "
            "DINGO/AMPLFI portfolio"
        ),
        "backend": backend,
        "required_split": "val",
        "test_rows_read": 0,
        "bundle_receipt_path": str(source_receipt_path),
        "bundle_receipt_sha256": file_sha256(source_receipt_path),
        "projected_summary_path": str(projected_summary_path),
        "projected_summary_sha256": file_sha256(projected_summary_path),
        "projected_batch_report_path": str(projected_batch),
        "projected_batch_report_sha256": file_sha256(projected_batch),
        "projected_robustness_report_path": str(projected_robustness),
        "projected_robustness_report_sha256": file_sha256(projected_robustness),
        "projected_manifest_path": str(projected_manifest),
        "projected_manifest_sha256": file_sha256(projected_manifest),
        "rows": len(projected_rows),
        **execution_provenance(),
    }
    receipt_path = root / "within_backend_pe_evidence_import.json"
    atomic_write_json(receipt_path, result)
    return result
