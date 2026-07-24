from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, file_sha256
from .runtime import execution_provenance


_PATH_HASH_FIELDS = (
    ("analysis_input_path", "analysis_input_sha256"),
    ("base_injection_manifest_path", "base_injection_manifest_sha256"),
    ("common_prior_path", "common_prior_sha256"),
    ("contamination_manifest_path", "contamination_manifest_sha256"),
    ("mask_artifact_path", "mask_artifact_sha256"),
    ("mask_model_path", "mask_model_sha256"),
    ("mask_policy_path", "mask_policy_sha256"),
    ("native_conditioning_path", "native_conditioning_sha256"),
    ("native_conditioning_config_path", "native_conditioning_config_sha256"),
)

_MANIFEST_ROLES = (
    "common_manifest",
    "dingo_manifest",
    "amplfi_manifest",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return value


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a non-empty JSONL object manifest in {source}")
    return rows


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


def _report_path(report: dict[str, Any], role: str) -> Path:
    manifest = Path(str(report.get("manifest_path", ""))).resolve()
    if role == "common_report":
        return manifest.parent / "common_pe_inputs_report.json"
    return manifest.parent / "native_conditioning_report.json"


def _validate_source(
    summary_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, list[dict[str, Any]]],
]:
    summary = _load_json(summary_path)
    reports = summary.get("reports")
    if (
        summary.get("status") != "paired_pe_native_inputs_smoke_complete"
        or summary.get("scientific_claim_allowed") is not False
        or int(summary.get("test_rows_read", -1)) != 0
        or not isinstance(reports, dict)
    ):
        raise ValueError("Paired PE input summary violates the validation-only boundary")
    selected = {
        "common_report": reports.get("common_sources"),
        "dingo_report": reports.get("dingo_native"),
        "amplfi_report": reports.get("amplfi_native"),
    }
    if any(not isinstance(report, dict) for report in selected.values()):
        raise ValueError("Paired PE input summary lacks a required report")
    typed_reports = {role: report for role, report in selected.items() if isinstance(report, dict)}
    common = typed_reports["common_report"]
    dingo = typed_reports["dingo_report"]
    amplfi = typed_reports["amplfi_report"]
    if (
        common.get("status") != "backend_neutral_paired_pe_inputs_materialized"
        or common.get("required_split") != "val"
        or dingo.get("status") != "native_pe_conditioning_materialized"
        or dingo.get("backend") != "DINGO"
        or dingo.get("run_identity", {}).get("required_split") != "val"
        or amplfi.get("status") != "native_pe_conditioning_materialized"
        or amplfi.get("backend") != "AMPLFI"
        or amplfi.get("run_identity", {}).get("required_split") != "val"
    ):
        raise ValueError("Paired PE input reports have another backend or split identity")

    paths: dict[str, Path] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    report_to_manifest = {
        "common_report": "common_manifest",
        "dingo_report": "dingo_manifest",
        "amplfi_report": "amplfi_manifest",
    }
    for report_role, manifest_role in report_to_manifest.items():
        report = typed_reports[report_role]
        manifest = Path(str(report.get("manifest_path", ""))).resolve()
        report_path = _report_path(report, report_role)
        if (
            not manifest.is_file()
            or file_sha256(manifest) != report.get("manifest_sha256")
            or not report_path.is_file()
            or _load_json(report_path) != report
        ):
            raise ValueError(f"Paired PE source artifact changed: {report_role}")
        paths[report_role] = report_path
        paths[manifest_role] = manifest
        rows[manifest_role] = _load_rows(manifest)

    common_rows = rows["common_manifest"]
    common_keys = {
        (str(row.get("injection_id")), str(row.get("condition"))): row for row in common_rows
    }
    if (
        len(common_keys) != len(common_rows)
        or any(row.get("split") != "val" for row in common_rows)
        or set(row.get("condition") for row in common_rows)
        - {"clean", "contaminated", "mask_conditioned"}
    ):
        raise ValueError("Common paired PE manifest is not unique validation data")
    matched_fields = (
        "analysis_input_sha256",
        "source_event_hash",
        "common_asd_sha256",
        "truth",
        "input_ifos",
        "input_sample_rate_hz",
        "input_duration_seconds",
        "input_post_trigger_seconds",
    )
    for backend_role in ("dingo_manifest", "amplfi_manifest"):
        backend_rows = rows[backend_role]
        backend_keys = {
            (str(row.get("injection_id")), str(row.get("condition"))): row for row in backend_rows
        }
        if (
            len(backend_keys) != len(backend_rows)
            or set(backend_keys) != set(common_keys)
            or any(row.get("split") != "val" for row in backend_rows)
        ):
            raise ValueError("Backend-native PE rows do not match the common event set")
        for key, row in backend_keys.items():
            source = common_keys[key]
            if any(row.get(field) != source.get(field) for field in matched_fields):
                raise ValueError("Backend-native PE rows do not preserve the common input identity")

    for manifest_role, manifest_rows in rows.items():
        for row_index, row in enumerate(manifest_rows):
            for path_field, hash_field in _PATH_HASH_FIELDS:
                path_value = row.get(path_field)
                hash_value = row.get(hash_field)
                if path_value in (None, "") and hash_value in (None, ""):
                    continue
                if path_value in (None, "") or hash_value in (None, ""):
                    raise ValueError(
                        f"{manifest_role} row {row_index} has an incomplete "
                        f"{path_field}/{hash_field} identity"
                    )
                path = Path(str(path_value)).resolve()
                if not path.is_file() or file_sha256(path) != str(hash_value):
                    raise ValueError(
                        f"{manifest_role} row {row_index} artifact failed hash replay: {path_field}"
                    )
    return summary, typed_reports, paths, rows


def export_paired_pe_input_bundle(
    summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze common and backend-native PE inputs into one content-addressed bundle."""

    source_summary = Path(summary_path).resolve()
    summary, _reports, paths, rows = _validate_source(source_summary)
    root = Path(output_dir).resolve()
    receipt_path = root / "paired_pe_input_bundle.json"
    if receipt_path.is_file():
        existing = _load_json(receipt_path)
        if existing.get("status") != "portable_paired_pe_input_bundle" or existing.get(
            "source_summary_sha256"
        ) != file_sha256(source_summary):
            raise ValueError("Existing paired PE input bundle has another identity")
        for identity in existing.get("files", []):
            path = root / str(identity.get("relative_path", ""))
            if (
                not path.is_file()
                or path.stat().st_size != int(identity.get("bytes", -1))
                or file_sha256(path) != identity.get("sha256")
            ):
                raise ValueError("Existing paired PE input bundle file changed")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    payload_sources = {"summary": source_summary, **paths}
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

    def add_object(source: Path, digest: str) -> Path:
        relative = Path("objects") / digest[:2] / digest
        target = root / relative
        if digest not in objects_by_hash:
            _atomic_copy(source, target)
            objects_by_hash[digest] = {
                "role": "content_object",
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "bytes": target.stat().st_size,
            }
        return relative

    for manifest_role, manifest_rows in rows.items():
        for row_index, row in enumerate(manifest_rows):
            for path_field, hash_field in _PATH_HASH_FIELDS:
                if row.get(path_field) in (None, ""):
                    continue
                digest = str(row[hash_field])
                relative = add_object(Path(str(row[path_field])).resolve(), digest)
                row_bindings.append(
                    {
                        "manifest_role": manifest_role,
                        "row_index": row_index,
                        "path_field": path_field,
                        "hash_field": hash_field,
                        "sha256": digest,
                        "relative_path": relative.as_posix(),
                    }
                )

    receipt_bindings = []
    for label, identity in sorted(summary.get("source_receipts", {}).items()):
        if not isinstance(identity, dict):
            raise ValueError("Paired PE source receipt identity is not a mapping")
        source = Path(str(identity.get("path", ""))).resolve()
        digest = str(identity.get("sha256", ""))
        if not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f"Paired PE source receipt changed: {label}")
        relative = add_object(source, digest)
        receipt_bindings.append(
            {
                "label": label,
                "sha256": digest,
                "relative_path": relative.as_posix(),
            }
        )

    files.extend(sorted(objects_by_hash.values(), key=lambda row: row["relative_path"]))
    result = {
        "status": "portable_paired_pe_input_bundle",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "this transport bundle preserves validation-only paired inputs; real "
            "within-backend posteriors and a matched-event portfolio remain required"
        ),
        "required_split": "val",
        "test_rows_read": 0,
        "source_summary_path": str(source_summary),
        "source_summary_sha256": file_sha256(source_summary),
        "paired_injections": int(summary["paired_injections"]),
        "rows": {role: len(manifest_rows) for role, manifest_rows in rows.items()},
        "row_bindings": row_bindings,
        "source_receipt_bindings": receipt_bindings,
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(int(identity["bytes"]) for identity in files),
        **execution_provenance(),
    }
    atomic_write_json(receipt_path, result)
    return result


def import_paired_pe_input_bundle(
    bundle_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify a transferred input bundle and project all paths onto this machine."""

    source_receipt_path = Path(bundle_receipt_path).resolve()
    receipt = _load_json(source_receipt_path)
    bundle_root = source_receipt_path.parent
    if (
        receipt.get("status") != "portable_paired_pe_input_bundle"
        or receipt.get("passed") is not True
        or receipt.get("scientific_claim_allowed") is not False
        or receipt.get("required_split") != "val"
        or int(receipt.get("test_rows_read", -1)) != 0
    ):
        raise ValueError("Transferred paired PE input bundle violates its boundary")

    by_role: dict[str, Path] = {}
    for identity in receipt.get("files", []):
        relative = Path(str(identity.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Paired PE input bundle contains an unsafe relative path")
        path = bundle_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(identity.get("bytes", -1))
            or file_sha256(path) != identity.get("sha256")
        ):
            raise ValueError("Transferred paired PE input file failed hash replay")
        role = str(identity.get("role", ""))
        if role != "content_object":
            if role in by_role:
                raise ValueError(f"Paired PE input bundle repeats role: {role}")
            by_role[role] = path
    expected_roles = {
        "summary",
        "common_report",
        "common_manifest",
        "dingo_report",
        "dingo_manifest",
        "amplfi_report",
        "amplfi_manifest",
    }
    if set(by_role) != expected_roles:
        raise ValueError("Paired PE input bundle report inventory is incomplete")

    summary = _load_json(by_role["summary"])
    reports = {
        "common_report": _load_json(by_role["common_report"]),
        "dingo_report": _load_json(by_role["dingo_report"]),
        "amplfi_report": _load_json(by_role["amplfi_report"]),
    }
    rows = {role: _load_rows(by_role[role]) for role in _MANIFEST_ROLES}
    if (
        summary.get("reports", {}).get("common_sources") != reports["common_report"]
        or summary.get("reports", {}).get("dingo_native") != reports["dingo_report"]
        or summary.get("reports", {}).get("amplfi_native") != reports["amplfi_report"]
        or {role: len(manifest_rows) for role, manifest_rows in rows.items()} != receipt.get("rows")
    ):
        raise ValueError("Transferred paired PE input reports changed")

    projected_rows = {
        role: [dict(row) for row in manifest_rows] for role, manifest_rows in rows.items()
    }
    seen_bindings: set[tuple[str, int, str]] = set()
    for binding in receipt.get("row_bindings", []):
        manifest_role = str(binding.get("manifest_role", ""))
        row_index = int(binding.get("row_index", -1))
        path_field = str(binding.get("path_field", ""))
        hash_field = str(binding.get("hash_field", ""))
        key = (manifest_role, row_index, path_field)
        if (
            manifest_role not in projected_rows
            or not 0 <= row_index < len(projected_rows[manifest_role])
            or (path_field, hash_field) not in _PATH_HASH_FIELDS
            or projected_rows[manifest_role][row_index].get(hash_field) != binding.get("sha256")
            or key in seen_bindings
        ):
            raise ValueError("Transferred paired PE row binding has another identity")
        relative = Path(str(binding.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Transferred paired PE row binding has an unsafe path")
        object_path = bundle_root / relative
        if not object_path.is_file() or file_sha256(object_path) != binding.get("sha256"):
            raise ValueError("Transferred paired PE row object failed hash replay")
        projected_rows[manifest_role][row_index][path_field] = str(object_path.resolve())
        seen_bindings.add(key)
    expected_bindings = {
        (manifest_role, row_index, path_field)
        for manifest_role, manifest_rows in rows.items()
        for row_index, row in enumerate(manifest_rows)
        for path_field, hash_field in _PATH_HASH_FIELDS
        if row.get(path_field) not in (None, "") or row.get(hash_field) not in (None, "")
    }
    if seen_bindings != expected_bindings:
        raise ValueError("Transferred paired PE row bindings are incomplete")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_targets = {
        "common_manifest": root / "common-sources/common_pe_inputs.jsonl",
        "dingo_manifest": root / "dingo-native/dingo_native_conditioning.jsonl",
        "amplfi_manifest": root / "amplfi-native/amplfi_native_conditioning.jsonl",
    }
    for role, target in manifest_targets.items():
        atomic_write_text(
            target,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in projected_rows[role]),
        )

    transport_identity = {
        "status": "path_projection_from_portable_paired_pe_input_bundle",
        "bundle_receipt_path": str(source_receipt_path),
        "bundle_receipt_sha256": file_sha256(source_receipt_path),
    }
    projected_reports = {role: dict(report) for role, report in reports.items()}
    report_manifest_roles = {
        "common_report": "common_manifest",
        "dingo_report": "dingo_manifest",
        "amplfi_report": "amplfi_manifest",
    }
    for report_role, manifest_role in report_manifest_roles.items():
        report = projected_reports[report_role]
        manifest = manifest_targets[manifest_role]
        report["manifest_path"] = str(manifest)
        report["manifest_sha256"] = file_sha256(manifest)
        report["transport_projection"] = transport_identity
    common_sha = file_sha256(manifest_targets["common_manifest"])
    for report_role in ("dingo_report", "amplfi_report"):
        run_identity = dict(projected_reports[report_role]["run_identity"])
        run_identity["source_manifest_sha256"] = common_sha
        projected_reports[report_role]["run_identity"] = run_identity

    report_targets = {
        "common_report": root / "common-sources/common_pe_inputs_report.json",
        "dingo_report": root / "dingo-native/native_conditioning_report.json",
        "amplfi_report": root / "amplfi-native/native_conditioning_report.json",
    }
    for role, target in report_targets.items():
        atomic_write_json(target, projected_reports[role])

    projected_summary = dict(summary)
    projected_summary["reports"] = dict(summary["reports"])
    projected_summary["reports"]["common_sources"] = projected_reports["common_report"]
    projected_summary["reports"]["dingo_native"] = projected_reports["dingo_report"]
    projected_summary["reports"]["amplfi_native"] = projected_reports["amplfi_report"]
    projected_receipts = {}
    seen_receipts = set()
    for binding in receipt.get("source_receipt_bindings", []):
        label = str(binding.get("label", ""))
        relative = Path(str(binding.get("relative_path", "")))
        if not label or label in seen_receipts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Transferred paired PE source receipt binding is invalid")
        object_path = bundle_root / relative
        if (
            not object_path.is_file()
            or file_sha256(object_path) != binding.get("sha256")
            or summary.get("source_receipts", {}).get(label, {}).get("sha256")
            != binding.get("sha256")
        ):
            raise ValueError("Transferred paired PE source receipt failed replay")
        projected_receipts[label] = {
            "path": str(object_path.resolve()),
            "sha256": binding["sha256"],
        }
        seen_receipts.add(label)
    if seen_receipts != set(summary.get("source_receipts", {})):
        raise ValueError("Transferred paired PE source receipt bindings are incomplete")
    projected_summary["source_receipts"] = projected_receipts
    projected_summary["transport_projection"] = transport_identity
    projected_summary_path = root / "paired_pe_smoke_summary.json"
    atomic_write_json(projected_summary_path, projected_summary)

    result = {
        "status": "imported_portable_paired_pe_inputs",
        "passed": True,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "the imported validation inputs still require real backend posteriors and "
            "a matched-event robustness portfolio"
        ),
        "required_split": "val",
        "test_rows_read": 0,
        "bundle_receipt_path": str(source_receipt_path),
        "bundle_receipt_sha256": file_sha256(source_receipt_path),
        "projected_summary_path": str(projected_summary_path),
        "projected_summary_sha256": file_sha256(projected_summary_path),
        "paired_injections": int(projected_summary["paired_injections"]),
        "rows": receipt["rows"],
        "manifest_sha256": {role: file_sha256(path) for role, path in manifest_targets.items()},
        **execution_provenance(),
    }
    atomic_write_json(root / "paired_pe_input_import.json", result)
    return result
