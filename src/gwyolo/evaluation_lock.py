from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance


_REQUIRED_FROZEN_ARTIFACTS = {
    "config",
    "model",
    "threshold_calibration",
    "ood_policy",
}
_LOCKED_SUITE_OUTPUT_KEYS = {
    "raw_candidate_search",
    "mask_candidate_search",
    "paired_raw_mask_search",
    "locked_ood_transfer",
    "dingo_batch",
    "amplfi_batch",
    "joint_pe",
    "catalog_diagnostic",
    "suite_receipt",
}


def freeze_locked_evaluation_suite_plan(
    validation_evidence_report_path: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    code_commit: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze every final output and endpoint before one-time locked-corpus access."""

    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("Locked evaluation suite plans are immutable")
    evidence_path = Path(validation_evidence_report_path).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("status") != "publication_evidence_ready"
        or evidence.get("publication_ready") is not True
        or evidence.get("phase") != "validation_freeze"
        or evidence.get("scientific_claim_allowed") is not False
        or evidence.get("summary", {}).get("required_pending") != 0
        or evidence.get("summary", {}).get("required_failed") != 0
        or evidence.get("summary", {}).get("required_passed")
        != evidence.get("summary", {}).get("required_total")
    ):
        raise ValueError("Locked suite requires a complete validation-freeze evidence audit")
    config = load_yaml(config_path)
    settings = config.get("locked_evaluation_suite")
    if not isinstance(settings, dict) or settings.get("schema") != "locked_suite_v1":
        raise ValueError("Configuration requires locked_evaluation_suite schema v1")
    if str(settings.get("required_split")) != "test":
        raise ValueError("Locked evaluation suite must use the test split")
    if settings.get("observing_runs") != ["O4b"]:
        raise ValueError("Locked evaluation suite must remain restricted to O4b")
    if settings.get("catalog_release") != "GWTC-5.0":
        raise ValueError("Locked evaluation suite must predeclare GWTC-5.0")
    if not code_commit.strip():
        raise ValueError("Locked evaluation suite requires an exact code commit")
    outputs = settings.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != _LOCKED_SUITE_OUTPUT_KEYS:
        raise ValueError("Locked evaluation suite output inventory is incomplete")
    root = Path(output_root).resolve()
    resolved_outputs = {}
    for key, relative_value in sorted(outputs.items()):
        relative = Path(str(relative_value))
        if relative.is_absolute() or ".." in relative.parts or relative.name == "":
            raise ValueError(f"Locked suite output must be a safe relative path: {key}")
        resolved = (root / relative).resolve()
        if root not in resolved.parents or resolved.exists():
            raise ValueError(f"Locked suite output exists or escapes its root: {key}")
        resolved_outputs[key] = str(resolved)
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise ValueError("Locked evaluation suite output paths must be unique")
    endpoints = settings.get("endpoints")
    if not isinstance(endpoints, dict):
        raise ValueError("Locked evaluation suite requires predeclared endpoints")
    numeric_minima = {
        "target_far_per_year": 0.0,
        "minimum_test_live_time_years": 0.0,
        "minimum_test_injections": 0,
        "minimum_paired_pe_injections": 0,
        "minimum_locked_ood_rows": 0,
        "bootstrap_replicates": 9999,
    }
    for field, lower in numeric_minima.items():
        value = endpoints.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= lower:
            raise ValueError(f"Locked evaluation endpoint is invalid: {field}")
    bootstrap_seed = endpoints.get("bootstrap_seed")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("Locked evaluation endpoint is invalid: bootstrap_seed")
    if endpoints.get("primary_search_metric") != "paired_delta_recovered_vt_at_common_far":
        raise ValueError("Locked suite primary endpoint must be paired fixed-FAR recovered VT")
    if endpoints.get("threshold_policy") != "validation_frozen_no_test_retuning":
        raise ValueError("Locked suite must prohibit test threshold retuning")
    result = {
        "status": "frozen_locked_evaluation_suite_plan",
        "passed": True,
        "scientific_claim_allowed": False,
        "locked_corpus_opened": False,
        "test_rows_read": 0,
        "candidate_scores_inspected": False,
        "schema": settings["schema"],
        "corpus_label": str(settings.get("corpus_label")),
        "required_split": "test",
        "observing_runs": ["O4b"],
        "catalog_release": "GWTC-5.0",
        "code_commit": code_commit,
        "output_root": str(root),
        "outputs": resolved_outputs,
        "endpoints": endpoints,
        "validation_evidence": {
            "path": str(evidence_path),
            "sha256": file_sha256(evidence_path),
        },
        "config": {
            "path": str(Path(config_path).resolve()),
            "sha256": file_sha256(config_path),
            "canonical_hash": canonical_hash(config, 64),
        },
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = code_commit
    atomic_write_json(target, result)
    return result


def validate_locked_evaluation_suite_access(
    plan_path: str | Path,
    access_log_path: str | Path,
    output_key: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Replay the suite plan and one-time access receipt for one final output."""

    plan_file = Path(plan_path).resolve()
    access_file = Path(access_log_path).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    access = json.loads(access_file.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "frozen_locked_evaluation_suite_plan"
        or plan.get("passed") is not True
        or plan.get("locked_corpus_opened") is not False
        or plan.get("test_rows_read") != 0
        or access.get("status") != "locked_evaluation_corpus_opened_once"
        or access.get("evaluation_opened") is not True
        or access.get("test_metrics") is not None
        or access.get("code_commit") != plan.get("code_commit")
        or access.get("corpus_label") != plan.get("corpus_label")
    ):
        raise ValueError("Locked suite plan or one-time access receipt is invalid")
    frozen = access.get("frozen_artifacts", {}).get("locked_suite_plan", {})
    if (
        Path(str(frozen.get("path", ""))).resolve() != plan_file
        or frozen.get("sha256") != file_sha256(plan_file)
        or access.get("predeclared_evaluation_output")
        != plan.get("outputs", {}).get("suite_receipt")
    ):
        raise ValueError("One-time access receipt does not bind the frozen suite plan")
    expected = plan.get("outputs", {}).get(output_key)
    if expected is None or Path(str(expected)).resolve() != Path(output_path).resolve():
        raise ValueError("Locked evaluator output is not predeclared by the suite plan")
    return {
        "plan_path": str(plan_file),
        "plan_sha256": file_sha256(plan_file),
        "access_log_path": str(access_file),
        "access_log_sha256": file_sha256(access_file),
        "output_key": output_key,
        "output_path": str(Path(output_path).resolve()),
        "code_commit": plan["code_commit"],
        "corpus_label": plan["corpus_label"],
        "endpoints": plan["endpoints"],
    }


def finalize_locked_evaluation_suite_receipt(
    plan_path: str | Path,
    access_log_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Hash every predeclared locked output into one immutable completion receipt."""

    target = Path(output_path).resolve()
    if target.exists():
        raise FileExistsError("locked evaluation suite receipts are immutable")
    suite_binding = validate_locked_evaluation_suite_access(
        plan_path, access_log_path, "suite_receipt", target
    )
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    expected_statuses = {
        "raw_candidate_search": "locked_candidate_search_evaluation",
        "mask_candidate_search": "locked_candidate_search_evaluation",
        "paired_raw_mask_search": (
            "locked_paired_raw_mask_candidate_search_comparison"
        ),
        "locked_ood_transfer": "locked_detector_set_ood_transfer_evaluation",
        "dingo_batch": "locked_dingo_paired_pe_batch_complete",
        "amplfi_batch": "locked_amplfi_paired_pe_batch_complete",
        "joint_pe": "locked_joint_paired_pe_complete",
        "catalog_diagnostic": "locked_gwtc5_catalog_diagnostic",
    }
    outputs = {}
    endpoint_outcomes = {}
    for key, expected_status in expected_statuses.items():
        path = Path(plan["outputs"][key]).resolve()
        binding = validate_locked_evaluation_suite_access(
            plan_path, access_log_path, key, path
        )
        if not path.is_file():
            raise FileNotFoundError(f"predeclared locked suite output is missing: {key}")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"locked suite output is not readable JSON: {key}") from error
        if (
            not isinstance(report, dict)
            or report.get("status") != expected_status
            or report.get("locked_suite_access") != binding
        ):
            raise ValueError(f"locked suite output failed plan replay: {key}")
        outputs[key] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "status": expected_status,
        }
        endpoint_outcomes[key] = {
            field: report[field]
            for field in (
                "candidate_endpoint_gates_passed",
                "endpoint_complete",
                "promote_to_paper",
                "primary_endpoint_result",
            )
            if field in report
        }
    result = {
        "status": "completed_locked_evaluation_suite_receipt",
        "passed": True,
        "scientific_claim_allowed": False,
        "all_predeclared_outputs_present": len(outputs) == len(expected_statuses),
        "negative_and_null_results_retained": True,
        "protocol": (
            "hash every predeclared output without filtering on endpoint direction or "
            "statistical significance"
        ),
        "outputs": outputs,
        "endpoint_outcomes": endpoint_outcomes,
        "locked_suite_access": suite_binding,
        "code_commit": suite_binding["code_commit"],
        **execution_provenance(),
    }
    result["runtime_provenance"] = {
        "runtime_code_commit": result.pop("code_commit"),
        "exact_command": result.pop("exact_command"),
        "environment": result.pop("environment"),
    }
    result["code_commit"] = suite_binding["code_commit"]
    result["exact_command"] = result["runtime_provenance"]["exact_command"]
    result["environment"] = result["runtime_provenance"]["environment"]
    atomic_write_json(target, result)
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"evaluation manifest cannot be empty: {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"evaluation manifest must contain JSON objects: {path}")
    return rows


def _exclusive_atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete JSON file exactly once using an atomic hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"evaluation corpus was already opened: {path}"
            ) from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def open_evaluation_corpus_once(
    freeze_report_path: str | Path,
    code_commit: str,
    frozen_artifacts: dict[str, str | Path],
    comparison_manifests: tuple[str | Path, ...],
    evaluation_output_path: str | Path,
    evaluation_command: str,
    overlap_fields: tuple[str, ...] = (
        "injection_id",
        "waveform_id",
        "gps_block",
        "glitch_id",
    ),
) -> dict[str, Any]:
    """Irreversibly record the first authorized access to a locked corpus.

    This command does not calculate test metrics. It is the one-time gate immediately
    before score extraction and records every frozen analysis dependency by SHA-256.
    """
    freeze_path = Path(freeze_report_path).resolve()
    with freeze_path.open("r", encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze.get("status") != "locked_evaluation_corpus_unopened":
        raise ValueError("evaluation freeze report is not an unopened corpus contract")
    if not code_commit.strip() or not evaluation_command.strip():
        raise ValueError("code commit and exact evaluation command must be frozen")
    if not comparison_manifests or not overlap_fields:
        raise ValueError("comparison manifests and overlap fields are required")
    missing_artifacts = sorted(_REQUIRED_FROZEN_ARTIFACTS - set(frozen_artifacts))
    if missing_artifacts:
        raise ValueError(f"missing frozen evaluation artifacts: {missing_artifacts}")

    manifest = Path(freeze["manifest_path"]).resolve()
    if file_sha256(manifest) != freeze["manifest_sha256"]:
        raise ValueError("locked evaluation manifest changed after freezing")
    test_rows = _load_jsonl(manifest)
    expected_split = str(freeze["expected_split"])
    if any(str(row.get("split")) != expected_split for row in test_rows):
        raise ValueError("locked evaluation manifest split changed after freezing")

    artifact_hashes: dict[str, dict[str, str]] = {}
    for label, raw_path in sorted(frozen_artifacts.items()):
        if not label.strip():
            raise ValueError("frozen artifact labels cannot be empty")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"frozen artifact does not exist: {path}")
        artifact_hashes[label] = {"path": str(path), "sha256": file_sha256(path)}

    test_values = {
        field: {str(row[field]) for row in test_rows if row.get(field) is not None}
        for field in overlap_fields
    }
    manifest_audits = []
    for raw_path in comparison_manifests:
        path = Path(raw_path).resolve()
        rows = _load_jsonl(path)
        compared: dict[str, dict[str, int]] = {}
        overlaps: dict[str, list[str]] = {}
        for field in overlap_fields:
            other = {str(row[field]) for row in rows if row.get(field) is not None}
            if not test_values[field] or not other:
                continue
            shared = sorted(test_values[field] & other)
            compared[field] = {
                "locked_unique": len(test_values[field]),
                "comparison_unique": len(other),
                "overlap": len(shared),
            }
            if shared:
                overlaps[field] = shared[:20]
        if not compared:
            raise ValueError(
                f"comparison manifest has no auditable identity field: {path}"
            )
        if overlaps:
            raise ValueError(f"locked evaluation group overlap in {path}: {overlaps}")
        manifest_audits.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "fields": compared,
                "passed": True,
            }
        )

    access_log = Path(freeze["access_log_path"]).resolve()
    evaluation_output = Path(evaluation_output_path).resolve()
    if evaluation_output.exists():
        raise FileExistsError(
            f"predeclared evaluation output already exists: {evaluation_output}"
        )
    if evaluation_output == access_log:
        raise ValueError("evaluation output and access log must differ")
    report = {
        "status": "locked_evaluation_corpus_opened_once",
        "scientific_claim_allowed": False,
        "evaluation_opened": True,
        "test_metrics": None,
        "freeze_report_path": str(freeze_path),
        "freeze_report_sha256": file_sha256(freeze_path),
        "corpus_label": freeze["corpus_label"],
        "manifest_path": str(manifest),
        "manifest_sha256": freeze["manifest_sha256"],
        "rows": len(test_rows),
        "code_commit": code_commit,
        "frozen_artifacts": artifact_hashes,
        "comparison_manifest_audits": manifest_audits,
        "overlap_fields": list(overlap_fields),
        "predeclared_evaluation_output": str(evaluation_output),
        "predeclared_evaluation_command": evaluation_command,
        "protocol": (
            "irreversible one-time opening recorded immediately before locked score "
            "extraction; this receipt alone is not a scientific result"
        ),
        **execution_provenance(),
    }
    # Preserve the explicitly frozen code identity instead of allowing the runtime
    # environment variable to silently replace it.
    report["opening_command_provenance"] = {
        "runtime_code_commit": report.pop("code_commit"),
        "exact_command": report.pop("exact_command"),
        "environment": report.pop("environment"),
    }
    report["code_commit"] = code_commit
    _exclusive_atomic_json(access_log, report)
    return report


def freeze_evaluation_corpus(
    manifest_path: str | Path,
    output_path: str | Path,
    access_log_path: str | Path,
    corpus_label: str,
    expected_split: str = "test",
    minimum_rows: int = 1,
    group_fields: tuple[str, ...] = (
        "injection_id",
        "waveform_id",
        "gps_block",
        "source_family",
    ),
) -> dict[str, Any]:
    """Write an immutable, unopened evaluation-corpus identity contract."""
    manifest = Path(manifest_path).resolve()
    target = Path(output_path).resolve()
    access_log = Path(access_log_path).resolve()
    if not manifest.is_file() or not corpus_label.strip():
        raise ValueError("evaluation corpus freeze requires a manifest and label")
    if minimum_rows < 1 or not expected_split or not group_fields:
        raise ValueError("evaluation corpus freeze settings are invalid")
    if target == access_log:
        raise ValueError("evaluation freeze report and access log must differ")
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) < minimum_rows:
        raise ValueError("evaluation corpus is smaller than the declared minimum")
    if any(str(row.get("split")) != expected_split for row in rows):
        raise ValueError("evaluation corpus contains rows outside the locked split")
    missing = {
        field: [index for index, row in enumerate(rows) if field not in row][:10]
        for field in group_fields
    }
    missing = {field: indices for field, indices in missing.items() if indices}
    if missing:
        raise ValueError(f"evaluation corpus lacks frozen group fields: {missing}")
    for identity_field in ("injection_id", "waveform_id"):
        if identity_field in group_fields:
            values = [str(row[identity_field]) for row in rows]
            if len(set(values)) != len(values):
                raise ValueError(
                    f"evaluation corpus repeats physical identity {identity_field}"
                )
    group_counts = {
        field: len({str(row[field]) for row in rows}) for field in group_fields
    }
    value_counts = {
        field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
        for field in group_fields
        if field in {"source_family", "observing_run", "ifo", "detector_subset"}
    }
    identity = {
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "access_log_path": str(access_log),
        "corpus_label": corpus_label,
        "expected_split": expected_split,
        "minimum_rows": minimum_rows,
        "group_fields": list(group_fields),
    }
    if target.is_file():
        completed = json.loads(target.read_text(encoding="utf-8"))
        if completed.get("freeze_identity") != identity:
            raise ValueError("existing evaluation freeze report has another identity")
        if file_sha256(manifest) != completed["manifest_sha256"]:
            raise ValueError("locked evaluation manifest changed after freezing")
        return completed
    if access_log.exists():
        raise FileExistsError("evaluation access log exists before corpus freezing")
    report = {
        "status": "locked_evaluation_corpus_unopened",
        "scientific_claim_allowed": False,
        "evaluation_opened": False,
        "test_metrics": None,
        "freeze_identity": identity,
        "corpus_label": corpus_label,
        "expected_split": expected_split,
        "rows": len(rows),
        "manifest_path": str(manifest),
        "manifest_sha256": identity["manifest_sha256"],
        "access_log_path": str(access_log),
        "access_log_exists": False,
        "group_fields": list(group_fields),
        "unique_group_counts": group_counts,
        "categorical_counts": value_counts,
        "opening_requirements": [
            "frozen code commit, config, model, threshold calibration and OOD policy hashes",
            "one-time locked evaluator that atomically writes the predeclared access log",
            "zero group overlap with every training/selection/calibration manifest",
        ],
        **execution_provenance(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, report)
    return report
