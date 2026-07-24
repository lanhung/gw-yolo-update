from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256, load_yaml
from .runtime import execution_provenance


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON mapping: {path}")
    return value


def _require_hash(path_value: Any, expected: Any, label: str) -> Path:
    path = Path(str(path_value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is absent: {path}")
    if file_sha256(path) != str(expected):
        raise ValueError(f"{label} SHA256 mismatch")
    return path


def adjudicate_dingo_runtime_failure(
    failure_receipt_path: str | Path,
    policy_path: str | Path,
) -> dict[str, Any]:
    """Decide whether a pinned DINGO failure permits the predeclared native fallback."""

    failure_path = Path(failure_receipt_path).resolve()
    policy_file = Path(policy_path).resolve()
    failure = _load_json(failure_path)
    policy = load_yaml(policy_file)
    if policy.get("schema_version") != 1:
        raise ValueError("DINGO compatibility policy schema_version must be 1")
    if (
        failure.get("status") != "official_dingo_dual_model_load_failed"
        or failure.get("passed") is not False
        or failure.get("fallback_allowed") is not False
        or failure.get("scientific_claim_allowed") is not False
        or failure.get("backend") != "DINGO"
        or failure.get("test_rows_read") != 0
    ):
        raise ValueError("input is not a fail-closed official DINGO load receipt")
    primary_version = str(policy.get("primary_runtime_version", ""))
    fallback_version = str(policy.get("fallback_runtime_version", ""))
    if not primary_version or not fallback_version or primary_version == fallback_version:
        raise ValueError("DINGO compatibility policy versions are invalid")
    if failure.get("backend_version") != primary_version:
        raise ValueError("failure receipt does not use the policy primary runtime")

    acquisition_path = _require_hash(
        failure.get("model_acquisition_report_path"),
        failure.get("model_acquisition_report_sha256"),
        "DINGO acquisition report",
    )
    acquisition = _load_json(acquisition_path)
    if acquisition.get("status") != "verified" or acquisition.get("download_enabled") is not True:
        raise ValueError("DINGO acquisition report is not a verified download")
    config_path = _require_hash(
        failure.get("model_source_config_path"),
        failure.get("model_source_config_sha256"),
        "DINGO source configuration",
    )
    if acquisition.get("config_sha256") != file_sha256(config_path):
        raise ValueError("failure receipt and acquisition report bind different source configs")
    attempt_log_path = _require_hash(
        failure.get("attempt_log_path"),
        failure.get("attempt_log_sha256"),
        "DINGO model-load attempt log",
    )
    posterior_path = _require_hash(
        failure.get("posterior_model_path"),
        failure.get("posterior_model_sha256"),
        "DINGO posterior model",
    )
    initialization_path = _require_hash(
        failure.get("initialization_model_path"),
        failure.get("initialization_model_sha256"),
        "DINGO time-initialization model",
    )
    acquired_by_role = {
        str(row.get("role")): row for row in acquisition.get("files", [])
    }
    for role, path, sha in (
        ("posterior_model", posterior_path, failure.get("posterior_model_sha256")),
        (
            "time_initialization_model",
            initialization_path,
            failure.get("initialization_model_sha256"),
        ),
    ):
        row = acquired_by_role.get(role, {})
        if (
            row.get("valid") is not True
            or Path(str(row.get("path", ""))).resolve() != path
            or row.get("sha256") != sha
        ):
            raise ValueError(f"failure receipt does not replay acquired model role: {role}")

    log_text = attempt_log_path.read_text(encoding="utf-8", errors="replace")
    allow_patterns = policy.get("compatibility_allow_patterns")
    deny_patterns = policy.get("infrastructure_deny_patterns")
    if not isinstance(allow_patterns, list) or not allow_patterns:
        raise ValueError("DINGO compatibility policy requires allow patterns")
    if not isinstance(deny_patterns, list) or not deny_patterns:
        raise ValueError("DINGO compatibility policy requires deny patterns")
    matched_allow = [
        str(pattern)
        for pattern in allow_patterns
        if re.search(str(pattern), log_text, flags=re.IGNORECASE | re.MULTILINE)
    ]
    matched_deny = [
        str(pattern)
        for pattern in deny_patterns
        if re.search(str(pattern), log_text, flags=re.IGNORECASE | re.MULTILINE)
    ]
    authorized = bool(matched_allow) and not matched_deny
    return {
        "status": (
            "dingo_native_runtime_fallback_authorized"
            if authorized
            else "dingo_native_runtime_fallback_rejected"
        ),
        "passed": authorized,
        "fallback_allowed": authorized,
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "runtime compatibility authorization is not posterior or calibration evidence"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "backend": "DINGO",
        "primary_runtime_version": primary_version,
        "authorized_fallback_runtime_version": fallback_version if authorized else None,
        "model_substitution_allowed": False,
        "failure_receipt_path": str(failure_path),
        "failure_receipt_sha256": file_sha256(failure_path),
        "policy_path": str(policy_file),
        "policy_sha256": file_sha256(policy_file),
        "model_acquisition_report_path": str(acquisition_path),
        "model_acquisition_report_sha256": file_sha256(acquisition_path),
        "model_source_config_path": str(config_path),
        "model_source_config_sha256": file_sha256(config_path),
        "attempt_log_path": str(attempt_log_path),
        "attempt_log_sha256": file_sha256(attempt_log_path),
        "matched_compatibility_patterns": matched_allow,
        "matched_infrastructure_patterns": matched_deny,
        "posterior_model_path": str(posterior_path),
        "posterior_model_sha256": file_sha256(posterior_path),
        "initialization_model_path": str(initialization_path),
        "initialization_model_sha256": file_sha256(initialization_path),
        **execution_provenance(),
    }


def run_dingo_runtime_failure_adjudication(
    failure_receipt_path: str | Path,
    policy_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = adjudicate_dingo_runtime_failure(failure_receipt_path, policy_path)
    atomic_write_json(output_path, report)
    if not report["passed"]:
        raise RuntimeError("DINGO failure is not eligible for the native-runtime fallback")
    return report
