from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, file_sha256, load_yaml
from .runtime import execution_provenance


_MISSING = object()
_OPERATORS = {
    "equals",
    "not_equals",
    "at_least",
    "at_most",
    "greater_than",
    "less_than",
    "in",
    "contains",
    "nonempty",
    "length_at_least",
    "all_empty",
}


def _field(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit():
            index = int(component)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite")
    return numeric


def _check(observed: Any, operation: str, expected: Any) -> bool:
    if operation == "equals":
        return observed == expected
    if operation == "not_equals":
        return observed != expected
    if operation in {"at_least", "at_most", "greater_than", "less_than"}:
        left = _finite_number(observed, "observed value")
        right = _finite_number(expected, "expected value")
        return {
            "at_least": left >= right,
            "at_most": left <= right,
            "greater_than": left > right,
            "less_than": left < right,
        }[operation]
    if operation == "in":
        return isinstance(expected, list) and observed in expected
    if operation == "contains":
        return isinstance(observed, (list, tuple, set, str, dict)) and expected in observed
    if operation == "nonempty":
        return observed is not None and hasattr(observed, "__len__") and len(observed) > 0
    if operation == "length_at_least":
        return hasattr(observed, "__len__") and len(observed) >= int(expected)
    if operation == "all_empty":
        if not isinstance(observed, dict):
            return False
        return all(not value for value in observed.values())
    raise AssertionError(operation)


def _validate_protocol(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = config.get("publication_evidence")
    if not isinstance(protocol, dict):
        raise ValueError("Configuration requires a publication_evidence mapping")
    if protocol.get("schema") != "publication_evidence_v1":
        raise ValueError("Unsupported publication evidence schema")
    if not isinstance(protocol.get("protocol"), str) or not protocol["protocol"]:
        raise ValueError("Publication evidence protocol needs a stable identity")
    if protocol.get("phase") not in {"validation_freeze", "locked_final"}:
        raise ValueError("Publication evidence phase must be validation_freeze or locked_final")
    requirements = protocol.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("Publication evidence protocol requires a non-empty requirements list")
    groups = protocol.get("groups")
    if not isinstance(groups, list) or not groups or any(not isinstance(item, str) for item in groups):
        raise ValueError("Publication evidence protocol requires named groups")
    if len(set(groups)) != len(groups):
        raise ValueError("Publication evidence groups must be unique")
    identifiers: set[str] = set()
    for requirement in requirements:
        if not isinstance(requirement, dict):
            raise ValueError("Publication evidence requirements must be mappings")
        identifier = requirement.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Every publication evidence requirement needs an id")
        if identifier in identifiers:
            raise ValueError(f"Duplicate publication evidence requirement: {identifier}")
        identifiers.add(identifier)
        if requirement.get("group") not in groups:
            raise ValueError(f"Unknown evidence group for {identifier}")
        checks = requirement.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"Publication evidence requirement {identifier} has no checks")
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get("field"), str):
                raise ValueError(f"Invalid evidence check for {identifier}")
            operation = check.get("op")
            if operation not in _OPERATORS:
                raise ValueError(f"Unsupported evidence operator for {identifier}: {operation}")
            if operation not in {"nonempty", "all_empty"} and "value" not in check:
                raise ValueError(f"Evidence check {identifier}.{check['field']} lacks a value")
        replay = requirement.get("replay_artifacts", [])
        if not isinstance(replay, list):
            raise ValueError(f"replay_artifacts must be a list for {identifier}")
        for item in replay:
            if not isinstance(item, dict) or set(item) != {"path_field", "sha256_field"}:
                raise ValueError(f"Invalid replay artifact declaration for {identifier}")
    return protocol, requirements


def _parse_bindings(bindings: list[str], identifiers: set[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for binding in bindings:
        identifier, separator, raw_path = binding.partition("=")
        if not separator or not identifier or not raw_path:
            raise ValueError("Evidence bindings must use REQUIREMENT_ID=/path/report.json")
        if identifier not in identifiers:
            raise ValueError(f"Evidence binding is not declared by the protocol: {identifier}")
        if identifier in parsed:
            raise ValueError(f"Evidence requirement is bound more than once: {identifier}")
        parsed[identifier] = Path(raw_path).resolve()
    return parsed


def _evaluate_requirement(
    requirement: dict[str, Any], evidence_path: Path | None
) -> dict[str, Any]:
    identifier = str(requirement["id"])
    required = bool(requirement.get("required", True))
    base = {
        "id": identifier,
        "group": str(requirement["group"]),
        "description": str(requirement.get("description", "")),
        "required": required,
    }
    if evidence_path is None:
        return {**base, "state": "pending" if required else "skipped", "passed": False}
    identity = {"path": str(evidence_path)}
    if not evidence_path.is_file():
        return {
            **base,
            "state": "failed",
            "passed": False,
            "evidence": identity,
            "failures": ["bound evidence file is absent"],
        }
    identity["sha256"] = file_sha256(evidence_path)
    try:
        report = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            **base,
            "state": "failed",
            "passed": False,
            "evidence": identity,
            "failures": [f"evidence is not readable JSON: {exc}"],
        }
    if not isinstance(report, dict):
        return {
            **base,
            "state": "failed",
            "passed": False,
            "evidence": identity,
            "failures": ["evidence JSON is not an object"],
        }
    results = []
    failures = []
    for declaration in requirement["checks"]:
        field = str(declaration["field"])
        operation = str(declaration["op"])
        expected = declaration.get("value")
        observed = _field(report, field)
        check_error = None
        try:
            passed = observed is not _MISSING and _check(observed, operation, expected)
        except (TypeError, ValueError, OverflowError) as exc:
            passed = False
            check_error = str(exc)
        row = {"field": field, "op": operation, "passed": passed}
        if operation not in {"nonempty", "all_empty"}:
            row["expected"] = expected
        row["observed"] = None if observed is _MISSING else observed
        if check_error is not None:
            row["error"] = check_error
        results.append(row)
        if not passed:
            failures.append(f"predicate failed: {field} {operation}")
    replay_results = []
    for declaration in requirement.get("replay_artifacts", []):
        path_field = str(declaration["path_field"])
        sha_field = str(declaration["sha256_field"])
        raw_path = _field(report, path_field)
        expected_hash = _field(report, sha_field)
        artifact = Path(str(raw_path)).resolve() if raw_path is not _MISSING else None
        replay_error = None
        try:
            observed_hash = (
                file_sha256(artifact)
                if artifact is not None and artifact.is_file()
                else None
            )
        except OSError as exc:
            observed_hash = None
            replay_error = str(exc)
        passed = (
            artifact is not None
            and isinstance(expected_hash, str)
            and len(expected_hash) == 64
            and observed_hash == expected_hash
        )
        replay_results.append(
            {
                "path_field": path_field,
                "sha256_field": sha_field,
                "path": str(artifact) if artifact is not None else None,
                "expected_sha256": None if expected_hash is _MISSING else expected_hash,
                "observed_sha256": observed_hash,
                "passed": passed,
                "error": replay_error,
            }
        )
        if not passed:
            failures.append(f"artifact replay failed: {path_field}")
    passed = not failures
    return {
        **base,
        "state": "passed" if passed else "failed",
        "passed": passed,
        "evidence": identity,
        "checks": results,
        "artifact_replay": replay_results,
        "failures": failures,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GW-YOLO publication evidence readiness",
        "",
        f"Protocol: `{report['protocol']}`  ",
        f"Phase: `{report['phase']}`  ",
        f"Ready: **{'yes' if report['publication_ready'] else 'no'}**  ",
        f"Required gates passed: **{report['summary']['required_passed']}/"
        f"{report['summary']['required_total']}**",
        "",
        "| Gate | Group | State | Evidence SHA-256 |",
        "|---|---|---:|---|",
    ]
    for row in report["requirements"]:
        digest = row.get("evidence", {}).get("sha256", "—")
        lines.append(f"| {row['id']} | {row['group']} | {row['state']} | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def run_publication_evidence_audit(
    config_path: str | Path,
    bindings: list[str],
    output_path: str | Path,
    markdown_path: str | Path | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Hash, replay and evaluate the complete predeclared publication evidence ledger."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Publication evidence audit outputs are immutable")
    if markdown_path is not None and Path(markdown_path).exists():
        raise FileExistsError("Publication evidence Markdown outputs are immutable")
    config = load_yaml(config_path)
    protocol, requirements = _validate_protocol(config)
    identifiers = {str(item["id"]) for item in requirements}
    parsed = _parse_bindings(bindings, identifiers)
    rows = [_evaluate_requirement(item, parsed.get(str(item["id"]))) for item in requirements]
    required_rows = [row for row in rows if row["required"]]
    groups = {}
    for group in protocol["groups"]:
        selected = [row for row in rows if row["group"] == group]
        required_selected = [row for row in selected if row["required"]]
        groups[group] = {
            "required_total": len(required_selected),
            "required_passed": sum(row["passed"] for row in required_selected),
            "pending": sum(row["state"] == "pending" for row in selected),
            "failed": sum(row["state"] == "failed" for row in selected),
        }
    required_passed = sum(row["passed"] for row in required_rows)
    ready = required_passed == len(required_rows)
    result = {
        "status": "publication_evidence_ready" if ready else "publication_evidence_incomplete",
        "schema": "publication_evidence_audit_v1",
        "publication_ready": ready,
        "locked_final_evidence_complete": ready and protocol.get("phase") == "locked_final",
        "scientific_claim_allowed": False,
        "scientific_claim_blocker": (
            "an evidence ledger cannot authorize a scientific claim; the immutable locked "
            "evaluation reports and their statistical interpretation remain authoritative"
        ),
        "protocol": str(protocol.get("protocol")),
        "phase": str(protocol.get("phase")),
        "config": {"path": str(Path(config_path).resolve()), "sha256": file_sha256(config_path)},
        "summary": {
            "required_total": len(required_rows),
            "required_passed": required_passed,
            "required_pending": sum(row["state"] == "pending" for row in required_rows),
            "required_failed": sum(row["state"] == "failed" for row in required_rows),
            "completion_percent": 100.0 * required_passed / len(required_rows),
        },
        "groups": groups,
        "requirements": rows,
        **execution_provenance(),
    }
    atomic_write_json(output, result)
    if markdown_path is not None:
        atomic_write_text(markdown_path, _markdown(result))
    if require_ready and not ready:
        raise RuntimeError(
            f"Publication evidence is incomplete: {required_passed}/{len(required_rows)} gates"
        )
    return result
