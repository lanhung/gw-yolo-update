from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json, canonical_hash, file_sha256
from .runtime import execution_provenance


_EXCLUDED_ORCHESTRATION_MODULES = {
    "src/gwyolo/cli.py",
    "src/gwyolo/code_compatibility.py",
    "src/gwyolo/mask_timing.py",
    "src/gwyolo/streaming.py",
}
_NORMALIZED_ORCHESTRATION_FUNCTIONS = {
    "src/gwyolo/candidates.py": {"run_apply_candidate_timing_calibration"},
}
_CALIBRATION_TIMING_TRANSFER_FUNCTIONS = {
    "src/gwyolo/candidates.py": {
        "_active_runs",
        "_parabolic_offset",
        "extract_temporal_clusters",
        "_clusters_from_scored_row",
        "build_injection_candidate_rankings",
        "_cluster_network_rows",
        "run_candidate_block_permutations",
    },
    "src/gwyolo/trigger.py": {"network_ranking"},
}


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _implementation_digest(path: Path, relative: str) -> str:
    excluded_functions = _NORMALIZED_ORCHESTRATION_FUNCTIONS.get(relative)
    if not excluded_functions:
        return file_sha256(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in excluded_functions
        )
    ]
    observed = {
        node.name
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in excluded_functions
    }
    if observed != excluded_functions:
        raise ValueError(f"candidate compatibility normalization target mismatch in {relative}")
    return canonical_hash(ast.dump(tree, annotate_fields=True, include_attributes=False), 64)


def _implementation_inventory(root: Path) -> dict[str, str]:
    source = root / "src" / "gwyolo"
    if not source.is_dir():
        raise ValueError(f"candidate scoring code root is invalid: {root}")
    inventory = {}
    for path in sorted(source.glob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative not in _EXCLUDED_ORCHESTRATION_MODULES:
            inventory[relative] = _implementation_digest(path, relative)
    if not inventory:
        raise ValueError("candidate scoring implementation inventory is empty")
    return inventory


def _selected_function_inventory(
    root: Path, targets: dict[str, set[str]]
) -> dict[str, str]:
    inventory = {}
    for relative, names in sorted(targets.items()):
        path = root / relative
        if not path.is_file():
            raise ValueError(f"timing-transfer source is absent: {path}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing = sorted(names - set(functions))
        if missing:
            raise ValueError(f"timing-transfer functions are absent in {relative}: {missing}")
        for name in sorted(names):
            inventory[f"{relative}:{name}"] = canonical_hash(
                ast.dump(functions[name], annotate_fields=True, include_attributes=False),
                64,
            )
    return inventory


def audit_calibration_timing_transfer_compatibility(
    reference_code_dir: str | Path,
    candidate_code_dir: str | Path,
    reference_commit: str,
    candidate_commit: str,
    output: str | Path,
) -> dict[str, Any]:
    """Prove calibration stresses preserve frozen candidate timing/ranking semantics."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("Calibration timing-transfer reports are immutable")
    reference_root = Path(reference_code_dir).resolve()
    candidate_root = Path(candidate_code_dir).resolve()
    if (
        _git_commit(reference_root) != reference_commit
        or _git_commit(candidate_root) != candidate_commit
    ):
        raise ValueError("Calibration timing-transfer checkout commit mismatch")
    reference = _selected_function_inventory(
        reference_root, _CALIBRATION_TIMING_TRANSFER_FUNCTIONS
    )
    candidate = _selected_function_inventory(
        candidate_root, _CALIBRATION_TIMING_TRANSFER_FUNCTIONS
    )
    differences = [
        {
            "function": name,
            "reference_sha256": reference.get(name),
            "candidate_sha256": candidate.get(name),
        }
        for name in sorted(set(reference) | set(candidate))
        if reference.get(name) != candidate.get(name)
    ]
    result = {
        "status": "calibration_timing_transfer_implementation_compatibility",
        "passed": not differences,
        "scientific_claim_allowed": False,
        "scope": "candidate timing extraction, network ranking, and block clustering only",
        "allowed_scoring_change": (
            "frozen frequency-dependent calibration response before fresh whitening and transform"
        ),
        "reference_code_dir": str(reference_root),
        "reference_commit": reference_commit,
        "candidate_code_dir": str(candidate_root),
        "candidate_commit": candidate_commit,
        "function_targets": {
            path: sorted(names)
            for path, names in sorted(_CALIBRATION_TIMING_TRANSFER_FUNCTIONS.items())
        },
        "compared_functions": len(reference),
        "reference_inventory_hash": canonical_hash(reference, 64),
        "candidate_inventory_hash": canonical_hash(candidate, 64),
        "differences": differences,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    if differences:
        raise ValueError("Calibration stress changes frozen timing/ranking semantics")
    return result


def validate_calibration_timing_transfer_compatibility(
    report_path: str | Path,
    reference_commit: str,
    candidate_commit: str,
) -> dict[str, Any]:
    """Replay the narrow calibration timing-transfer implementation proof."""

    import json

    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status")
        != "calibration_timing_transfer_implementation_compatibility"
        or report.get("passed") is not True
        or report.get("differences") != []
        or report.get("reference_commit") != reference_commit
        or report.get("candidate_commit") != candidate_commit
        or report.get("reference_inventory_hash")
        != report.get("candidate_inventory_hash")
        or int(report.get("compared_functions", 0))
        != sum(len(names) for names in _CALIBRATION_TIMING_TRANSFER_FUNCTIONS.values())
    ):
        raise ValueError("Calibration timing-transfer compatibility report failed replay")
    reference_root = Path(str(report.get("reference_code_dir", ""))).resolve()
    candidate_root = Path(str(report.get("candidate_code_dir", ""))).resolve()
    reference = _selected_function_inventory(
        reference_root, _CALIBRATION_TIMING_TRANSFER_FUNCTIONS
    )
    candidate = _selected_function_inventory(
        candidate_root, _CALIBRATION_TIMING_TRANSFER_FUNCTIONS
    )
    if (
        _git_commit(reference_root) != reference_commit
        or _git_commit(candidate_root) != candidate_commit
        or canonical_hash(reference, 64) != report["reference_inventory_hash"]
        or canonical_hash(candidate, 64) != report["candidate_inventory_hash"]
    ):
        raise ValueError("Calibration timing-transfer source inventory changed")
    return report


def audit_candidate_scoring_implementation_compatibility(
    reference_code_dir: str | Path,
    candidate_code_dir: str | Path,
    reference_commit: str,
    candidate_commit: str,
    output: str | Path,
) -> dict[str, Any]:
    """Prove that a newer orchestrator preserves the calibrated scoring implementation."""

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("candidate scoring compatibility reports are immutable")
    reference_root = Path(reference_code_dir).resolve()
    candidate_root = Path(candidate_code_dir).resolve()
    observed_reference = _git_commit(reference_root)
    observed_candidate = _git_commit(candidate_root)
    if observed_reference != reference_commit or observed_candidate != candidate_commit:
        raise ValueError("candidate scoring compatibility checkout commit mismatch")
    reference = _implementation_inventory(reference_root)
    candidate = _implementation_inventory(candidate_root)
    all_paths = sorted(set(reference) | set(candidate))
    differences = [
        {
            "path": path,
            "reference_sha256": reference.get(path),
            "candidate_sha256": candidate.get(path),
        }
        for path in all_paths
        if reference.get(path) != candidate.get(path)
    ]
    result = {
        "status": "candidate_scoring_implementation_compatibility",
        "passed": not differences,
        "scientific_claim_allowed": False,
        "reference_code_dir": str(reference_root),
        "reference_commit": reference_commit,
        "candidate_code_dir": str(candidate_root),
        "candidate_commit": candidate_commit,
        "excluded_orchestration_modules": sorted(_EXCLUDED_ORCHESTRATION_MODULES),
        "normalized_orchestration_functions": {
            path: sorted(functions)
            for path, functions in sorted(_NORMALIZED_ORCHESTRATION_FUNCTIONS.items())
        },
        "compared_files": len(all_paths),
        "reference_inventory_hash": canonical_hash(reference, 64),
        "candidate_inventory_hash": canonical_hash(candidate, 64),
        "differences": differences,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    if differences:
        raise ValueError(
            "candidate scoring implementation differs from the timing-calibrated checkout"
        )
    return result


def validate_candidate_scoring_compatibility(
    report_path: str | Path,
    reference_commit: str,
    candidate_commit: str,
) -> dict[str, Any]:
    """Replay a compatibility report before applying cross-commit timing calibration."""

    import json

    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "candidate_scoring_implementation_compatibility"
        or report.get("passed") is not True
        or report.get("differences") != []
        or report.get("reference_commit") != reference_commit
        or report.get("candidate_commit") != candidate_commit
        or report.get("reference_inventory_hash") != report.get("candidate_inventory_hash")
        or int(report.get("compared_files", 0)) <= 0
    ):
        raise ValueError("candidate scoring compatibility report failed replay")
    reference = _implementation_inventory(Path(str(report.get("reference_code_dir", ""))).resolve())
    candidate = _implementation_inventory(Path(str(report.get("candidate_code_dir", ""))).resolve())
    if (
        _git_commit(Path(report["reference_code_dir"])) != reference_commit
        or _git_commit(Path(report["candidate_code_dir"])) != candidate_commit
        or canonical_hash(reference, 64) != report["reference_inventory_hash"]
        or canonical_hash(candidate, 64) != report["candidate_inventory_hash"]
    ):
        raise ValueError("candidate scoring compatibility source inventory changed")
    return report
