from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .io import atomic_write_json
from .runtime import execution_provenance


_OVERLAP_TRAINING_SYMBOLS = (
    "_read_rows",
    "glitch_family_sampling_weights",
    "overlap_training_split_audit",
    "PhysicalOverlapDataset",
    "_forward",
    "_availability_mask",
    "_masked_focal_dice",
    "_counts",
    "_metrics",
    "summarize_glitch_family_counts",
    "_overlap_epoch",
    "_clean_metrics",
    "_train_epoch",
    "configure_overlap_training_scope",
    "resolve_overlap_training_control",
    "overlap_checkpoint_selection_score",
    "_calibrate_overlap_thresholds",
    "run_physical_overlap_finetune",
)
_WHOLE_TRAINING_FILES = (
    "src/gwyolo/numeric.py",
    "src/gwyolo/physical_training.py",
    "src/gwyolo/io.py",
    "src/gwyolo/runtime.py",
)


def _git_output(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        raise ValueError(
            completed.stderr.decode("utf-8", errors="replace").strip()
            or "Git compatibility query failed"
        )
    return completed.stdout


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _overlap_surface_hash(source: bytes) -> tuple[str, list[str]]:
    tree = ast.parse(source.decode("utf-8"))
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(set(_OVERLAP_TRAINING_SYMBOLS) - set(nodes))
    if missing:
        raise ValueError(f"Overlap training surface omits symbols: {missing}")
    payload = {
        name: ast.dump(nodes[name], annotate_fields=True, include_attributes=False)
        for name in _OVERLAP_TRAINING_SYMBOLS
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(encoded), list(_OVERLAP_TRAINING_SYMBOLS)


def audit_overlap_training_code_compatibility(
    repository_path: str | Path,
    commits: list[str],
    config_relative_path: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Prove that model-training surfaces are identical across Git commits."""

    repository = Path(repository_path).resolve()
    if not (repository / ".git").exists():
        raise ValueError("Training compatibility repository is not a Git checkout")
    if len(set(commits)) < 2:
        raise ValueError("Training compatibility requires at least two commits")
    if config_relative_path.startswith("/") or ".." in Path(
        config_relative_path
    ).parts:
        raise ValueError("Training compatibility config path must be repository-relative")

    resolved = [
        _git_output(repository, "rev-parse", f"{commit}^{{commit}}")
        .decode()
        .strip()
        for commit in commits
    ]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Training compatibility commits are not unique")
    revisions: dict[str, Any] = {}
    for commit in resolved:
        overlap_source = _git_output(
            repository, "show", f"{commit}:src/gwyolo/overlap_training.py"
        )
        surface_hash, symbols = _overlap_surface_hash(overlap_source)
        whole_files = {
            relative: _sha256(
                _git_output(repository, "show", f"{commit}:{relative}")
            )
            for relative in (*_WHOLE_TRAINING_FILES, config_relative_path)
        }
        revisions[commit] = {
            "overlap_training_surface_sha256": surface_hash,
            "overlap_training_symbols": symbols,
            "whole_file_sha256": whole_files,
        }
    surface_hashes = {
        row["overlap_training_surface_sha256"] for row in revisions.values()
    }
    file_hash_sets = {
        relative: {
            row["whole_file_sha256"][relative] for row in revisions.values()
        }
        for relative in (*_WHOLE_TRAINING_FILES, config_relative_path)
    }
    checks = {
        "overlap_training_surface_identical": len(surface_hashes) == 1,
        **{
            f"whole_file_identical:{relative}": len(values) == 1
            for relative, values in file_hash_sets.items()
        },
    }
    passed = all(checks.values())
    result = {
        "status": "audited_overlap_training_code_compatibility",
        "passed": passed,
        "scientific_claim_allowed": False,
        "test_data_opened": False,
        "repository_path": str(repository),
        "audited_commits": resolved,
        "config_relative_path": config_relative_path,
        "checks": checks,
        "revisions": revisions,
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result
