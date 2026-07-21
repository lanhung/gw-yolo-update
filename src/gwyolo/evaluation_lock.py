from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256
from .runtime import execution_provenance


_REQUIRED_FROZEN_ARTIFACTS = {
    "config",
    "model",
    "threshold_calibration",
    "ood_policy",
}


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
