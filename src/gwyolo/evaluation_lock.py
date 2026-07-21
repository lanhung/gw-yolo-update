from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, file_sha256
from .runtime import execution_provenance


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
