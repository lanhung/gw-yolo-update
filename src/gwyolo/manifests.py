from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, atomic_write_text, file_sha256
from .runtime import execution_provenance


def select_jsonl_split(
    manifest: str | Path,
    split: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize one explicit split while preserving complete input rows."""
    source = Path(manifest)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number} must contain a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError("Input manifest cannot be empty")
    missing = [index for index, row in enumerate(rows) if "split" not in row]
    if missing:
        raise ValueError(f"Manifest rows lack split at indices {missing[:10]}")
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"Manifest contains no rows for split {split!r}")
    identifier_field = next(
        (
            field
            for field in ("window_id", "injection_id", "sample_id", "id")
            if all(field in row for row in selected)
        ),
        None,
    )
    if identifier_field is None:
        raise ValueError("Selected rows have no common stable identifier field")
    identifiers = [str(row[identifier_field]) for row in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Selected split contains duplicate {identifier_field} values")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{split}_manifest.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
    )
    report = {
        "status": "explicit_split_manifest",
        "input_manifest_path": str(source),
        "input_manifest_sha256": file_sha256(source),
        "input_rows": len(rows),
        "input_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in rows).items())
        ),
        "selected_split": split,
        "selected_rows": len(selected),
        "identifier_field": identifier_field,
        "unique_identifiers": len(set(identifiers)),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **execution_provenance(),
    }
    atomic_write_json(output / "split_manifest_report.json", report)
    return report
