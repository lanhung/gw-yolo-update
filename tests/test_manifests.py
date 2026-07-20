from __future__ import annotations

import json

import pytest

from gwyolo.manifests import select_jsonl_split


def test_select_jsonl_split_preserves_rows_and_records_hashes(tmp_path):
    source = tmp_path / "all.jsonl"
    rows = [
        {"window_id": "train-0", "split": "train", "value": 1},
        {"window_id": "val-0", "split": "val", "value": 2},
        {"window_id": "val-1", "split": "val", "value": 3},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = select_jsonl_split(source, "val", tmp_path / "selected")
    selected = [
        json.loads(line)
        for line in open(report["manifest_path"], encoding="utf-8")
        if line.strip()
    ]
    assert selected == rows[1:]
    assert report["input_split_counts"] == {"train": 1, "val": 2}
    assert report["selected_rows"] == 2
    assert report["unique_identifiers"] == 2


def test_select_jsonl_split_rejects_duplicate_physical_ids(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    source.write_text(
        json.dumps({"injection_id": "same", "split": "val"})
        + "\n"
        + json.dumps({"injection_id": "same", "split": "val"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate injection_id"):
        select_jsonl_split(source, "val", tmp_path / "selected")
