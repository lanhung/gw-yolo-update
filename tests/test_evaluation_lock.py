from __future__ import annotations

import json

import pytest

from gwyolo.evaluation_lock import freeze_evaluation_corpus


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_freeze_evaluation_corpus_records_physical_counts_and_is_idempotent(
    tmp_path,
) -> None:
    manifest = tmp_path / "test.jsonl"
    rows = [
        {
            "split": "test",
            "injection_id": "i0",
            "waveform_id": "w0",
            "gps_block": "g0",
            "source_family": "BBH",
        },
        {
            "split": "test",
            "injection_id": "i1",
            "waveform_id": "w1",
            "gps_block": "g0",
            "source_family": "BNS",
        },
    ]
    _write(manifest, rows)
    report = tmp_path / "freeze.json"
    access = tmp_path / "access.json"
    first = freeze_evaluation_corpus(
        manifest, report, access, "o4a-endpoint", minimum_rows=2
    )
    second = freeze_evaluation_corpus(
        manifest, report, access, "o4a-endpoint", minimum_rows=2
    )
    assert first == second
    assert first["evaluation_opened"] is False
    assert first["unique_group_counts"] == {
        "injection_id": 2,
        "waveform_id": 2,
        "gps_block": 1,
        "source_family": 2,
    }
    assert first["categorical_counts"]["source_family"] == {"BBH": 1, "BNS": 1}
    assert not access.exists()


def test_freeze_evaluation_corpus_rejects_wrong_split_and_duplicate_waveform(
    tmp_path,
) -> None:
    manifest = tmp_path / "bad.jsonl"
    base = {
        "injection_id": "i0",
        "waveform_id": "w0",
        "gps_block": "g0",
        "source_family": "BBH",
    }
    _write(manifest, [{**base, "split": "val"}])
    with pytest.raises(ValueError, match="outside the locked split"):
        freeze_evaluation_corpus(
            manifest, tmp_path / "freeze.json", tmp_path / "access.json", "bad"
        )
    _write(
        manifest,
        [
            {**base, "split": "test"},
            {**base, "split": "test", "injection_id": "i1"},
        ],
    )
    with pytest.raises(ValueError, match="waveform_id"):
        freeze_evaluation_corpus(
            manifest, tmp_path / "freeze.json", tmp_path / "access.json", "bad"
        )
