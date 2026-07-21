from __future__ import annotations

import json

import pytest

from gwyolo.evaluation_lock import (
    freeze_evaluation_corpus,
    open_evaluation_corpus_once,
)


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


def test_open_evaluation_corpus_once_hashes_dependencies_and_rejects_reopening(
    tmp_path,
) -> None:
    test_manifest = tmp_path / "test.jsonl"
    _write(
        test_manifest,
        [
            {
                "split": "test",
                "injection_id": "test-i0",
                "waveform_id": "test-w0",
                "gps_block": "test-g0",
                "source_family": "BBH",
            }
        ],
    )
    access = tmp_path / "access.json"
    freeze = tmp_path / "freeze.json"
    freeze_evaluation_corpus(test_manifest, freeze, access, "o4a-endpoint")
    train_manifest = tmp_path / "train.jsonl"
    _write(
        train_manifest,
        [
            {
                "split": "train",
                "injection_id": "train-i0",
                "waveform_id": "train-w0",
                "gps_block": "train-g0",
            }
        ],
    )
    artifacts = {}
    for label in ("config", "model", "threshold_calibration", "ood_policy"):
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        artifacts[label] = path
    report = open_evaluation_corpus_once(
        freeze,
        "abc123",
        artifacts,
        (train_manifest,),
        tmp_path / "metrics.json",
        "python -m gwyolo.cli candidate-search-evaluate-frozen ...",
    )
    assert report["evaluation_opened"] is True
    assert report["code_commit"] == "abc123"
    assert report["comparison_manifest_audits"][0]["passed"] is True
    assert json.loads(access.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="already opened"):
        open_evaluation_corpus_once(
            freeze,
            "abc123",
            artifacts,
            (train_manifest,),
            tmp_path / "metrics.json",
            "same frozen command",
        )


def test_open_evaluation_corpus_once_rejects_group_overlap_before_access(tmp_path) -> None:
    test_manifest = tmp_path / "test.jsonl"
    row = {
        "split": "test",
        "injection_id": "i0",
        "waveform_id": "w0",
        "gps_block": "g0",
        "source_family": "BBH",
    }
    _write(test_manifest, [row])
    access = tmp_path / "access.json"
    freeze = tmp_path / "freeze.json"
    freeze_evaluation_corpus(test_manifest, freeze, access, "o4a-endpoint")
    train_manifest = tmp_path / "train.jsonl"
    _write(train_manifest, [{**row, "split": "train", "injection_id": "i1"}])
    artifacts = {}
    for label in ("config", "model", "threshold_calibration", "ood_policy"):
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        artifacts[label] = path
    with pytest.raises(ValueError, match="group overlap"):
        open_evaluation_corpus_once(
            freeze,
            "abc123",
            artifacts,
            (train_manifest,),
            tmp_path / "metrics.json",
            "frozen command",
        )
    assert not access.exists()
