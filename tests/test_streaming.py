from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import atomic_write_json, atomic_write_text, file_sha256
from gwyolo.streaming import (
    evict_candidate_probability_artifacts,
    evict_scored_background_batch_sources,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _score_and_candidate(
    root: Path, split: str, rows: list[dict], suffix: str
) -> tuple[Path, Path]:
    triggers = root / f"triggers-{suffix}.jsonl"
    _jsonl(triggers, rows)
    score = root / f"score-{suffix}.json"
    atomic_write_json(
        score,
        {
            "probabilities_saved": True,
            "failed_windows": 0,
            "required_split": split,
            "triggers_path": str(triggers),
            "triggers_sha256": file_sha256(triggers),
        },
    )
    candidates = root / f"candidates-{suffix}.jsonl"
    _jsonl(candidates, [])
    candidate = root / f"candidate-{suffix}.json"
    atomic_write_json(
        candidate,
        {
            "status": "subwindow_cluster_integration_only",
            "input_windows": len(rows),
            "manifest_path": str(candidates),
            "manifest_sha256": file_sha256(candidates),
            "source_scoring_provenance": {
                "available": True,
                "score_report_sha256": file_sha256(score),
            },
        },
    )
    return score, candidate


def test_probability_eviction_is_hash_bound_and_recoverable(tmp_path: Path) -> None:
    cache = tmp_path / "probability-cache"
    cache.mkdir()
    probability = cache / "window.npz"
    probability.write_bytes(b"model probabilities and strain")
    triggers = tmp_path / "triggers.jsonl"
    _jsonl(
        triggers,
        [
            {
                "window_id": "window-1",
                "probability_path": str(probability),
                "probability_sha256": file_sha256(probability),
            }
        ],
    )
    score = tmp_path / "trigger_score_report.json"
    atomic_write_json(
        score,
        {
            "probabilities_saved": True,
            "failed_windows": 0,
            "triggers_path": str(triggers),
            "triggers_sha256": file_sha256(triggers),
        },
    )
    candidates = tmp_path / "candidate_extraction_report.json"
    atomic_write_json(
        candidates,
        {
            "status": "subwindow_cluster_integration_only",
            "input_windows": 1,
            "source_scoring_provenance": {
                "available": True,
                "score_report_sha256": file_sha256(score),
            },
        },
    )

    output = tmp_path / "reports" / "probability-eviction.json"
    result = evict_candidate_probability_artifacts(candidates, score, cache, output)

    assert result["removed_files"] == 1
    assert result["removed_bytes"] == len(b"model probabilities and strain")
    assert result["recoverable"] is True
    assert not probability.exists()
    assert output.is_file()
    assert output.with_suffix(".json.intent.json").is_file()
    with pytest.raises(FileExistsError):
        evict_candidate_probability_artifacts(candidates, score, cache, output)


def test_background_source_eviction_allows_empty_required_split(tmp_path: Path) -> None:
    cache = tmp_path / "gwosc-cache"
    cache.mkdir()
    h1 = cache / "H1.hdf5"
    l1 = cache / "L1.hdf5"
    h1.write_bytes(b"H1 public strain")
    l1.write_bytes(b"L1 public strain")
    batch = tmp_path / "batch.json"
    atomic_write_json(
        batch,
        {
            "status": "verified_development_strain_batch",
            "passed": True,
            "files": [
                {"path": str(h1), "sha256": file_sha256(h1)},
                {"path": str(l1), "sha256": file_sha256(l1)},
            ],
        },
    )
    source_files = {
        "H1": {"path": str(h1), "sha256": file_sha256(h1)},
        "L1": {"path": str(l1), "sha256": file_sha256(l1)},
    }
    manifest = tmp_path / "background.jsonl"
    _jsonl(
        manifest,
        [
            {
                "window_id": "val-window",
                "split": "val",
                "source_files": source_files,
            },
            {
                "window_id": "train-window",
                "split": "train",
                "source_files": source_files,
            },
        ],
    )
    plan = tmp_path / "background-report.json"
    atomic_write_json(
        plan,
        {
            "status": "verified_multi_segment_development_background",
            "passed": True,
            "split_strategy": "hash_threshold_v1",
            "source_batch_report_sha256s": [file_sha256(batch)],
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        },
    )
    probability = cache / "val-window.npz"
    probability.write_bytes(b"temporary probability")
    score_rows = [
        {
            "window_id": "val-window",
            "split": "val",
            "probability_path": str(probability),
            "probability_sha256": file_sha256(probability),
        }
    ]
    score, candidate = _score_and_candidate(tmp_path, "val", score_rows, "val")

    output = tmp_path / "source-eviction.json"
    result = evict_scored_background_batch_sources(
        batch, plan, [score], [candidate], cache, output
    )

    assert result["observed_required_splits"] == ["val"]
    assert result["scored_windows"] == 1
    assert result["unscored_training_windows"] == 1
    assert result["removed_files"] == 2
    assert not h1.exists() and not l1.exists()


def test_background_source_eviction_rejects_unscored_required_window(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "gwosc-cache"
    cache.mkdir()
    h1 = cache / "H1.hdf5"
    h1.write_bytes(b"H1")
    batch = tmp_path / "batch.json"
    atomic_write_json(
        batch,
        {
            "status": "verified_development_strain_batch",
            "passed": True,
            "files": [{"path": str(h1), "sha256": file_sha256(h1)}],
        },
    )
    manifest = tmp_path / "background.jsonl"
    _jsonl(
        manifest,
        [
            {
                "window_id": "val-window",
                "split": "val",
                "source_files": {
                    "H1": {"path": str(h1), "sha256": file_sha256(h1)}
                },
            }
        ],
    )
    plan = tmp_path / "plan.json"
    atomic_write_json(
        plan,
        {
            "status": "verified_multi_segment_development_background",
            "passed": True,
            "split_strategy": "hash_threshold_v1",
            "source_batch_report_sha256s": [file_sha256(batch)],
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        },
    )

    with pytest.raises(ValueError, match="non-empty required splits"):
        evict_scored_background_batch_sources(batch, plan, [], [], cache, tmp_path / "out.json")
    assert h1.exists()
