from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from gwyolo.streaming import (
    calibrate_streamed_morphology_candidate_rate,
    evict_candidate_probability_artifacts,
    evict_scored_background_batch_sources,
    merge_streamed_background_shards,
    run_streamed_background_shard,
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


def test_streamed_background_shard_resumes_after_verified_source_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gwyolo import background as background_module
    from gwyolo import gwosc as gwosc_module

    parent_plan = tmp_path / "parent-plan.json"
    exclusions = tmp_path / "exclusions.json"
    timing = tmp_path / "timing.json"
    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "config.yaml"
    coherence = tmp_path / "coherence.yaml"
    for path in (parent_plan, exclusions, timing, config, coherence):
        path.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    cache = tmp_path / "bounded" / "gwosc-cache"
    output = tmp_path / "stream-shard"

    def fake_plan(_parent: Path, destination: Path, index: int, count: int) -> dict:
        value = {
            "status": "development_acquisition_plan",
            "parent_plan_sha256": file_sha256(parent_plan),
            "parent_selected_pairs": 1,
            "shard_count": 1,
            "shard_index": index,
            "pairs_per_shard": count,
            "pair_index_start_inclusive": 0,
            "pair_index_stop_exclusive": 1,
            "selected_pair_ids_hash": "pair-hash",
        }
        atomic_write_json(destination, value)
        return value

    def fake_download(
        plan_path: Path,
        cache_root: Path,
        output_dir: Path,
        _maximum_pairs: None,
        _workers: int,
    ) -> dict:
        cache_path = Path(cache_root)
        cache_path.mkdir(parents=True, exist_ok=True)
        h1 = cache_path / "H1.hdf5"
        l1 = cache_path / "L1.hdf5"
        h1.write_bytes(b"H1 strain")
        l1.write_bytes(b"L1 strain")
        value = {
            "status": "verified_development_strain_batch",
            "passed": True,
            "plan_sha256": file_sha256(plan_path),
            "files": [
                {"path": str(h1), "sha256": file_sha256(h1)},
                {"path": str(l1), "sha256": file_sha256(l1)},
            ],
        }
        atomic_write_json(Path(output_dir) / "batch_download_report.json", value)
        return value

    def fake_background(
        batch_path: Path,
        _exclusions: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> dict:
        batch = json.loads(Path(batch_path).read_text(encoding="utf-8"))
        sources = {
            Path(row["path"]).stem: {"path": row["path"], "sha256": row["sha256"]}
            for row in batch["files"]
        }
        manifest = Path(output_dir) / "background_windows.jsonl"
        _jsonl(
            manifest,
            [
                {
                    "window_id": "train-only-window",
                    "split": "train",
                    "source_files": sources,
                }
            ],
        )
        value = {
            "status": "verified_multi_segment_development_background",
            "passed": True,
            "split_strategy": kwargs["split_strategy"],
            "source_batch_report_sha256s": [file_sha256(batch_path)],
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        }
        atomic_write_json(Path(output_dir) / "background_plan_report.json", value)
        return value

    monkeypatch.setattr(gwosc_module, "run_gwosc_plan_shard", fake_plan)
    monkeypatch.setattr(gwosc_module, "run_gwosc_batch_download", fake_download)
    monkeypatch.setattr(background_module, "run_batch_background_plan", fake_background)

    result = run_streamed_background_shard(
        parent_plan,
        exclusions,
        timing,
        checkpoint,
        config,
        coherence,
        cache,
        output,
        0,
    )

    assert result["status"] == "verified_streamed_candidate_background_shard"
    assert result["split_counts"] == {"train": 1, "val": 0, "test": 0}
    assert result["source_files_removed"] == 2
    assert not (cache / "H1.hdf5").exists()
    assert run_streamed_background_shard(
        parent_plan,
        exclusions,
        timing,
        checkpoint,
        config,
        coherence,
        cache,
        output,
        0,
    ) == result


def test_streamed_background_merge_checks_global_groups_and_pair_ranges(
    tmp_path: Path,
) -> None:
    reports = []
    for index, split in enumerate(("val", "test")):
        shard = tmp_path / f"shard-{index}"
        shard.mkdir()
        background = shard / "background.jsonl"
        window_id = f"window-{index}"
        _jsonl(
            background,
            [
                {
                    "window_id": window_id,
                    "split": split,
                    "gps_block": f"block-{index}",
                    "gps_start": 100.0 + 8 * index,
                    "gps_end": 108.0 + 8 * index,
                    "observing_run": "O4a",
                    "ifos": ["H1", "L1"],
                }
            ],
        )
        candidates = shard / f"{split}-candidates.jsonl"
        _jsonl(
            candidates,
            [
                {
                    "candidate_id": f"candidate-{index}",
                    "window_id": window_id,
                    "split": split,
                    "ifo": "H1",
                    "gps_peak": 104.0 + 8 * index,
                    "timing_empirically_calibrated": True,
                }
            ],
        )
        report_path = shard / "report.json"
        atomic_write_json(
            report_path,
            {
                "status": "verified_streamed_candidate_background_shard",
                "split_strategy": "hash_threshold_v1",
                "run_identity": {
                    "parent_plan_sha256": "parent",
                    "event_exclusions_sha256": "events",
                    "timing_calibration_report_sha256": "timing",
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "coherence_config_sha256": "coherence",
                    "validation_fraction": 0.2,
                    "test_fraction": 0.2,
                    "seed": 7,
                    "model_ifos": ["H1", "L1", "V1"],
                    "q_values": [4.0, 8.0, 16.0],
                    "target_sample_rate": 1024,
                    "context_duration": 64.0,
                    "chirp_threshold": 0.3,
                    "minimum_bins": 1,
                    "code_commit": "commit",
                    "shard_index": index,
                },
                "parent_selected_pairs": 2,
                "shard_count": 2,
                "pair_index_start_inclusive": index,
                "pair_index_stop_exclusive": index + 1,
                "background_manifest_path": str(background),
                "background_manifest_sha256": file_sha256(background),
                "split_artifacts": {
                    split: {
                        "calibrated_candidate_manifest_path": str(candidates),
                        "calibrated_candidate_manifest_sha256": file_sha256(candidates),
                    }
                },
            },
        )
        reports.append(report_path)

    result = merge_streamed_background_shards(reports, tmp_path / "merged")

    assert result["complete_parent_plan"] is True
    assert result["cross_split_gps_block_overlap"] is False
    assert result["split_counts"] == {"train": 0, "val": 1, "test": 1}
    assert result["observing_runs"] == {"O4a": 2}
    assert result["available_ifos"] == {"H1": 2, "L1": 2}
    assert result["detector_subset_counts"] == {"H1L1": 2}
    assert result["zero_lag_live_time_seconds"] == 16
    assert result["detector_time_seconds"] == 32
    assert result["split_live_time_seconds"] == {
        "train": 0,
        "val": 8,
        "test": 8,
    }
    assert result["candidate_manifests"]["val"]["candidates"] == 1
    assert result["candidate_manifests"]["test"]["candidates"] == 1
    with pytest.raises(FileExistsError, match="immutable"):
        merge_streamed_background_shards(reports, tmp_path / "merged")


def test_streamed_background_merge_accepts_verified_parent_extension_lineage(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base-plan.json"
    base_pairs = [{"pair_id": "pair-0"}]
    atomic_write_json(
        base,
        {
            "status": "development_acquisition_plan",
            "locked_evaluation_data": False,
            "selected_pairs": 1,
            "pairs": base_pairs,
        },
    )
    extended = tmp_path / "extended-plan.json"
    extended_pairs = [*base_pairs, {"pair_id": "pair-1"}]
    atomic_write_json(
        extended,
        {
            "status": "development_acquisition_plan",
            "locked_evaluation_data": False,
            "selected_pairs": 2,
            "pairs": extended_pairs,
            "selection_rule": "frozen_prefix_stratified_complement_v1",
            "candidate_scores_inspected": False,
            "base_parent_plan_path": str(base),
            "base_parent_plan_sha256": file_sha256(base),
            "base_selected_pairs": 1,
            "base_pair_ids_hash": canonical_hash(["pair-0"], 64),
        },
    )

    reports = []
    for index, parent in enumerate((base, extended)):
        shard = tmp_path / f"lineage-shard-{index}"
        shard.mkdir()
        background = shard / "background.jsonl"
        _jsonl(
            background,
            [
                {
                    "window_id": f"lineage-window-{index}",
                    "split": "val",
                    "gps_block": f"gps:{100 + 256 * index}:256",
                    "gps_start": 100.0 + 256 * index,
                    "gps_end": 108.0 + 256 * index,
                    "observing_run": "O4a",
                    "ifos": ["H1", "L1"],
                }
            ],
        )
        candidates = shard / "candidates.jsonl"
        _jsonl(candidates, [])
        report = shard / "report.json"
        atomic_write_json(
            report,
            {
                "status": "verified_streamed_morphology_background_shard",
                "split_strategy": "hash_threshold_v1",
                "run_identity": {
                    "parent_plan_sha256": file_sha256(parent),
                    "event_exclusions_sha256": "events",
                    "timing_calibration_report_sha256": None,
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "coherence_config_sha256": "coherence",
                    "validation_fraction": 0.2,
                    "test_fraction": 0.0,
                    "seed": 7,
                    "model_ifos": ["H1", "L1", "V1"],
                    "q_values": [4.0, 8.0, 16.0],
                    "target_sample_rate": 1024,
                    "context_duration": 64.0,
                    "chirp_threshold": 0.3,
                    "minimum_bins": 1,
                    "pairs_per_shard": 1,
                    "streaming_mode": "morphology_only_validation",
                    "code_commit": "commit",
                    "shard_index": index,
                },
                "parent_selected_pairs": 1 + index,
                "shard_count": 1 + index,
                "pair_index_start_inclusive": index,
                "pair_index_stop_exclusive": index + 1,
                "selected_pair_ids_hash": canonical_hash([f"pair-{index}"], 64),
                "background_manifest_path": str(background),
                "background_manifest_sha256": file_sha256(background),
                "split_artifacts": {
                    "val": {
                        "candidate_manifest_path": str(candidates),
                        "candidate_manifest_sha256": file_sha256(candidates),
                    }
                },
            },
        )
        reports.append(report)

    with pytest.raises(ValueError, match="one parent plan"):
        merge_streamed_background_shards(reports, tmp_path / "no-lineage")
    merged = merge_streamed_background_shards(
        reports, tmp_path / "lineage-merged", extended
    )
    assert merged["complete_parent_plan"] is True
    assert merged["parent_selected_pairs"] == 2
    assert merged["parent_shard_count"] == 2
    assert merged["parent_plan_lineage"]["base_parent_plan_sha256"] == file_sha256(
        base
    )
    assert merged["common_run_identity"]["parent_plan_sha256"] == file_sha256(
        extended
    )

    bad = json.loads(reports[1].read_text())
    bad["selected_pair_ids_hash"] = "wrong"
    bad_report = tmp_path / "bad-lineage-report.json"
    atomic_write_json(bad_report, bad)
    with pytest.raises(ValueError, match="pair IDs differ"):
        merge_streamed_background_shards(
            [reports[0], bad_report], tmp_path / "bad-lineage", extended
        )


def test_streamed_morphology_merge_retains_uncalibrated_validation_candidates(
    tmp_path: Path,
) -> None:
    reports = []
    for index in range(2):
        shard = tmp_path / f"morph-{index}"
        shard.mkdir()
        background = shard / "background.jsonl"
        candidates = shard / "candidates.jsonl"
        window_id = f"window-{index}"
        _jsonl(
            background,
            [
                {
                    "window_id": window_id,
                    "split": "val",
                    "gps_block": f"block-{index}",
                    "gps_start": 200.0 + 8 * index,
                    "gps_end": 208.0 + 8 * index,
                    "observing_run": "O4a",
                    "ifos": ["H1", "L1"],
                }
            ],
        )
        _jsonl(
            candidates,
            [
                {
                    "candidate_id": f"candidate-{index}",
                    "window_id": window_id,
                    "split": "val",
                    "ifo": "H1",
                    "gps_peak": 204.0 + 8 * index,
                    "timing_empirically_calibrated": False,
                }
            ],
        )
        report_path = shard / "report.json"
        atomic_write_json(
            report_path,
            {
                "status": "verified_streamed_morphology_background_shard",
                "split_strategy": "hash_threshold_v1",
                "run_identity": {
                    "parent_plan_sha256": "parent",
                    "event_exclusions_sha256": "events",
                    "timing_calibration_report_sha256": None,
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "coherence_config_sha256": "coherence",
                    "validation_fraction": 0.2,
                    "test_fraction": 0.0,
                    "seed": 7,
                    "model_ifos": ["H1", "L1", "V1"],
                    "q_values": [4.0, 8.0, 16.0],
                    "target_sample_rate": 1024,
                    "context_duration": 64.0,
                    "chirp_threshold": 0.3,
                    "minimum_bins": 1,
                    "streaming_mode": "morphology_only_validation",
                    "code_commit": "commit",
                    "shard_index": index,
                },
                "parent_selected_pairs": 2,
                "shard_count": 2,
                "pair_index_start_inclusive": index,
                "pair_index_stop_exclusive": index + 1,
                "background_manifest_path": str(background),
                "background_manifest_sha256": file_sha256(background),
                "split_artifacts": {
                    "val": {
                        "candidate_manifest_path": str(candidates),
                        "candidate_manifest_sha256": file_sha256(candidates),
                    }
                },
            },
        )
        reports.append(report_path)

    result = merge_streamed_background_shards(reports, tmp_path / "morph-merged")

    assert result["status"] == "verified_merged_streamed_morphology_background"
    assert result["morphology_only"] is True
    assert result["network_coherence_claim_allowed"] is False
    assert result["candidate_manifests"]["val"]["candidates"] == 2


def test_morphology_rate_calibration_uses_detector_time_not_window_count(
    tmp_path: Path,
) -> None:
    background = tmp_path / "background.jsonl"
    _jsonl(
        background,
        [
            {
                "window_id": "w0",
                "split": "val",
                "gps_block": "b0",
                "gps_start": 0.0,
                "gps_end": 10.0,
                "ifos": ["H1", "L1"],
            },
            {
                "window_id": "w1",
                "split": "val",
                "gps_block": "b1",
                "gps_start": 10.0,
                "gps_end": 20.0,
                "ifos": ["H1"],
            },
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    _jsonl(
        candidates,
        [
            {
                "candidate_id": "c0",
                "window_id": "w0",
                "split": "val",
                "ifo": "H1",
                "gps_peak": 5.0,
                "chirp_score": 0.9,
                "timing_empirically_calibrated": False,
            },
            {
                "candidate_id": "c1",
                "window_id": "w0",
                "split": "val",
                "ifo": "L1",
                "gps_peak": 6.0,
                "chirp_score": 0.8,
                "timing_empirically_calibrated": False,
            },
        ],
    )
    merge = tmp_path / "merge.json"
    atomic_write_json(
        merge,
        {
            "status": "verified_merged_streamed_morphology_background",
            "morphology_only": True,
            "test_evaluation": None,
            "split_counts": {"train": 0, "val": 2, "test": 0},
            "common_run_identity": {"chirp_threshold": 0.5},
            "background_manifest_path": str(background),
            "background_manifest_sha256": file_sha256(background),
            "candidate_manifests": {
                "val": {
                    "path": str(candidates),
                    "sha256": file_sha256(candidates),
                    "candidates": 2,
                }
            },
        },
    )
    seconds_per_year = 31_557_600.0
    target_rate = seconds_per_year / 30.0
    report = calibrate_streamed_morphology_candidate_rate(
        merge, target_rate, tmp_path / "calibration.json"
    )
    assert report["detector_time_seconds"] == pytest.approx(30.0)
    assert report["exposure_by_ifo_seconds"] == {"H1": 20.0, "L1": 10.0}
    assert report["target_expected_count"] == pytest.approx(1.0)
    assert report["calibration"]["threshold"] == pytest.approx(0.9)
    assert report["calibration"]["background_count"] == 1
    assert report["network_far_claim_allowed"] is False
