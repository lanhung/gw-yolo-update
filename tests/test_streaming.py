from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from gwyolo.streaming import (
    calibrate_streamed_morphology_candidate_rate,
    evict_amplfi_background_batch_sources,
    evict_candidate_probability_artifacts,
    evict_mask_conditioned_background_overrides,
    evict_scored_background_batch_sources,
    merge_raw_mask_streamed_background_shards,
    merge_streamed_background_shards,
    run_streamed_background_shard,
    validate_mask_conditioned_stream_gate,
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


def test_mask_stream_gate_requires_paired_ranking_and_timing_receipts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "model.yaml"
    checkpoint.write_bytes(b"model")
    config.write_text("model: fixture\n", encoding="utf-8")
    pipeline = tmp_path / "pipeline" / "mask_search_pipeline_report.json"
    atomic_write_json(
        pipeline,
        {
            "status": "validation_only_end_to_end_mask_search_pipeline",
            "development_gates_passed": True,
            "scientific_claim_allowed": False,
            "promotion_allowed": False,
            "test_rows_read": 0,
            "test_evaluation": None,
            "checkpoint_sha256": file_sha256(checkpoint),
            "config_sha256": file_sha256(config),
            "code_commit": "calibrated-commit",
            "strength": 0.9,
        },
    )
    ranking = tmp_path / "mask-ranking.json"
    atomic_write_json(
        ranking,
        {
            "status": "completed_validation_only_mask_deglitch_gate",
            "execution_passed": True,
            "development_gates_passed": True,
            "coherent_background_scale_allowed": False,
            "scientific_claim_allowed": False,
            "locked_test_allowed": False,
            "test_rows_read": 0,
            "artifacts": {
                "pipeline_report": {
                    "path": str(pipeline),
                    "sha256": file_sha256(pipeline),
                }
            },
        },
    )
    timing_reports = {}
    injection_ranking_reports = {}
    probability_eviction_reports = {}
    score_reports = {}
    for condition in ("raw", "mask"):
        path = tmp_path / f"{condition}-candidate-timing.json"
        atomic_write_json(
            path,
            {
                "status": "validation_only_candidate_timing_calibration",
                "scientific_claim_allowed": False,
                "selection_data": "validation_injections_only",
                "test_evaluation": None,
                "methods": {
                    "local_whitened_strain_envelope_per_mask_cluster_v1": {
                        "matches": 40,
                        "empirical_timing_uncertainty_seconds": 0.008,
                        "calibration_gate_passed": True,
                    }
                },
                "source_scoring_provenance": {
                    "available": True,
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "config_sha256": file_sha256(config),
                    "code_commit": "calibrated-commit",
                },
            },
        )
        timing_reports[condition] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "gate_passed": True,
        }
        ranking_manifest = tmp_path / f"{condition}-injection-rankings.jsonl"
        ranking_manifest.write_text("", encoding="utf-8")
        ranking_report = tmp_path / f"{condition}-injection-ranking-report.json"
        candidate_report = tmp_path / f"{condition}-candidate-report.json"
        atomic_write_json(
            candidate_report,
            {"status": "single_ifo_physical_injection_candidates"},
        )
        atomic_write_json(
            ranking_report,
            {
                "status": "physical_network_injection_candidate_rankings",
                "split": "val",
                "timing_calibration_report_sha256": file_sha256(path),
                "candidate_checkpoint_sha256": file_sha256(checkpoint),
                "candidate_config_sha256": file_sha256(config),
                "candidate_code_commit": "calibrated-commit",
                "candidate_scoring_provenance_consistent": True,
                "manifest_path": str(ranking_manifest),
                "manifest_sha256": file_sha256(ranking_manifest),
            },
        )
        injection_ranking_reports[condition] = {
            "path": str(ranking_report),
            "sha256": file_sha256(ranking_report),
            "candidate_extraction_report_path": str(candidate_report),
            "candidate_extraction_report_sha256": file_sha256(candidate_report),
        }
        score_report = tmp_path / f"{condition}-score-report.json"
        atomic_write_json(score_report, {"status": "fixture-score"})
        score_reports[condition] = {
            "path": str(score_report),
            "sha256": file_sha256(score_report),
        }
        eviction_report = tmp_path / f"{condition}-probability-eviction.json"
        atomic_write_json(
            eviction_report,
            {
                "status": "verified_candidate_probability_eviction",
                "recoverable": True,
                "score_report_sha256": file_sha256(score_report),
                "candidate_extraction_report_sha256": file_sha256(candidate_report),
                "removed_files": 100,
                "removed_bytes": 409600,
            },
        )
        probability_eviction_reports[condition] = {
            "path": str(eviction_report),
            "sha256": file_sha256(eviction_report),
            "removed_files": 100,
            "removed_bytes": 409600,
        }
    timing = tmp_path / "mask-timing.json"
    timing_value = {
        "status": "completed_validation_only_mask_timing_gate",
        "scientific_claim_allowed": False,
        "locked_test_allowed": False,
        "test_rows_read": 0,
        "ranking_development_gates_passed": True,
        "timing_evaluated": True,
        "raw_timing_gate_passed": True,
        "mask_timing_gate_passed": True,
        "coherent_background_scale_allowed": True,
        "paired_injections": 100,
        "mask_validation_receipt_path": str(ranking),
        "mask_validation_receipt_sha256": file_sha256(ranking),
        "pipeline_report_path": str(pipeline),
        "pipeline_report_sha256": file_sha256(pipeline),
        "required_method": "local_whitened_strain_envelope_per_mask_cluster_v1",
        "timing_reports": timing_reports,
        "injection_ranking_reports": injection_ranking_reports,
        "probability_eviction_reports": probability_eviction_reports,
        "raw_score_report": score_reports["raw"],
        "mask_score_report": score_reports["mask"],
        "reference_ifo": "H1",
        "second_ifo": "L1",
        "physical_delay_limit_seconds": 0.01,
    }
    atomic_write_json(timing, timing_value)

    gate = validate_mask_conditioned_stream_gate(
        ranking, timing, checkpoint, config
    )
    assert gate["deglitch_strength"] == 0.9
    assert set(gate["timing_reports"]) == {"raw", "mask"}
    assert set(gate["probability_eviction_reports"]) == {"raw", "mask"}

    timing_value["mask_timing_gate_passed"] = False
    atomic_write_json(timing, timing_value)
    with pytest.raises(ValueError, match="does not authorize"):
        validate_mask_conditioned_stream_gate(ranking, timing, checkpoint, config)


def test_mask_override_eviction_waits_for_complete_rescored_arm(
    tmp_path: Path,
) -> None:
    override_root = tmp_path / "cleaning" / "arrays"
    override_root.mkdir(parents=True)
    override = override_root / "window-1.npz"
    override.write_bytes(b"cleaned numeric strain")
    cleaned_manifest = tmp_path / "cleaning" / "learned_background_deglitch.jsonl"
    _jsonl(
        cleaned_manifest,
        [
            {
                "window_id": "window-1",
                "analysis_override_path": str(override),
                "analysis_override_sha256": file_sha256(override),
            }
        ],
    )
    cleaning_report = tmp_path / "cleaning" / "learned_background_deglitch_report.json"
    atomic_write_json(
        cleaning_report,
        {
            "status": "learned_mask_background_analysis_overrides",
            "windows": 1,
            "manifest_path": str(cleaned_manifest),
            "manifest_sha256": file_sha256(cleaned_manifest),
        },
    )
    triggers = tmp_path / "mask-score" / "background_triggers.jsonl"
    _jsonl(
        triggers,
        [
            {
                "window_id": "window-1",
                "analysis_override_sha256": file_sha256(override),
            }
        ],
    )
    score_report = tmp_path / "mask-score" / "trigger_score_report.json"
    atomic_write_json(
        score_report,
        {
            "status": "real_o4a_domain_transfer_diagnostic",
            "probabilities_saved": True,
            "failed_windows": 0,
            "analysis_override_windows": 1,
            "scored_windows": 1,
            "manifest_sha256": file_sha256(cleaned_manifest),
            "triggers_path": str(triggers),
            "triggers_sha256": file_sha256(triggers),
        },
    )
    candidate_manifest = tmp_path / "mask-candidates" / "single_ifo_candidates.jsonl"
    _jsonl(candidate_manifest, [])
    candidate_report = tmp_path / "mask-candidates" / "candidate_extraction_report.json"
    atomic_write_json(
        candidate_report,
        {
            "status": "subwindow_cluster_integration_only",
            "input_windows": 1,
            "manifest_path": str(candidate_manifest),
            "manifest_sha256": file_sha256(candidate_manifest),
            "source_scoring_provenance": {
                "available": True,
                "score_report_sha256": file_sha256(score_report),
            },
        },
    )
    output = tmp_path / "mask-override-eviction.json"

    result = evict_mask_conditioned_background_overrides(
        cleaning_report,
        score_report,
        candidate_report,
        override_root,
        output,
    )

    assert result["removed_files"] == 1
    assert result["recoverable"] is True
    assert not override.exists()


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


def test_amplfi_source_eviction_requires_complete_hash_bound_export(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "gwosc-cache"
    cache.mkdir()
    sources = {}
    batch_rows = []
    for ifo in ("H1", "L1"):
        path = cache / f"{ifo}.hdf5"
        path.write_bytes(f"{ifo} public strain".encode())
        sources[ifo] = {"path": str(path), "sha256": file_sha256(path)}
        batch_rows.append({**sources[ifo], "detector": ifo})
    batch = tmp_path / "batch.json"
    atomic_write_json(
        batch,
        {
            "status": "verified_development_strain_batch",
            "passed": True,
            "files": batch_rows,
        },
    )
    manifest = tmp_path / "background.jsonl"
    _jsonl(
        manifest,
        [{"window_id": "train-1", "split": "train", "source_files": sources}],
    )
    background = tmp_path / "background-report.json"
    atomic_write_json(
        background,
        {
            "status": "verified_multi_segment_development_background",
            "passed": True,
            "split_strategy": "hash_threshold_v1",
            "source_batch_report_sha256s": [file_sha256(batch)],
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "splits": {"test": {"windows": 0}},
        },
    )
    exported = tmp_path / "exported.hdf5"
    exported.write_bytes(b"downsampled H1 L1 background")
    export = tmp_path / "export-report.json"
    atomic_write_json(
        export,
        {
            "status": "group_safe_amplfi_background",
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "split_file_counts": {"train": 1, "val": 0, "test": 0},
            "files": [
                {
                    "path": str(exported),
                    "sha256": file_sha256(exported),
                    "source_files": sources,
                }
            ],
        },
    )
    output = tmp_path / "amplfi-source-eviction.json"
    result = evict_amplfi_background_batch_sources(
        batch, background, export, cache, output
    )
    assert result["status"] == "verified_exported_amplfi_source_eviction"
    assert result["recoverable"] is True
    assert result["removed_files"] == 2
    assert all(not Path(value["path"]).exists() for value in sources.values())
    assert output.with_suffix(".json.intent.json").is_file()


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
        _chunk_samples: int,
        _verified_source_inventories: list[Path],
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


def test_raw_mask_streamed_shard_runs_both_arms_before_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gwyolo import background as background_module
    from gwyolo import candidates as candidates_module
    from gwyolo import gwosc as gwosc_module
    from gwyolo import learned_deglitch as learned_module
    from gwyolo import streaming as streaming_module
    from gwyolo import trigger as trigger_module
    from gwyolo.runtime import execution_provenance

    parent_plan = tmp_path / "parent-plan.json"
    exclusions = tmp_path / "exclusions.json"
    checkpoint = tmp_path / "model.pt"
    config = tmp_path / "config.yaml"
    coherence = tmp_path / "coherence.yaml"
    ranking = tmp_path / "ranking.json"
    timing_receipt = tmp_path / "timing-receipt.json"
    raw_timing = tmp_path / "raw-timing.json"
    mask_timing = tmp_path / "mask-timing.json"
    raw_ranking = tmp_path / "raw-ranking.json"
    mask_ranking = tmp_path / "mask-ranking.json"
    for path in (
        parent_plan,
        exclusions,
        config,
        coherence,
        ranking,
        timing_receipt,
        raw_timing,
        mask_timing,
        raw_ranking,
        mask_ranking,
    ):
        path.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    cache = tmp_path / "bounded" / "gwosc-cache"
    output = tmp_path / "paired-stream-shard"
    code_commit = str(execution_provenance()["code_commit"])

    monkeypatch.setattr(
        streaming_module,
        "validate_mask_conditioned_stream_gate",
        lambda *args: {
            "mask_validation_receipt_sha256": file_sha256(ranking),
            "mask_timing_receipt_sha256": file_sha256(timing_receipt),
            "pipeline_report_sha256": "pipeline",
            "deglitch_strength": 0.9,
            "timing_reports": {
                "raw": {
                    "path": str(raw_timing),
                    "sha256": file_sha256(raw_timing),
                    "source_code_commit": code_commit,
                },
                "mask": {
                    "path": str(mask_timing),
                    "sha256": file_sha256(mask_timing),
                    "source_code_commit": code_commit,
                },
            },
            "injection_ranking_reports": {
                "raw": {
                    "path": str(raw_ranking),
                    "sha256": file_sha256(raw_ranking),
                },
                "mask": {
                    "path": str(mask_ranking),
                    "sha256": file_sha256(mask_ranking),
                },
            },
            "reference_ifo": "H1",
            "second_ifo": "L1",
            "physical_delay_limit_seconds": 0.01,
        },
    )

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
        *_args: object,
    ) -> dict:
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        sources = []
        for ifo in ("H1", "L1"):
            path = Path(cache_root) / f"{ifo}.hdf5"
            path.write_bytes(ifo.encode())
            sources.append({"path": str(path), "sha256": file_sha256(path)})
        value = {
            "status": "verified_development_strain_batch",
            "passed": True,
            "plan_sha256": file_sha256(plan_path),
            "files": sources,
        }
        atomic_write_json(Path(output_dir) / "batch_download_report.json", value)
        return value

    def fake_background(
        batch_path: Path,
        _exclusions: Path,
        output_dir: Path,
        **kwargs: object,
    ) -> dict:
        batch = json.loads(Path(batch_path).read_text())
        sources = {
            ifo: {"path": row["path"], "sha256": row["sha256"]}
            for ifo, row in zip(("H1", "L1"), batch["files"])
        }
        manifest = Path(output_dir) / "background_windows.jsonl"
        _jsonl(
            manifest,
            [
                {
                    "window_id": "window-1",
                    "split": "val",
                    "gps_block": "block-1",
                    "gps_start": 100.0,
                    "gps_end": 108.0,
                    "duration": 8.0,
                    "observing_run": "O4a",
                    "ifos": ["H1", "L1"],
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

    score_calls = []

    def fake_score(manifest: Path, *_args: object) -> dict:
        output_dir = Path(_args[2])
        rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        score_calls.append(str(output_dir))
        probabilities = output_dir / "probabilities" / "window-1.npz"
        probabilities.parent.mkdir(parents=True, exist_ok=True)
        probabilities.write_bytes(b"probabilities")
        triggers = output_dir / "background_triggers.jsonl"
        _jsonl(
            triggers,
            [
                {
                    "window_id": row["window_id"],
                    "split": "val",
                    "gps_block": row["gps_block"],
                    "valid_ifos": ["H1", "L1"],
                    "probability_path": str(probabilities),
                    "probability_sha256": file_sha256(probabilities),
                    "analysis_override_sha256": row.get("analysis_override_sha256"),
                }
                for row in rows
            ],
        )
        report = {
            "status": "real_o4a_domain_transfer_diagnostic",
            "probabilities_saved": True,
            "failed_windows": 0,
            "scored_windows": len(rows),
            "analysis_override_windows": sum(
                bool(row.get("analysis_override_sha256")) for row in rows
            ),
            "manifest_sha256": file_sha256(manifest),
            "triggers_path": str(triggers),
            "triggers_sha256": file_sha256(triggers),
        }
        atomic_write_json(output_dir / "trigger_score_report.json", report)
        return report

    def fake_candidates(triggers: Path, output_dir: Path, *_args: object) -> dict:
        output_dir = Path(output_dir)
        manifest = output_dir / "single_ifo_candidates.jsonl"
        _jsonl(
            manifest,
            [
                {
                    "candidate_id": f"candidate-{len(score_calls)}",
                    "window_id": "window-1",
                    "split": "val",
                    "ifo": "H1",
                    "gps_peak": 104.0,
                }
            ],
        )
        report = {
            "status": "subwindow_cluster_integration_only",
            "input_windows": 1,
            "manifest_path": str(manifest),
            "manifest_sha256": file_sha256(manifest),
            "source_scoring_provenance": {"available": True},
        }
        atomic_write_json(output_dir / "candidate_extraction_report.json", report)
        return report

    def fake_apply(
        candidates: Path, calibration: Path, output_path: Path, *_args: object
    ) -> dict:
        rows = [json.loads(line) for line in Path(candidates).read_text().splitlines()]
        for row in rows:
            row["timing_empirically_calibrated"] = True
        _jsonl(Path(output_path), rows)
        result = {
            "status": "candidate_timing_calibration_applied",
            "uncalibrated_candidates": 0,
            "calibration_report_sha256": file_sha256(calibration),
        }
        atomic_write_json(
            Path(output_path).with_suffix(Path(output_path).suffix + ".report.json"),
            result,
        )
        return result

    def fake_clean(
        manifest: Path, _triggers: Path, output_dir: Path, *_args: object
    ) -> dict:
        source = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        override = Path(output_dir) / "arrays" / "window-1.npz"
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_bytes(b"cleaned")
        rows = [
            {
                **row,
                "analysis_override_path": str(override),
                "analysis_override_sha256": file_sha256(override),
            }
            for row in source
        ]
        cleaned = Path(output_dir) / "learned_background_deglitch.jsonl"
        _jsonl(cleaned, rows)
        report = {
            "status": "learned_mask_background_analysis_overrides",
            "windows": len(rows),
            "manifest_path": str(cleaned),
            "manifest_sha256": file_sha256(cleaned),
        }
        atomic_write_json(
            Path(output_dir) / "learned_background_deglitch_report.json", report
        )
        return report

    def fake_probability_evict(
        candidate: Path, score: Path, _root: Path, output_path: Path
    ) -> dict:
        result = {
            "status": "verified_candidate_probability_eviction",
            "candidate_extraction_report_sha256": file_sha256(candidate),
            "score_report_sha256": file_sha256(score),
            "removed_files": 1,
        }
        atomic_write_json(output_path, result)
        return result

    def fake_override_evict(
        clean: Path, score: Path, candidate: Path, _root: Path, output_path: Path
    ) -> dict:
        result = {
            "status": "verified_mask_conditioned_override_eviction",
            "cleaning_report_sha256": file_sha256(clean),
            "mask_score_report_sha256": file_sha256(score),
            "mask_candidate_report_sha256": file_sha256(candidate),
            "removed_files": 1,
        }
        atomic_write_json(output_path, result)
        return result

    def fake_source_evict(
        batch: Path,
        background: Path,
        _scores: object,
        _candidates: object,
        _cache: Path,
        output_path: Path,
    ) -> dict:
        result = {
            "status": "verified_scored_gwosc_source_eviction",
            "batch_download_report_sha256": file_sha256(batch),
            "background_plan_report_sha256": file_sha256(background),
            "removed_files": 2,
            "removed_bytes": 4,
        }
        atomic_write_json(output_path, result)
        return result

    monkeypatch.setattr(gwosc_module, "run_gwosc_plan_shard", fake_plan)
    monkeypatch.setattr(gwosc_module, "run_gwosc_batch_download", fake_download)
    monkeypatch.setattr(background_module, "run_batch_background_plan", fake_background)
    monkeypatch.setattr(trigger_module, "score_background_manifest", fake_score)
    monkeypatch.setattr(candidates_module, "run_candidate_extraction", fake_candidates)
    monkeypatch.setattr(
        candidates_module, "run_apply_candidate_timing_calibration", fake_apply
    )
    monkeypatch.setattr(learned_module, "run_learned_background_deglitch", fake_clean)
    monkeypatch.setattr(
        streaming_module,
        "evict_candidate_probability_artifacts",
        fake_probability_evict,
    )
    monkeypatch.setattr(
        streaming_module,
        "evict_mask_conditioned_background_overrides",
        fake_override_evict,
    )
    monkeypatch.setattr(
        streaming_module, "evict_scored_background_batch_sources", fake_source_evict
    )

    result = run_streamed_background_shard(
        parent_plan,
        exclusions,
        None,
        checkpoint,
        config,
        coherence,
        cache,
        output,
        0,
        test_fraction=0.0,
        mask_validation_receipt=ranking,
        mask_timing_receipt=timing_receipt,
    )

    assert result["status"] == "verified_streamed_raw_mask_candidate_background_shard"
    assert result["paired_raw_mask_arms"] is True
    assert set(result["split_artifacts"]["val"]["arms"]) == {"raw", "mask"}
    assert len(score_calls) == 2
    assert result["source_files_removed"] == 2
    assert run_streamed_background_shard(
        parent_plan,
        exclusions,
        None,
        checkpoint,
        config,
        coherence,
        cache,
        output,
        0,
        test_fraction=0.0,
        mask_validation_receipt=ranking,
        mask_timing_receipt=timing_receipt,
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


def test_raw_mask_stream_merge_preserves_paired_physical_background(
    tmp_path: Path,
) -> None:
    reports = []
    for index in range(2):
        shard = tmp_path / f"paired-shard-{index}"
        shard.mkdir()
        background = shard / "background.jsonl"
        window_id = f"window-{index}"
        _jsonl(
            background,
            [
                {
                    "window_id": window_id,
                    "split": "val",
                    "gps_block": f"block-{index}",
                    "gps_start": 100.0 + 8 * index,
                    "gps_end": 108.0 + 8 * index,
                    "observing_run": "O4a",
                    "ifos": ["H1", "L1"],
                }
            ],
        )
        arms = {}
        for arm, offset in (("raw", 0.0), ("mask", 0.001)):
            candidates = shard / f"{arm}-candidates.jsonl"
            _jsonl(
                candidates,
                [
                    {
                        "candidate_id": f"{arm}-candidate-{index}",
                        "window_id": window_id,
                        "split": "val",
                        "ifo": "H1",
                        "gps_peak": 104.0 + 8 * index + offset,
                        "timing_empirically_calibrated": True,
                    }
                ],
            )
            arms[arm] = {
                "calibrated_candidate_manifest_path": str(candidates),
                "calibrated_candidate_manifest_sha256": file_sha256(candidates),
            }
        report_path = shard / "report.json"
        atomic_write_json(
            report_path,
            {
                "status": "verified_streamed_raw_mask_candidate_background_shard",
                "paired_raw_mask_arms": True,
                "split_strategy": "hash_threshold_v1",
                "run_identity": {
                    "parent_plan_sha256": "parent",
                    "event_exclusions_sha256": "events",
                    "timing_calibration_report_sha256": "raw-timing",
                    "raw_timing_calibration_report_sha256": "raw-timing",
                    "mask_timing_calibration_report_sha256": "mask-timing",
                    "raw_injection_ranking_report_sha256": "raw-ranking",
                    "mask_injection_ranking_report_sha256": "mask-ranking",
                    "reference_ifo": "H1",
                    "second_ifo": "L1",
                    "physical_delay_limit_seconds": 0.01,
                    "mask_validation_receipt_sha256": "ranking",
                    "mask_timing_receipt_sha256": "timing-gate",
                    "mask_pipeline_report_sha256": "pipeline",
                    "scoring_compatibility_report_sha256": "compatibility",
                    "deglitch_strength": 0.9,
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
                    "streaming_mode": "paired_raw_mask_empirically_calibrated_timing",
                    "code_commit": "commit",
                    "shard_index": index,
                },
                "parent_selected_pairs": 2,
                "shard_count": 2,
                "pair_index_start_inclusive": index,
                "pair_index_stop_exclusive": index + 1,
                "selected_pair_ids_hash": f"pair-hash-{index}",
                "background_manifest_path": str(background),
                "background_manifest_sha256": file_sha256(background),
                "split_counts": {"train": 0, "val": 1, "test": 0},
                "split_artifacts": {"val": {"windows": 1, "arms": arms}},
            },
        )
        reports.append(report_path)

    result = merge_raw_mask_streamed_background_shards(
        reports, tmp_path / "paired-merged"
    )

    assert result["status"] == "verified_merged_streamed_raw_mask_candidate_background"
    assert result["test_rows_read"] == 0
    assert result["split_counts"] == {"train": 0, "val": 2, "test": 0}
    assert result["complete_parent_plan"] is True
    assert result["arm_merges"]["raw"]["candidate_manifests"]["val"]["candidates"] == 2
    assert result["arm_merges"]["mask"]["candidate_manifests"]["val"]["candidates"] == 2


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
