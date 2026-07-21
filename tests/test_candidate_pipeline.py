from __future__ import annotations

import json

import pytest

from gwyolo.candidate_pipeline import (
    compare_candidate_validation_pipelines,
    recalibrate_candidate_validation_pipeline_with_block_permutations,
    select_candidate_timing_method,
    validate_candidate_model_selection,
)
from gwyolo.io import file_sha256


def test_candidate_pipeline_selects_only_calibrated_local_cluster_method() -> None:
    calibration = {
        "methods": {
            "mask_profile_parabolic": {
                "calibration_gate_passed": False,
                "empirical_timing_uncertainty_seconds": 0.08,
            },
            "local_whitened_strain_envelope_per_mask_cluster_v1": {
                "calibration_gate_passed": True,
                "empirical_timing_uncertainty_seconds": 0.004,
            },
        }
    }
    method, uncertainty = select_candidate_timing_method(calibration)
    assert method == "local_whitened_strain_envelope_per_mask_cluster_v1"
    assert uncertainty == 0.004


def test_candidate_pipeline_refuses_resolution_only_timing() -> None:
    with pytest.raises(ValueError, match="exactly one passing"):
        select_candidate_timing_method(
            {
                "methods": {
                    "mask_profile_parabolic": {
                        "calibration_gate_passed": True,
                        "empirical_timing_uncertainty_seconds": 0.08,
                    }
                }
            }
        )


def test_candidate_pipeline_binds_five_seed_model_and_config(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("model: detector_set\n")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "completed_five_seed_source_safe_overlap_validation",
                "passed": True,
                "test_data_opened": False,
                "selected_checkpoint_sha256": file_sha256(checkpoint),
                "common_artifact_hashes": {
                    "config_file_sha256": file_sha256(config)
                },
            }
        )
    )
    result = validate_candidate_model_selection(selection, checkpoint, config)
    assert result["selected_checkpoint_sha256"] == file_sha256(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        validate_candidate_model_selection(selection, checkpoint, config)


def test_candidate_validation_comparison_uses_paired_injections(tmp_path) -> None:
    common_identity = {
        "background_manifest_sha256": "background",
        "injection_manifest_sha256": "injections",
        "coherence_config_sha256": "coherence",
        "reference_ifo": "H1",
        "second_ifo": "L1",
        "model_ifos": ["H1", "L1", "V1"],
        "q_values": [4, 8, 16],
        "target_sample_rate": 1024,
        "context_duration": 64.0,
        "chirp_threshold": 0.3,
        "minimum_bins": 1,
        "timing_association_window_seconds": 0.25,
        "timing_uncertainty_quantile": 0.99,
        "minimum_timing_matches": 30,
        "maximum_timing_uncertainty_seconds": 0.01,
        "truth_association_window_seconds": 0.25,
        "slide_count": 512,
        "slide_step_seconds": 8.0,
        "cluster_window_seconds": 0.1,
        "target_far_per_year": 100.0,
        "bootstrap_replicates": 10000,
        "seed": 20260720,
        "code_commit": "same-scorer",
    }
    report_paths = {}
    for name, recovered_count, timing in (
        ("baseline", 10, 0.008),
        ("promoted", 90, 0.006),
    ):
        manifest = tmp_path / f"{name}-rankings.jsonl"
        rows = [
            {
                "injection_id": f"i-{index}",
                "waveform_id": f"w-{index}",
                "gps_block": f"b-{index}",
                "source_family": "BBH" if index < 50 else "BNS",
                "stratum": "BBH" if index < 50 else "BNS",
                "vt_weight": 1.0,
                "vt_weight_unit": "arbitrary",
                "ranking_score": 1.0 if index < recovered_count else 0.0,
            }
            for index in range(100)
        ]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = tmp_path / f"{name}-pipeline.json"
        report.write_text(
            json.dumps(
                {
                    "status": "validation_only_clustered_candidate_search_pipeline",
                    "test_evaluation": None,
                    "run_identity": common_identity,
                    "model_selection": {"selected": True}
                    if name == "promoted"
                    else None,
                    "empirical_timing_uncertainty_seconds": timing,
                    "injection_rankings": {
                        "manifest_path": str(manifest),
                        "manifest_sha256": file_sha256(manifest),
                        "ranked_injections": len(rows),
                    },
                    "frozen_search": {
                        "publication_calibration_eligible": True,
                        "calibration": {"threshold": 0.5},
                    },
                }
            )
        )
        report_paths[name] = report
    config = tmp_path / "promotion.yaml"
    config.write_text(
        """candidate_validation_promotion:
  minimum_weighted_efficiency_gain: 0.01
  maximum_stratum_efficiency_regression: 0.03
  maximum_regressed_strata: 0
  maximum_promoted_timing_uncertainty_seconds: 0.01
  maximum_timing_uncertainty_regression_seconds: 0.002
  bootstrap_replicates: 1000
  seed: 7
"""
    )
    result = compare_candidate_validation_pipelines(
        report_paths["baseline"],
        report_paths["promoted"],
        config,
        tmp_path / "comparison.json",
    )
    assert result["passed"]
    assert result["scale_continuous_background"]
    assert result["weighted_efficiency_delta_promoted_minus_baseline"] == pytest.approx(
        0.8
    )
    assert result["paired_bootstrap_95"][0] > 0

    promoted = json.loads(report_paths["promoted"].read_text())
    promoted["run_identity"] = {**common_identity, "code_commit": "different"}
    report_paths["promoted"].write_text(json.dumps(promoted))
    with pytest.raises(ValueError, match="not paired"):
        compare_candidate_validation_pipelines(
            report_paths["baseline"],
            report_paths["promoted"],
            config,
            tmp_path / "bad-comparison.json",
        )


def test_candidate_pipeline_block_recalibration_closes_frozen_schedule(tmp_path) -> None:
    background = tmp_path / "background.jsonl"
    background_rows = []
    candidates = []
    for block_index in range(3):
        block_start = 1000 + block_index * 256
        for slot in range(2):
            background_rows.append(
                {
                    "window_id": f"w-{block_index}-{slot}",
                    "split": "val",
                    "gps_start": block_start + slot * 8,
                    "gps_end": block_start + (slot + 1) * 8,
                    "gps_block": f"gps:{block_start}:256",
                    "ifos": ["H1", "L1"],
                }
            )
        for ifo, offset in (("H1", 1.0), ("L1", 1.005)):
            candidates.append(
                {
                    "candidate_id": f"{ifo}-{block_index}",
                    "window_id": f"w-{block_index}-0",
                    "split": "val",
                    "ifo": ifo,
                    "gps_peak": block_start + offset,
                    "chirp_score": 0.8 if ifo == "H1" else 0.7,
                    "glitch_score_at_peak": 0.1,
                    "bin_width_seconds": 0.08,
                    "timing_resolution_seconds": 1 / 1024,
                    "timing_empirically_calibrated": True,
                    "empirical_timing_uncertainty_seconds": 0.001,
                    "timing_calibration_report_sha256": "a" * 64,
                    "candidate_checkpoint_sha256": "b" * 64,
                    "candidate_config_sha256": "c" * 64,
                    "candidate_code_commit": "deadbee",
                }
            )
    background.write_text(
        "".join(json.dumps(row) + "\n" for row in background_rows),
        encoding="utf-8",
    )
    candidate_manifest = tmp_path / "candidates.jsonl"
    candidate_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8"
    )
    injection_manifest = tmp_path / "injection-rankings.jsonl"
    injection_manifest.write_text(
        json.dumps(
            {
                "split": "val",
                "injection_id": "i1",
                "waveform_id": "wave1",
                "gps_block": "gps:5000:256",
                "source_family": "BBH",
                "stratum": "BBH",
                "vt_weight": 1.0,
                "ranking_score": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "candidate_checkpoint_sha256": "b" * 64,
        "candidate_config_sha256": "c" * 64,
        "candidate_code_commit": "deadbee",
        "timing_calibration_report_sha256": "a" * 64,
        "physical_delay_limit_seconds": 0.010,
        "empirical_timing_uncertainty_seconds": 0.001,
    }
    injection_report = tmp_path / "injection-ranking-report.json"
    injection_report.write_text(
        json.dumps(
            {
                "status": "physical_network_injection_candidate_rankings",
                "split": "val",
                "manifest_path": str(injection_manifest),
                "manifest_sha256": file_sha256(injection_manifest),
                "reference_ifo": "H1",
                "second_ifo": "L1",
                "timing_calibration_consistent": True,
                "candidate_scoring_provenance_consistent": True,
                **provenance,
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source-pipeline.json"
    source.write_text(
        json.dumps(
            {
                "status": "validation_only_clustered_candidate_search_pipeline",
                "test_evaluation": None,
                "run_identity": {
                    "background_manifest_sha256": file_sha256(background),
                    "reference_ifo": "H1",
                    "second_ifo": "L1",
                    "target_far_per_year": 1_000_000,
                    "cluster_window_seconds": 0.1,
                    "bootstrap_replicates": 20,
                    "seed": 1,
                },
                "time_slides": {
                    "candidate_manifest_sha256": file_sha256(candidate_manifest)
                },
                "injection_ranking_report_sha256": file_sha256(injection_report),
                "physical_delay_limit_seconds": 0.010,
                "empirical_timing_uncertainty_seconds": 0.001,
                "coincidence_window_seconds": 0.012,
                **provenance,
            }
        ),
        encoding="utf-8",
    )
    result = recalibrate_candidate_validation_pipeline_with_block_permutations(
        source,
        background,
        candidate_manifest,
        injection_report,
        tmp_path / "recalibrated",
    )
    assert result["frozen_search"]["publication_calibration_eligible"] is True
    assert result["background_resampling_method"] == (
        "circular_gps_block_relative_window_permutation_v1"
    )
    assert result["time_slides"]["equivalent_live_time_seconds"] == 96
    resumed = recalibrate_candidate_validation_pipeline_with_block_permutations(
        source,
        background,
        candidate_manifest,
        injection_report,
        tmp_path / "recalibrated",
    )
    assert resumed == result
    with pytest.raises(ValueError, match="another identity"):
        recalibrate_candidate_validation_pipeline_with_block_permutations(
            source,
            background,
            candidate_manifest,
            injection_report,
            tmp_path / "recalibrated",
            zero_count_confidence=0.8,
        )
