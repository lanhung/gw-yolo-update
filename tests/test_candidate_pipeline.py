from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.candidate_pipeline import (
    compare_candidate_validation_pipelines,
    freeze_raw_mask_detector_set_ranking_successor,
    recalibrate_candidate_validation_pipeline_with_block_permutations,
    recalibrate_candidate_validation_pipeline_with_detector_sets,
    select_candidate_timing_method,
    validate_candidate_model_selection,
)
from gwyolo.io import file_sha256
from gwyolo.search import bind_candidate_search_calibration_to_independent_endpoint


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


def test_raw_mask_detector_set_ranking_successor_replays_sources(
    tmp_path: Path,
) -> None:
    timing_root = tmp_path / "timing"
    timing_root.mkdir()
    timing_reports = {}
    ranking_reports = {}
    score_reports = {}
    variable_reports = {}
    for arm in ("raw", "mask"):
        timing_report = timing_root / f"{arm}-timing.json"
        timing_report.write_text(json.dumps({"arm": arm}))
        timing_reports[arm] = {
            "path": str(timing_report.resolve()),
            "sha256": file_sha256(timing_report),
        }
        trigger = timing_root / f"{arm}-triggers.jsonl"
        trigger.write_text('{"injection_id":"i"}\n')
        score = timing_root / f"{arm}-score.json"
        score.write_text(
            json.dumps({"triggers_path": str(trigger.resolve())})
        )
        score_reports[f"{arm}_score_report"] = {
            "path": str(score.resolve()),
            "sha256": file_sha256(score),
        }
        candidate_dir = timing_root / f"{arm}_injection_candidates"
        candidate_dir.mkdir()
        candidate_report = (
            candidate_dir / "injection_candidate_extraction_report.json"
        )
        candidate_report.write_text(json.dumps({"arm": arm}))
        calibrated = (
            timing_root / f"{arm}_injection_candidates_calibrated.jsonl"
        )
        calibrated.write_text('{"candidate_id":"c"}\n')
        source_ranking = timing_root / f"{arm}-source-ranking.json"
        source_ranking.write_text(
            json.dumps(
                {
                    "status": (
                        "physical_network_injection_candidate_rankings"
                    ),
                    "split": "val",
                    "injection_trigger_manifest_sha256": file_sha256(
                        trigger
                    ),
                    "candidate_manifest_sha256": file_sha256(calibrated),
                    "candidate_checkpoint_sha256": "a" * 64,
                    "candidate_config_sha256": "b" * 64,
                    "candidate_code_commit": "deadbee",
                }
            )
        )
        ranking_reports[arm] = {
            "path": str(source_ranking.resolve()),
            "sha256": file_sha256(source_ranking),
            "candidate_extraction_report_path": str(
                candidate_report.resolve()
            ),
            "candidate_extraction_report_sha256": file_sha256(
                candidate_report
            ),
        }
        variable_manifest = timing_root / f"{arm}-variable.jsonl"
        variable_manifest.write_text(
            '{"split":"val","injection_id":"i"}\n'
        )
        variable_report = timing_root / f"{arm}-variable-report.json"
        variable_reports[arm] = variable_report
        variable_report.write_text(
            json.dumps(
                {
                    "status": (
                        "physical_variable_detector_set_injection_candidate_rankings"
                    ),
                    "split": "val",
                    "config_sha256": file_sha256(
                        Path(__file__).parents[1]
                        / "configs"
                        / "network_coherence_h1_l1_v1.yaml"
                    ),
                    "injection_trigger_manifest_sha256": file_sha256(
                        trigger
                    ),
                    "candidate_manifest_sha256": file_sha256(calibrated),
                    "timing_calibration_report_sha256": file_sha256(
                        timing_report
                    ),
                    "candidate_checkpoint_sha256": "a" * 64,
                    "candidate_config_sha256": "b" * 64,
                    "candidate_code_commit": "deadbee",
                    "timing_calibration_consistent": True,
                    "candidate_scoring_provenance_consistent": True,
                    "required_detector_subsets": [
                        "H1+L1",
                        "H1+V1",
                        "L1+V1",
                        "H1+L1+V1",
                    ],
                    "manifest_path": str(variable_manifest.resolve()),
                    "manifest_sha256": file_sha256(variable_manifest),
                }
            )
        )
    timing_receipt = tmp_path / "mask-timing.json"
    timing_receipt.write_text(
        json.dumps(
            {
                "status": "completed_validation_only_mask_timing_gate",
                "coherent_background_scale_allowed": True,
                "raw_timing_gate_passed": True,
                "mask_timing_gate_passed": True,
                "test_rows_read": 0,
                "locked_test_allowed": False,
                "timing_reports": timing_reports,
                "injection_ranking_reports": ranking_reports,
                **score_reports,
            }
        )
    )
    network_config = (
        Path(__file__).parents[1]
        / "configs"
        / "network_coherence_h1_l1_v1.yaml"
    )
    result = freeze_raw_mask_detector_set_ranking_successor(
        timing_receipt,
        variable_reports["raw"],
        variable_reports["mask"],
        network_config,
        tmp_path / "successor.json",
    )
    assert set(result["arms"]) == {"raw", "mask"}
    assert result["arms"]["mask"]["timing_report"]["sha256"] == (
        timing_reports["mask"]["sha256"]
    )

    tampered = json.loads(
        variable_reports["mask"].read_text(encoding="utf-8")
    )
    tampered["candidate_manifest_sha256"] = "f" * 64
    bad_mask = tmp_path / "bad-mask-variable-report.json"
    bad_mask.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="lineage replay"):
        freeze_raw_mask_detector_set_ranking_successor(
            timing_receipt,
            variable_reports["raw"],
            bad_mask,
            network_config,
            tmp_path / "bad-successor.json",
        )


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
                    "five_seed_stability": {
                        "status": "five_seed_reproducibility_gate_v1",
                        "passed": True,
                    },
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


def test_continuous_calibration_binds_independent_endpoint_and_model(tmp_path) -> None:
    components = {}
    for label in (
        "purpose_partition",
        "injection_plan",
        "waveform_validation",
        "materialization",
        "snr_annotation",
        "arrival_annotation",
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        components[label] = {"path": str(path), "sha256": file_sha256(path)}
    endpoint_background = tmp_path / "endpoint-background.jsonl"
    endpoint_injections = tmp_path / "endpoint-injections.jsonl"
    endpoint_background.write_text('{"gps_block":"calibration"}\n', encoding="utf-8")
    endpoint_injections.write_text('{"gps_block":"injection"}\n', encoding="utf-8")
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "status": "frozen_gps_and_purpose_disjoint_validation_endpoint",
                "passed": True,
                "scientific_claim_allowed": False,
                "test_rows_read": 0,
                "test_evaluation": None,
                "rows": 3000,
                "candidate_calibration_unique_gps_blocks": 25,
                "injection_validation_unique_gps_blocks": 25,
                "purpose_gps_block_overlap": 0,
                "component_reports": components,
                "candidate_calibration_background_manifest_path": str(
                    endpoint_background
                ),
                "candidate_calibration_background_manifest_sha256": file_sha256(
                    endpoint_background
                ),
                "injection_arrival_manifest_path": str(endpoint_injections),
                "injection_arrival_manifest_sha256": file_sha256(endpoint_injections),
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                    "status": "completed_five_seed_source_safe_overlap_validation",
                    "passed": True,
                    "five_seed_stability": {
                        "status": "five_seed_reproducibility_gate_v1",
                        "passed": True,
                    },
                    "test_data_opened": False,
                "selected_checkpoint_sha256": "checkpoint",
                "common_artifact_hashes": {"config_file_sha256": "config"},
            }
        ),
        encoding="utf-8",
    )
    ranking = tmp_path / "injection-ranking.json"
    ranking.write_text(json.dumps({"status": "ranked"}), encoding="utf-8")
    slides = tmp_path / "time-slides.json"
    slides.write_text(json.dumps({"status": "slides"}), encoding="utf-8")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps(
            {
                "status": "validation_only_clustered_candidate_search_pipeline",
                "scientific_claim_allowed": False,
                "test_evaluation": None,
                "run_identity": {
                    "injection_manifest_sha256": file_sha256(endpoint_injections),
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "model_selection_report_sha256": file_sha256(selection),
                },
                "model_selection": {
                    "model_selection_report_path": str(selection),
                    "model_selection_report_sha256": file_sha256(selection),
                },
                "injection_ranking_report_sha256": file_sha256(ranking),
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"

    def write_calibration(injection_blocks: list[str]) -> None:
        calibration.write_text(
            json.dumps(
                {
                    "status": "frozen_validation_candidate_search_calibration",
                    "scientific_claim_allowed": False,
                    "test_evaluation": None,
                    "selection_data": "validation_candidate_block_permutations_only",
                    "publication_calibration_eligible": True,
                    "target_far_has_at_least_one_expected_background_count": True,
                    "target_far_per_year": 0.1,
                    "bootstrap_replicates": 10000,
                    "slide_schedule_audit": {
                        "passed": True,
                        "schedule_kind": "gps_block_permutation",
                    },
                    "calibration": {"threshold": 0.75},
                    "background_dependence_audit": {
                        "status": "candidate_background_dependence_audit_v1",
                        "passed": True,
                        "split": "val",
                        "threshold": 0.75,
                        "three_way_cluster_bootstrap": {"replicates": 10000},
                        "background_manifest": {
                            "path": str(endpoint_background.resolve()),
                            "sha256": file_sha256(endpoint_background),
                        },
                        "time_slide_report": {
                            "path": str(slides.resolve()),
                            "sha256": file_sha256(slides),
                        },
                    },
                    "validation_background_gps_blocks": ["background-block"],
                    "validation_injection_gps_blocks": injection_blocks,
                        "validation_injection_diagnostic": {
                            "injections": 3000,
                            "bootstrap_independence": {
                                "status": "injection_bootstrap_independence_audit_v1",
                                "passed": True,
                                "method": "gps_block_then_paired_injection_hierarchical_bootstrap_v1",
                                "physical_groups": 25,
                            },
                        },
                    "validation_time_slide_report_path": str(slides),
                    "validation_time_slide_report_sha256": file_sha256(slides),
                    "validation_injection_ranking_report_path": str(ranking),
                    "validation_injection_ranking_report_sha256": file_sha256(ranking),
                }
            ),
            encoding="utf-8",
        )

    write_calibration(["injection-block"])
    result = bind_candidate_search_calibration_to_independent_endpoint(
        endpoint,
        pipeline,
        calibration,
        tmp_path / "binding.json",
    )
    assert result["passed"]
    assert result["validation_purpose_gps_block_overlap"] == 0
    assert result["target_far_per_year"] == 0.1
    assert result["independent_validation_rows"] == 3000

    expanded_background = tmp_path / "expanded-background.jsonl"
    expanded_background.write_text(
        '{"gps_block":"expanded-background"}\n',
        encoding="utf-8",
    )
    expanded_candidates = tmp_path / "expanded-candidates.jsonl"
    expanded_candidates.write_text(
        '{"candidate_id":"expanded"}\n',
        encoding="utf-8",
    )
    slides.write_text(
        json.dumps(
            {
                "status": (
                    "variable_detector_set_block_permutation_background"
                ),
                "candidate_manifest_path": str(
                    expanded_candidates.resolve()
                ),
                "candidate_manifest_sha256": file_sha256(
                    expanded_candidates
                ),
            }
        ),
        encoding="utf-8",
    )
    parent = tmp_path / "expanded-parent.json"
    parent.write_text(json.dumps({"status": "development_acquisition_plan"}))
    authorization = tmp_path / "expanded-authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "status": (
                    "authorized_validation_candidate_continuous_background_plan"
                ),
                "authorization_id": "expanded-auth",
                "passed": True,
                "scientific_claim_allowed": False,
                "candidate_scores_inspected": False,
                "test_rows_read": 0,
                "test_evaluation": None,
                "independent_validation_endpoint": {
                    "path": str(endpoint.resolve()),
                    "sha256": file_sha256(endpoint),
                },
                "parent_plan": {
                    "path": str(parent.resolve()),
                    "sha256": file_sha256(parent),
                },
                "authorization_identity": {
                    "target_far_per_year": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    merge = tmp_path / "expanded-merge.json"
    merge.write_text(
        json.dumps(
            {
                "status": "verified_merged_streamed_candidate_background",
                "scientific_claim_allowed": False,
                "complete_parent_plan": True,
                "split_counts": {"test": 0},
                "common_run_identity": {
                    "checkpoint_sha256": "checkpoint",
                    "config_sha256": "config",
                    "parent_plan_sha256": file_sha256(parent),
                },
                "background_manifest_path": str(
                    expanded_background.resolve()
                ),
                "background_manifest_sha256": file_sha256(
                    expanded_background
                ),
                "candidate_manifests": {
                    "val": {
                        "path": str(expanded_candidates.resolve()),
                        "sha256": file_sha256(expanded_candidates),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    expanded_calibration_value = json.loads(
        calibration.read_text(encoding="utf-8")
    )
    expanded_calibration_value["background_dependence_audit"][
        "background_manifest"
    ] = {
        "path": str(expanded_background.resolve()),
        "sha256": file_sha256(expanded_background),
    }
    expanded_calibration_value["background_dependence_audit"][
        "time_slide_report"
    ] = {
        "path": str(slides.resolve()),
        "sha256": file_sha256(slides),
    }
    expanded_calibration_value[
        "validation_time_slide_report_sha256"
    ] = file_sha256(slides)
    calibration.write_text(
        json.dumps(expanded_calibration_value),
        encoding="utf-8",
    )
    expanded_result = (
        bind_candidate_search_calibration_to_independent_endpoint(
            endpoint,
            pipeline,
            calibration,
            tmp_path / "expanded-binding.json",
            background_plan_authorization=authorization,
            expanded_background_merge_report=merge,
        )
    )
    assert expanded_result["passed"] is True
    assert expanded_result["expanded_background_lineage"][
        "background_plan_authorization"
    ]["authorization_id"] == "expanded-auth"

    tampered_merge_value = json.loads(merge.read_text(encoding="utf-8"))
    tampered_merge_value["common_run_identity"][
        "checkpoint_sha256"
    ] = "different"
    tampered_merge = tmp_path / "expanded-merge-tampered.json"
    tampered_merge.write_text(json.dumps(tampered_merge_value))
    with pytest.raises(ValueError, match="lineage failed replay"):
        bind_candidate_search_calibration_to_independent_endpoint(
            endpoint,
            pipeline,
            calibration,
            tmp_path / "expanded-binding-tampered.json",
            background_plan_authorization=authorization,
            expanded_background_merge_report=tampered_merge,
        )

    write_calibration(["background-block"])
    with pytest.raises(ValueError, match="purpose safe"):
        bind_candidate_search_calibration_to_independent_endpoint(
            endpoint,
            pipeline,
            calibration,
            tmp_path / "overlap-binding.json",
        )


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
  minimum_injection_gps_blocks: 25
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
    assert result["injection_bootstrap_independence"]["physical_groups"] == 100

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
    assert result["frozen_search"]["publication_calibration_eligible"] is False
    assert result["frozen_search"]["background_dependence_audit"]["passed"] is False
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


def test_candidate_pipeline_detector_set_recalibration_matches_locked_policy(
    tmp_path: Path,
) -> None:
    background_rows = []
    background_candidates = []
    provenance = {
        "timing_empirically_calibrated": True,
        "empirical_timing_uncertainty_seconds": 0.001,
        "timing_calibration_report_sha256": "a" * 64,
        "candidate_checkpoint_sha256": "b" * 64,
        "candidate_config_sha256": "c" * 64,
        "candidate_code_commit": "deadbee",
    }
    delays = {"H1": 0.001, "L1": 0.006, "V1": 0.021}
    for block_index in range(5):
        start = 1000.0 + block_index * 100.0
        window_id = f"w{block_index}"
        background_rows.append(
            {
                "window_id": window_id,
                "split": "val",
                "gps_start": start,
                "gps_end": start + 8.0,
                "gps_block": f"gps:{int(start)}:8",
                "ifos": ["H1", "L1", "V1"],
            }
        )
        for ifo, delay in delays.items():
            background_candidates.append(
                {
                    "candidate_id": f"b{block_index}-{ifo}",
                    "window_id": window_id,
                    "split": "val",
                    "ifo": ifo,
                    "gps_peak": start + delay,
                    "chirp_score": 0.8,
                    "glitch_score_at_peak": 0.1,
                    "bin_width_seconds": 0.005,
                    "timing_resolution_seconds": 0.005,
                    **provenance,
                }
            )
    background = tmp_path / "background.jsonl"
    background.write_text(
        "".join(json.dumps(row) + "\n" for row in background_rows),
        encoding="utf-8",
    )
    background_candidate_path = tmp_path / "background-candidates.jsonl"
    background_candidate_path.write_text(
        "".join(
            json.dumps(row) + "\n" for row in background_candidates
        ),
        encoding="utf-8",
    )

    injection_rows = []
    injection_candidates = []
    for index in range(25):
        gps = 5000.0 + index * 256.0
        injection_id = f"i{index}"
        injection_rows.append(
            {
                "injection_id": injection_id,
                "waveform_id": f"wave{index}",
                "split": "val",
                "source_family": "BBH",
                "stratum": "BBH",
                "gps_block": f"gps:{int(gps)}:256",
                "gps_time": gps,
                "vt_weight": 1.0,
                "vt_weight_unit": "relative",
                "valid_ifos": ["H1", "L1", "V1"],
                "detector_arrival_gps": {
                    ifo: gps + delay for ifo, delay in delays.items()
                },
            }
        )
        for ifo, delay in delays.items():
            injection_candidates.append(
                {
                    "candidate_id": f"{injection_id}-{ifo}",
                    "injection_id": injection_id,
                    "split": "val",
                    "ifo": ifo,
                    "gps_peak": gps + delay,
                    "chirp_score": 0.9,
                    "glitch_score_at_peak": 0.1,
                    **provenance,
                }
            )
    injection_triggers = tmp_path / "injection-triggers.jsonl"
    injection_triggers.write_text(
        "".join(json.dumps(row) + "\n" for row in injection_rows),
        encoding="utf-8",
    )
    injection_candidate_path = tmp_path / "injection-candidates.jsonl"
    injection_candidate_path.write_text(
        "".join(
            json.dumps(row) + "\n" for row in injection_candidates
        ),
        encoding="utf-8",
    )

    source = tmp_path / "source-pipeline.json"
    source.write_text(
        json.dumps(
            {
                "status": "validation_only_clustered_candidate_search_pipeline",
                "scientific_claim_allowed": False,
                "test_evaluation": None,
                "run_identity": {
                    "background_manifest_sha256": file_sha256(background),
                    "injection_manifest_sha256": "d" * 64,
                    "checkpoint_sha256": "b" * 64,
                    "config_sha256": "c" * 64,
                    "coherence_config_sha256": "e" * 64,
                    "model_ifos": ["H1", "L1", "V1"],
                    "q_values": [4, 8, 16],
                    "target_sample_rate": 1024,
                    "context_duration": 64.0,
                    "chirp_threshold": 0.3,
                    "minimum_bins": 1,
                    "code_commit": "deadbee",
                    "cluster_window_seconds": 0.1,
                    "target_far_per_year": 10_000_000.0,
                    "bootstrap_replicates": 20,
                    "seed": 1,
                },
                "time_slides": {
                    "candidate_manifest_sha256": file_sha256(
                        background_candidate_path
                    )
                },
                "injection_rankings": {
                    "injection_trigger_manifest_sha256": file_sha256(
                        injection_triggers
                    ),
                    "candidate_manifest_sha256": file_sha256(
                        injection_candidate_path
                    ),
                },
                "empirical_timing_uncertainty_seconds": 0.001,
                "timing_calibration_report_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    network_config = (
        Path(__file__).parents[1]
        / "configs"
        / "network_coherence_h1_l1_v1.yaml"
    )
    result = recalibrate_candidate_validation_pipeline_with_detector_sets(
        source,
        background,
        background_candidate_path,
        injection_triggers,
        injection_candidate_path,
        network_config,
        tmp_path / "detector-set",
        maximum_shifts=1,
    )
    assert result["time_slides"]["required_detector_subsets"] == [
        "H1+L1",
        "H1+V1",
        "L1+V1",
        "H1+L1+V1",
    ]
    assert result["injection_rankings"]["ranked_injections"] == 25
    assert result["frozen_search"]["selection_data"] == (
        "validation_variable_detector_set_block_permutations_only"
    )
    assert result["frozen_search"]["slide_schedule_audit"][
        "schedule_kind"
    ] == "variable_detector_set_block_permutation"
    assert result["frozen_search"]["background_dependence_audit"][
        "status"
    ] == "detector_set_candidate_background_dependence_audit_v1"
    assert result["frozen_search"]["identity"]["detector_set_policy"] == (
        "single_model_explicit_missing_ifo_validity_v1"
    )
    resumed = recalibrate_candidate_validation_pipeline_with_detector_sets(
        source,
        background,
        background_candidate_path,
        injection_triggers,
        injection_candidate_path,
        network_config,
        tmp_path / "detector-set",
        maximum_shifts=1,
    )
    assert resumed == result

    expanded_source_value = json.loads(source.read_text(encoding="utf-8"))
    expanded_source_value["run_identity"][
        "background_manifest_sha256"
    ] = "0" * 64
    expanded_source_value["time_slides"][
        "candidate_manifest_sha256"
    ] = "1" * 64
    expanded_source = tmp_path / "expanded-source-pipeline.json"
    expanded_source.write_text(
        json.dumps(expanded_source_value),
        encoding="utf-8",
    )
    parent = tmp_path / "parent-plan.json"
    parent.write_text(json.dumps({"status": "development_acquisition_plan"}))
    endpoint = tmp_path / "endpoint.json"
    endpoint.write_text(
        json.dumps(
            {
                "status": (
                    "frozen_gps_and_purpose_disjoint_validation_endpoint"
                ),
                "passed": True,
                "scientific_claim_allowed": False,
                "test_rows_read": 0,
                "test_evaluation": None,
                "purpose_gps_block_overlap": 0,
                "candidate_calibration_background_manifest_path": str(
                    background.resolve()
                ),
                "candidate_calibration_background_manifest_sha256": (
                    file_sha256(background)
                ),
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "status": (
                    "authorized_validation_candidate_continuous_background_plan"
                ),
                "authorization_id": "authorized-expanded",
                "passed": True,
                "scientific_claim_allowed": False,
                "candidate_scores_inspected": False,
                "test_rows_read": 0,
                "test_evaluation": None,
                "parent_plan": {
                    "path": str(parent.resolve()),
                    "sha256": file_sha256(parent),
                },
                "independent_validation_endpoint": {
                    "path": str(endpoint.resolve()),
                    "sha256": file_sha256(endpoint),
                },
                "authorization_identity": {
                    "target_far_per_year": 5_000_000.0,
                    "zero_count_confidence": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )
    common_identity = {
        key: expanded_source_value["run_identity"].get(key)
        for key in (
            "checkpoint_sha256",
            "config_sha256",
            "coherence_config_sha256",
            "model_ifos",
            "q_values",
            "target_sample_rate",
            "context_duration",
            "chirp_threshold",
            "minimum_bins",
            "code_commit",
        )
    }
    common_identity.update(
        {
            "timing_calibration_report_sha256": "a" * 64,
            "parent_plan_sha256": file_sha256(parent),
        }
    )
    merge = tmp_path / "merge.json"
    merge.write_text(
        json.dumps(
            {
                "status": "verified_merged_streamed_candidate_background",
                "scientific_claim_allowed": False,
                "complete_parent_plan": True,
                "split_counts": {"test": 0, "val": len(background_rows)},
                "common_run_identity": common_identity,
                "background_manifest_path": str(background.resolve()),
                "background_manifest_sha256": file_sha256(background),
                "candidate_manifests": {
                    "val": {
                        "path": str(background_candidate_path.resolve()),
                        "sha256": file_sha256(background_candidate_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    expanded = (
        recalibrate_candidate_validation_pipeline_with_detector_sets(
            expanded_source,
            background,
            background_candidate_path,
            injection_triggers,
            injection_candidate_path,
            network_config,
            tmp_path / "expanded-detector-set",
            maximum_shifts=1,
            expanded_background_merge_report=merge,
            background_plan_authorization=authorization,
        )
    )
    assert expanded["frozen_search"]["target_far_per_year"] == (
        5_000_000.0
    )
    assert expanded["run_identity"]["background_manifest_sha256"] == (
        file_sha256(background)
    )
    assert expanded["run_identity"]["target_far_per_year"] == 5_000_000.0
    assert expanded["detector_set_block_recalibration"][
        "expanded_background_lineage"
    ]["authorization_id"] == "authorized-expanded"

    tampered_merge_value = json.loads(merge.read_text(encoding="utf-8"))
    tampered_merge_value["common_run_identity"][
        "checkpoint_sha256"
    ] = "f" * 64
    tampered_merge = tmp_path / "tampered-merge.json"
    tampered_merge.write_text(json.dumps(tampered_merge_value))
    with pytest.raises(ValueError, match="lineage differs"):
        recalibrate_candidate_validation_pipeline_with_detector_sets(
            expanded_source,
            background,
            background_candidate_path,
            injection_triggers,
            injection_candidate_path,
            network_config,
            tmp_path / "tampered-expanded-detector-set",
            maximum_shifts=1,
            expanded_background_merge_report=tampered_merge,
            background_plan_authorization=authorization,
        )
