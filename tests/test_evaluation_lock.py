from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.evaluation_lock import (
    audit_locked_o4b_streaming_completion,
    download_locked_o4b_streaming_shard_sources,
    finalize_locked_o4b_streaming_shard,
    freeze_evaluation_corpus,
    freeze_locked_o4b_streaming_execution_plan,
    freeze_locked_evaluation_suite_plan,
    finalize_locked_evaluation_suite_receipt,
    merge_locked_o4b_streaming_shard_receipts,
    open_evaluation_corpus_once,
    prepare_locked_o4b_streaming_shard_manifests,
    reduce_locked_o4b_post_dq_injection_weights,
    validate_locked_evaluation_suite_access,
    validate_locked_evaluation_suite_input,
)
from gwyolo.io import file_sha256
from gwyolo.locked_streaming import (
    merge_locked_o4b_streaming_suite_input_sources,
    publish_locked_o4b_streaming_shard_artifacts,
)


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _complete_validation_evidence(path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "publication_evidence_ready",
                "publication_ready": True,
                "phase": "validation_freeze",
                "scientific_claim_allowed": False,
                "summary": {
                    "required_total": 3,
                    "required_passed": 3,
                    "required_pending": 0,
                    "required_failed": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def _pe_retention_inputs(tmp_path: Path) -> tuple[Path, Path]:
    common_prior = tmp_path / "common-prior.yaml"
    common_prior.write_text(
        "population: BBH\n"
        "distributions:\n"
        "  chirp_mass: {minimum: 1.0, maximum: 200.0}\n"
        "  mass_ratio: {minimum: 0.01, maximum: 1.0}\n"
        "  luminosity_distance: {minimum: 1.0, maximum: 10000.0}\n"
        "  theta_jn: {minimum: 0.0, maximum: 3.2}\n"
        "  ra: {minimum: 0.0, maximum: 6.3}\n"
        "  dec: {minimum: -1.6, maximum: 1.6}\n"
        "  psi: {minimum: 0.0, maximum: 3.2}\n",
        encoding="utf-8",
    )
    retention = tmp_path / "pe-retention.yaml"
    retention.write_text(
        "locked_pe_retention:\n"
        "  schema: locked_pe_retention_v1\n"
        "  population: BBH\n"
        f"  common_prior: {common_prior.name}\n"
        "  required_ifos: [H1, L1]\n"
        "  conditions: [clean, contaminated, mask_conditioned]\n"
        "  minimum_paired_injections: 1\n"
        "  retention_pool_injections: 2\n"
        "  minimum_gps_blocks: 1\n"
        "  selection_seed: 20260721\n"
        "  selection_method: gps_block_first_then_hash_rank_v1\n"
        "  post_access_replacement_allowed: false\n"
        "  score_dependent_selection_allowed: false\n",
        encoding="utf-8",
    )
    joint_manifest = tmp_path / "validation-pe.jsonl"
    _write(
        joint_manifest,
        [
            {
                "split": "val",
                "backend": "DINGO",
                "prior_hash": file_sha256(common_prior),
            }
        ],
    )
    joint_report = tmp_path / "validation-pe-joint.json"
    joint_report.write_text(
        json.dumps(
            {
                "status": "paired_dingo_amplfi_within_backend_portfolio_complete",
                "comparison_scope": "matched_event_within_backend_deltas_only",
                "matched_event_gate": True,
                "within_backend_provenance_gate": True,
                "absolute_cross_backend_comparison_allowed": False,
                "dingo_amplfi_joint_gate": False,
                "cross_backend_matched_input_gate": False,
                "publication_provenance_required": True,
                "manifest_path": str(joint_manifest.resolve()),
                "manifest_sha256": file_sha256(joint_manifest),
            }
        ),
        encoding="utf-8",
    )
    promotion_config = tmp_path / "promotion.yaml"
    promotion_config.write_text("pe_robustness_promotion: {}\n", encoding="utf-8")
    promotion = tmp_path / "validation-pe-promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "status": "pe_robustness_validation_promotion_decision",
                "passed": True,
                "promote_to_locked_test": True,
                "scientific_claim_allowed": False,
                "evidence_mode": "matched_event_within_backend_portfolio",
                "joint_report_path": str(joint_report.resolve()),
                "joint_report_sha256": file_sha256(joint_report),
                "joint_manifest_path": str(joint_manifest.resolve()),
                "joint_manifest_sha256": file_sha256(joint_manifest),
                "config_path": str(promotion_config.resolve()),
                "config_sha256": file_sha256(promotion_config),
            }
        ),
        encoding="utf-8",
    )
    return retention, promotion


def _locked_suite_config(path) -> None:
    outputs = {
        "raw_candidate_search": "search/raw.json",
        "mask_candidate_search": "search/mask.json",
        "paired_raw_mask_search": "search/paired.json",
        "locked_ood_transfer": "robustness/ood.json",
        "dingo_batch": "pe/dingo.json",
        "amplfi_batch": "pe/amplfi.json",
        "paired_pe_portfolio": "pe/portfolio.json",
        "catalog_diagnostic": "catalog/diagnostic.json",
        "suite_receipt": "suite.json",
    }
    inputs = {
        "raw_test_time_slide_report": "inputs/raw-slides.json",
        "mask_test_time_slide_report": "inputs/mask-slides.json",
        "raw_test_background_manifest": "inputs/raw-background.jsonl",
        "mask_test_background_manifest": "inputs/mask-background.jsonl",
        "raw_test_injection_ranking_report": "inputs/raw-rankings.json",
        "mask_test_injection_ranking_report": "inputs/mask-rankings.json",
        "locked_ood_score_manifest": "inputs/ood.jsonl",
        "locked_ood_score_report": "inputs/ood-report.json",
        "locked_ood_source_manifest": "inputs/ood-source.jsonl",
        "dingo_locked_source_batch_report": "inputs/dingo.json",
        "amplfi_locked_source_batch_report": "inputs/amplfi.json",
        "catalog_source_manifest": "inputs/catalog-source.jsonl",
        "catalog_candidate_manifest": "inputs/catalog-candidates.jsonl",
        "catalog_candidate_report": "inputs/catalog-candidate-report.json",
        "catalog_prediction_manifest": "inputs/catalog.jsonl",
        "catalog_prediction_report": "inputs/catalog-report.json",
    }
    path.write_text(
        "locked_evaluation_suite:\n"
        "  schema: locked_suite_v2\n"
        "  corpus_label: GWTC-5.0_O4b_locked_suite_v2\n"
        "  required_split: test\n"
        "  observing_runs: [O4b]\n"
        "  catalog_release: GWTC-5.0\n"
        "  required_frozen_artifacts:\n"
        "    - config\n"
        "    - model\n"
        "    - threshold_calibration\n"
        "    - ood_policy\n"
        "    - raw_candidate_calibration\n"
        "    - mask_candidate_calibration\n"
        "    - validation_raw_mask_comparison\n"
        "    - validation_ood_report\n"
        "    - validation_pe_promotion\n"
        "    - catalog_metadata\n"
        "    - locked_execution_plan\n"
        "  outputs:\n"
        + "".join(f"    {key}: {value}\n" for key, value in outputs.items())
        + "  inputs:\n"
        + "".join(f"    {key}: {value}\n" for key, value in inputs.items())
        + "  endpoints:\n"
        "    primary_search_metric: paired_delta_recovered_vt_at_common_far\n"
        "    threshold_policy: validation_frozen_no_test_retuning\n"
        "    target_far_per_year: 0.1\n"
        "    minimum_test_live_time_years: 23.02585093\n"
        "    minimum_test_injections: 3000\n"
        "    minimum_injection_gps_blocks: 25\n"
        "    minimum_paired_pe_injections: 1\n"
        "    minimum_locked_ood_rows: 500\n"
        "    minimum_background_gps_blocks: 25\n"
        "    minimum_background_shifts: 25\n"
        "    bootstrap_replicates: 10000\n"
        "    bootstrap_seed: 20260722\n"
        "    pe_credible_level: 0.9\n"
        "    uncertainty: gps_block_then_paired_injection_hierarchical_bootstrap_v1\n"
        "    background_dependence_uncertainty: physical_block_x_block_x_offset_pigeonhole_v1\n"
        "    catalog_search_arm: mask_candidate_search\n",
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


def test_freeze_locked_suite_and_validate_one_time_access_binding(tmp_path) -> None:
    evidence = tmp_path / "validation_evidence.json"
    config = tmp_path / "suite.yaml"
    output_root = tmp_path / "locked-results"
    plan_path = tmp_path / "locked-suite-plan.json"
    _complete_validation_evidence(evidence)
    _locked_suite_config(config)

    plan = freeze_locked_evaluation_suite_plan(
        evidence, config, output_root, "abc123", plan_path
    )
    assert plan["status"] == "frozen_locked_evaluation_suite_plan"
    assert plan["locked_corpus_opened"] is False
    assert plan["test_rows_read"] == 0
    assert plan["outputs"]["raw_candidate_search"] == str(
        (output_root / "search/raw.json").resolve()
    )

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
    access_path = tmp_path / "access.json"
    freeze_path = tmp_path / "corpus-freeze.json"
    freeze_evaluation_corpus(
        test_manifest,
        freeze_path,
        access_path,
        "GWTC-5.0_O4b_locked_suite_v2",
    )
    comparison = tmp_path / "train.jsonl"
    _write(
        comparison,
        [
            {
                "split": "train",
                "injection_id": "train-i0",
                "waveform_id": "train-w0",
                "gps_block": "train-g0",
            }
        ],
    )
    artifacts = {"locked_suite_plan": plan_path}
    for label in ("config", "model", "threshold_calibration", "ood_policy"):
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        artifacts[label] = path
    for label in (
        "raw_candidate_calibration",
        "mask_candidate_calibration",
        "validation_raw_mask_comparison",
        "validation_ood_report",
        "validation_pe_promotion",
        "catalog_metadata",
    ):
        path = tmp_path / label
        path.write_text(label, encoding="utf-8")
        artifacts[label] = path
    locked_execution = tmp_path / "locked_execution_plan"
    post_dq_manifest = tmp_path / "post-dq-weights.jsonl"
    post_dq_report = tmp_path / "post-dq-weights.json"
    locked_execution.write_text(
        json.dumps(
            {
                "status": "frozen_locked_o4b_streaming_execution_plan",
                "passed": True,
                "evaluation_opened": False,
                "candidate_scores_inspected": False,
                "test_strain_rows_read": 0,
                "code_commit": "abc123",
                "corpus_label": "GWTC-5.0_O4b_locked_suite_v2",
                "access_log_path": str(access_path.resolve()),
                "freeze_identity": {
                    "suite_plan_sha256": file_sha256(plan_path),
                    "corpus_freeze_sha256": file_sha256(freeze_path),
                },
                "post_dq_weight_manifest_path": str(post_dq_manifest.resolve()),
                "post_dq_weight_report_path": str(post_dq_report.resolve()),
            }
        ),
        encoding="utf-8",
    )
    artifacts["locked_execution_plan"] = locked_execution
    withheld_execution = artifacts.pop("locked_execution_plan")
    with pytest.raises(ValueError, match="complete frozen artifact inventory"):
        open_evaluation_corpus_once(
            freeze_path,
            "abc123",
            artifacts,
            (comparison,),
            plan["outputs"]["suite_receipt"],
            "python -m gwyolo.cli locked-suite-run ...",
        )
    assert not access_path.exists()
    artifacts["locked_execution_plan"] = withheld_execution
    open_evaluation_corpus_once(
        freeze_path,
        "abc123",
        artifacts,
        (comparison,),
        plan["outputs"]["suite_receipt"],
        "python -m gwyolo.cli locked-suite-run ...",
    )

    binding = validate_locked_evaluation_suite_access(
        plan_path,
        access_path,
        "raw_candidate_search",
        plan["outputs"]["raw_candidate_search"],
    )
    assert binding["code_commit"] == "abc123"
    assert binding["output_key"] == "raw_candidate_search"
    with pytest.raises(ValueError, match="not predeclared"):
        validate_locked_evaluation_suite_access(
            plan_path, access_path, "raw_candidate_search", tmp_path / "other.json"
        )

    expected_statuses = {
        "raw_candidate_search": "locked_candidate_search_evaluation",
        "mask_candidate_search": "locked_candidate_search_evaluation",
        "paired_raw_mask_search": "locked_paired_raw_mask_candidate_search_comparison",
        "locked_ood_transfer": "locked_detector_set_ood_transfer_evaluation",
        "dingo_batch": "locked_dingo_paired_pe_batch_complete",
        "amplfi_batch": "locked_amplfi_paired_pe_batch_complete",
        "paired_pe_portfolio": "locked_paired_pe_robustness_portfolio_complete",
        "catalog_diagnostic": "locked_gwtc5_catalog_diagnostic",
    }
    expected_inputs = {
        "raw_candidate_search": {
            "time_slide": "raw_test_time_slide_report",
            "background_manifest": "raw_test_background_manifest",
            "injection_ranking": "raw_test_injection_ranking_report",
        },
        "mask_candidate_search": {
            "time_slide": "mask_test_time_slide_report",
            "background_manifest": "mask_test_background_manifest",
            "injection_ranking": "mask_test_injection_ranking_report",
        },
        "locked_ood_transfer": {
            "source_manifest": "locked_ood_source_manifest",
            "score_manifest": "locked_ood_score_manifest",
            "score_report": "locked_ood_score_report",
        },
        "dingo_batch": {"single": "dingo_locked_source_batch_report"},
        "amplfi_batch": {"single": "amplfi_locked_source_batch_report"},
        "catalog_diagnostic": {
            "catalog_source_manifest": "catalog_source_manifest",
            "catalog_candidate_manifest": "catalog_candidate_manifest",
            "catalog_candidate_report": "catalog_candidate_report",
            "catalog_prediction_manifest": "catalog_prediction_manifest",
            "catalog_prediction_report": "catalog_prediction_report",
        },
    }
    for key, status in expected_statuses.items():
        path = Path(plan["outputs"][key])
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "status": status,
            "locked_suite_access": validate_locked_evaluation_suite_access(
                plan_path, access_path, key, path
            ),
        }
        if key in expected_inputs:
            inputs = {
                alias: validate_locked_evaluation_suite_input(
                    plan_path, input_key, plan["inputs"][input_key]
                )
                for alias, input_key in expected_inputs[key].items()
            }
            if "single" in inputs:
                value["locked_suite_input"] = inputs["single"]
            else:
                value["locked_suite_inputs"] = inputs
        path.write_text(
            json.dumps(value),
            encoding="utf-8",
        )
    streaming_completion = tmp_path / "streaming-completion.json"
    streaming_completion.write_text(
        json.dumps(
            {
                "status": "completed_locked_o4b_streaming_execution_audit",
                "passed": True,
                "all_predeclared_shards_reduced": True,
                "negative_and_null_results_retained": True,
                "result_dependent_stopping_used": False,
                "post_access_dq_replacement_used": False,
                "expected_shards": 1,
                "completed_shards": 1,
                "failed_shards": [],
                "rows": 1,
                "code_commit": "abc123",
                "execution_plan": {
                    "path": str(locked_execution.resolve()),
                    "sha256": file_sha256(locked_execution),
                },
                "access_log": {
                    "path": str(access_path.resolve()),
                    "sha256": file_sha256(access_path),
                },
            }
        ),
        encoding="utf-8",
    )
    incomplete_streaming = json.loads(streaming_completion.read_text(encoding="utf-8"))
    incomplete_streaming["completed_shards"] = 0
    streaming_completion.write_text(
        json.dumps(incomplete_streaming), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="complete all-shard streaming audit"):
        finalize_locked_evaluation_suite_receipt(
            plan_path,
            access_path,
            streaming_completion,
            plan["outputs"]["suite_receipt"],
        )
    incomplete_streaming["completed_shards"] = 1
    streaming_completion.write_text(
        json.dumps(incomplete_streaming), encoding="utf-8"
    )
    post_dq_manifest.write_text(
        json.dumps(
            {
                "injection_id": "test-i0",
                "eligible": True,
                "vt_weight": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    post_dq_report.write_text(
        json.dumps(
            {
                "status": "reduced_locked_o4b_post_dq_injection_weights",
                "passed": True,
                "candidate_scores_inspected": False,
                "raw_mask_shared_physical_denominator": True,
                "post_access_dq_replacement_used": False,
                "result_dependent_stopping_used": False,
                "planned_injections": 1,
                "eligible_injections": 1,
                "unavailable_injections": 0,
                "background_live_time_years": 1.0,
                "weight_manifest_path": str(post_dq_manifest.resolve()),
                "weight_manifest_sha256": file_sha256(post_dq_manifest),
                "streaming_completion_audit": {
                    "path": str(streaming_completion.resolve()),
                    "sha256": file_sha256(streaming_completion),
                },
                "code_commit": "abc123",
            }
        ),
        encoding="utf-8",
    )
    receipt = finalize_locked_evaluation_suite_receipt(
        plan_path,
        access_path,
        streaming_completion,
        plan["outputs"]["suite_receipt"],
    )
    assert receipt["passed"] is True
    assert receipt["all_predeclared_outputs_present"] is True
    assert len(receipt["outputs"]) == 8
    assert receipt["streaming_completion_audit"]["completed_shards"] == 1
    assert receipt["post_dq_injection_weights"]["eligible_injections"] == 1


def test_freeze_locked_o4b_streaming_plan_binds_every_source_before_access(
    tmp_path,
    monkeypatch,
) -> None:
    evidence = tmp_path / "validation_evidence.json"
    config = tmp_path / "suite.yaml"
    suite_root = tmp_path / "locked-results"
    suite_plan = tmp_path / "locked-suite-plan.json"
    _complete_validation_evidence(evidence)
    _locked_suite_config(config)
    freeze_locked_evaluation_suite_plan(
        evidence, config, suite_root, "abc123", suite_plan
    )

    access = tmp_path / "access.json"
    availability_manifest = tmp_path / "availability.jsonl"
    availability_rows = []
    injection_rows = []
    for index, gps_start in enumerate((1400000000, 1400004096)):
        availability_id = f"availability-{index}"
        gps_block = f"O4b:{gps_start}:4096"
        sources = {
            ifo: {
                "detector": ifo,
                "gps_start": gps_start,
                "duration": 4096,
                "hdf5_url": f"https://gwosc.org/archive/{ifo}-{gps_start}-4096.hdf5",
                "detail_url": f"https://gwosc.org/archive/{ifo}-{gps_start}-4096.json",
            }
            for ifo in ("H1", "L1")
        }
        availability_rows.append(
            {
                "availability_id": availability_id,
                "gps_block": gps_block,
                "available_ifos": ["H1", "L1"],
                "sources": sources,
            }
        )
        injection_rows.append(
            {
                "availability_id": availability_id,
                "injection_id": f"injection-{index}",
                "waveform_id": f"waveform-{index}",
                "gps_block": gps_block,
                "split": "test",
                "observing_run": "O4b",
                "ifos": ["H1", "L1"],
                "gps_time": gps_start + 2048.0,
                "required_context_duration_seconds": 256.0,
                "source_family": "BBH",
                "mass_1_detector_msun": 40.0,
                "mass_2_detector_msun": 30.0,
                "luminosity_distance_mpc": 1000.0,
                "inclination": 1.0,
                "right_ascension": 2.0,
                "declination": 0.2,
                "polarization": 0.5,
                "proposal_family_fraction": 1.0,
                "proposal_comoving_volume_mpc3": 100.0,
                "source_frame_time_factor": 0.5,
            }
        )
    _write(availability_manifest, availability_rows)
    inventory_manifest = tmp_path / "inventory.jsonl"
    _write(inventory_manifest, injection_rows)

    availability_report = tmp_path / "availability-report.json"
    availability_report.write_text(
        json.dumps(
            {
                "status": "score_blind_gwtc5_o4b_availability_inventory",
                "passed": True,
                "manifest_path": str(availability_manifest.resolve()),
                "manifest_sha256": file_sha256(availability_manifest),
                "access_log_path": str(access.resolve()),
                "candidate_scores_inspected": False,
                "test_strain_rows_read": 0,
            }
        ),
        encoding="utf-8",
    )
    inventory_report = tmp_path / "inventory-report.json"
    inventory_report.write_text(
        json.dumps(
            {
                "status": "score_blind_gwtc5_locked_injection_inventory",
                "passed": True,
                "manifest_path": str(inventory_manifest.resolve()),
                "manifest_sha256": file_sha256(inventory_manifest),
                "availability_manifest_sha256": file_sha256(availability_manifest),
                "access_log_path": str(access.resolve()),
                "post_access_dq_replacement_allowed": False,
                "candidate_scores_inspected": False,
                "test_strain_rows_read": 0,
            }
        ),
        encoding="utf-8",
    )
    corpus_freeze = tmp_path / "corpus-freeze.json"
    corpus_freeze.write_text(
        json.dumps(
            {
                "status": "locked_evaluation_corpus_unopened",
                "evaluation_opened": False,
                "candidate_scores_inspected": False,
                "corpus_label": "GWTC-5.0_O4b_locked_suite_v2",
                "manifest_path": str(inventory_manifest.resolve()),
                "manifest_sha256": file_sha256(inventory_manifest),
                "access_log_path": str(access.resolve()),
            }
        ),
        encoding="utf-8",
    )
    shard_manifest = tmp_path / "streaming-shards.jsonl"
    report_path = tmp_path / "streaming-plan.json"
    pe_retention_config, validation_pe_promotion = _pe_retention_inputs(tmp_path)
    result = freeze_locked_o4b_streaming_execution_plan(
        suite_plan,
        corpus_freeze,
        availability_manifest,
        availability_report,
        inventory_manifest,
        inventory_report,
        pe_retention_config,
        validation_pe_promotion,
        suite_root / "execution",
        shard_manifest,
        report_path,
        "abc123",
        blocks_per_shard=1,
        minimum_free_kb=1024 * 1024,
    )
    replay = freeze_locked_o4b_streaming_execution_plan(
        suite_plan,
        corpus_freeze,
        availability_manifest,
        availability_report,
        inventory_manifest,
        inventory_report,
        pe_retention_config,
        validation_pe_promotion,
        suite_root / "execution",
        shard_manifest,
        report_path,
        "abc123",
        blocks_per_shard=1,
        minimum_free_kb=1024 * 1024,
    )

    assert replay == result
    assert result["rows"] == 2
    assert result["shards"] == 2
    assert result["maximum_concurrent_shards"] == 1
    assert result["post_access_dq_replacement_allowed"] is False
    rows = [json.loads(line) for line in shard_manifest.read_text().splitlines()]
    assert rows[0]["injection_ids"] == ["injection-0"]
    assert len(rows[0]["source_files"]) == 2
    assert rows[1]["gps_blocks"] == ["O4b:1400004096:4096"]
    with pytest.raises(FileNotFoundError, match="plan/access"):
        download_locked_o4b_streaming_shard_sources(
            report_path, access, 0, "abc123"
        )
    access.write_text(
        json.dumps(
            {
                "status": "locked_evaluation_corpus_opened_once",
                "evaluation_opened": True,
                "corpus_label": result["corpus_label"],
                "code_commit": "abc123",
                "frozen_artifacts": {
                    "locked_suite_plan": {
                        "path": str(suite_plan.resolve()),
                        "sha256": file_sha256(suite_plan),
                    },
                    "locked_execution_plan": {
                        "path": str(report_path.resolve()),
                        "sha256": file_sha256(report_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_download(_url, destination, workers):
        assert workers == 2
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"locked-source")
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "downloaded": True,
        }

    monkeypatch.setattr("gwyolo.gwosc.download_resumable", fake_download)
    monkeypatch.setattr("gwyolo.gwosc._api_json", lambda _url: {})
    monkeypatch.setattr(
        "gwyolo.gwosc.verify_hdf5_against_detail",
        lambda _path, _detail, _chunks: {"passed": True},
    )
    with pytest.raises(ValueError, match="access/plan binding"):
        download_locked_o4b_streaming_shard_sources(
            report_path,
            access,
            0,
            "wrong-commit",
            download_workers=2,
        )
    source_result = download_locked_o4b_streaming_shard_sources(
        report_path,
        access,
        0,
        "abc123",
        download_workers=2,
    )
    assert source_result["passed"] is True
    assert source_result["verified_files"] == 2
    assert (
        download_locked_o4b_streaming_shard_sources(
            report_path,
            access,
            0,
            "abc123",
            download_workers=2,
        )
        == source_result
    )

    def fake_quality(path):
        path = Path(path)
        second = "shard-00001" in path.parts
        gps_start = 1400004096 if second else 1400000000
        return {
            "gps_start": gps_start,
            "gps_end": gps_start + 4096,
            "duration": 4096,
            "dqmask": np.ones(4096, dtype=np.int64),
            "injmask": np.full(
                4096,
                0 if second else 31,
                dtype=np.int64,
            ),
        }

    monkeypatch.setattr("gwyolo.background._read_quality", fake_quality)
    prepared = prepare_locked_o4b_streaming_shard_manifests(
        report_path,
        access,
        0,
        "abc123",
    )
    assert prepared["background_windows"] > 0
    assert prepared["eligible_injections"] == 1
    assert prepared["unavailable_injections"] == 0
    active_lease = suite_root / "execution" / ".active-shard.lock"
    active_lease.write_text("active", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already active"):
        download_locked_o4b_streaming_shard_sources(
            report_path,
            access,
            1,
            "abc123",
            download_workers=2,
        )
    active_lease.unlink()
    receipts = []
    for shard in rows:
        if not Path(shard["source_download_report_path"]).exists():
            download_locked_o4b_streaming_shard_sources(
                report_path,
                access,
                shard["shard_index"],
                "abc123",
                download_workers=2,
            )
        if not Path(shard["manifest_preparation_report_path"]).exists():
            shard_prepared = prepare_locked_o4b_streaming_shard_manifests(
                report_path,
                access,
                shard["shard_index"],
                "abc123",
            )
            assert shard_prepared["unavailable_injections"] == 1
            assert shard_prepared["post_access_dq_replacement_used"] is False
        work_dir = Path(shard["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        source_inputs = [
            work_dir / f"{label}.jsonl"
            for label in (
                "raw-background-candidates",
                "raw-injection-candidates",
                "mask-background-candidates",
                "mask-injection-candidates",
                "ood-source",
                "pe-input",
            )
        ]
        candidate_payloads = [[], [], [], [], [], []]
        if shard["shard_index"] == 0:
            background_row = json.loads(
                Path(shard["background_manifest_path"])
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            common_background = {
                "window_id": background_row["window_id"],
                "gps_block": background_row["gps_block"],
                "ifo": "H1",
                "split": "test",
                "gps_peak": float(background_row["gps_start"]) + 1.0,
            }
            common_injection = {
                "injection_id": "injection-0",
                "waveform_id": "waveform-0",
                "gps_block": shard["gps_blocks"][0],
                "ifo": "H1",
                "split": "test",
                "gps_peak": 1400002048.0,
            }
            candidate_payloads[0] = [
                {
                    **common_background,
                    "candidate_id": f"raw-bg-{index}",
                    "chirp_score": 0.4 + index / 10,
                }
                for index in range(2)
            ]
            candidate_payloads[1] = [
                {
                    **common_injection,
                    "candidate_id": f"raw-inj-{index}",
                    "chirp_score": 0.6 + index / 10,
                }
                for index in range(2)
            ]
            candidate_payloads[2] = [
                {
                    **common_background,
                    "candidate_id": "mask-bg-0",
                    "chirp_score": 0.5,
                }
            ]
            candidate_payloads[3] = [
                {
                    **common_injection,
                    "candidate_id": "mask-inj-0",
                    "chirp_score": 0.8,
                }
            ]
            pe_payload = []
            for condition in ("clean", "contaminated", "mask_conditioned"):
                array_path = work_dir / f"pe-{condition}.npz"
                np.savez_compressed(
                    array_path,
                    strain=np.ones((2, 64), dtype=np.float32),
                    asd=np.ones((2, 33), dtype=np.float64),
                    asd_frequencies=np.arange(33, dtype=np.float64),
                    ifos=np.asarray(["H1", "L1"]),
                    sample_rate=np.asarray(4, dtype=np.int64),
                    condition=np.asarray(condition),
                    injection_id=np.asarray("injection-0"),
                )
                pe_payload.append(
                    {
                        "injection_id": "injection-0",
                        "waveform_id": "waveform-0",
                        "gps_block": shard["gps_blocks"][0],
                        "split": "test",
                        "condition": condition,
                        "input_ifos": ["H1", "L1"],
                        "analysis_input_path": str(array_path.resolve()),
                        "analysis_input_sha256": file_sha256(array_path),
                    }
                )
            candidate_payloads[5] = pe_payload
        for artifact, payload in zip(source_inputs, candidate_payloads):
            _write(artifact, payload)
        if shard["shard_index"] == 0:
            _write(source_inputs[5], candidate_payloads[5][:-1])
            with pytest.raises(
                ValueError, match="source eviction is forbidden"
            ):
                publish_locked_o4b_streaming_shard_artifacts(
                    report_path,
                    access,
                    shard["shard_index"],
                    *source_inputs,
                    "abc123",
                )
            _write(source_inputs[5], candidate_payloads[5])
        published = publish_locked_o4b_streaming_shard_artifacts(
            report_path,
            access,
            shard["shard_index"],
            *source_inputs,
            "abc123",
        )
        assert published["passed"] is True
        assert published["all_candidate_instances_retained"] is True
        if shard["shard_index"] == 0:
            assert published["row_counts"]["raw_background_candidates"] == 2
            assert published["row_counts"]["raw_injection_candidates"] == 2
            assert published["row_counts"]["pe_retained_injections"] == 1
        receipts.append(
            finalize_locked_o4b_streaming_shard(
                report_path,
                access,
                shard["shard_index"],
                "abc123",
            )
        )
        if shard["shard_index"] == 0:
            with pytest.raises(FileNotFoundError, match="receipt is absent"):
                merge_locked_o4b_streaming_shard_receipts(
                    report_path, access, "abc123"
                )
    merged = merge_locked_o4b_streaming_shard_receipts(
        report_path, access, "abc123"
    )
    assert merged["completed_shards"] == 2
    receipt_manifest = Path(result["receipt_manifest_path"])
    completion = audit_locked_o4b_streaming_completion(
        report_path,
        access,
        receipt_manifest,
        result["completion_audit_path"],
        "abc123",
    )
    assert completion["passed"] is True
    assert completion["completed_shards"] == 2
    assert completion["unique_injections"] == 2
    assert len(completion["artifact_inventory"]["raw_candidate_rows"]) == 2
    weights = reduce_locked_o4b_post_dq_injection_weights(
        report_path,
        access,
        result["completion_audit_path"],
        "abc123",
    )
    assert weights["planned_injections"] == 2
    assert weights["eligible_injections"] == 1
    assert weights["unavailable_injections"] == 1
    assert weights["raw_mask_shared_physical_denominator"] is True
    weight_rows = [
        json.loads(line)
        for line in Path(weights["weight_manifest_path"]).read_text().splitlines()
    ]
    eligible_weight = next(row for row in weight_rows if row["eligible"])
    null_weight = next(row for row in weight_rows if not row["eligible"])
    assert eligible_weight["vt_weight"] == pytest.approx(
        100.0 * weights["background_live_time_years"] * 0.5
    )
    assert null_weight["vt_weight"] is None
    assert null_weight["vt_measure"] == "unavailable_post_dq_null"
    suite_inputs = merge_locked_o4b_streaming_suite_input_sources(
        suite_plan,
        report_path,
        access,
        result["completion_audit_path"],
        result["post_dq_weight_report_path"],
        "abc123",
    )
    assert suite_inputs["passed"] is True
    assert suite_inputs["background_windows"] == weights["background_windows"]
    assert suite_inputs["eligible_injections"] == 1
    assert suite_inputs["unavailable_injections"] == 1
    assert suite_inputs["ood_sources"] == 0
    assert suite_inputs["raw_background_candidates"] == 2
    assert suite_inputs["raw_injection_candidates"] == 2
    assert suite_inputs["mask_background_candidates"] == 1
    assert suite_inputs["mask_injection_candidates"] == 1
    merged_injections = [
        json.loads(line)
        for line in Path(
            suite_inputs["artifacts"]["raw_injection_candidates"]["path"]
        )
        .read_text()
        .splitlines()
    ]
    assert len(merged_injections) == 2
    assert all(
        row["vt_weight"] == pytest.approx(eligible_weight["vt_weight"])
        for row in merged_injections
    )
    assert Path(
        suite_inputs["artifacts"]["raw_background_manifest"]["path"]
    ).read_text() == Path(
        suite_inputs["artifacts"]["mask_background_manifest"]["path"]
    ).read_text()

    with pytest.raises(ValueError, match="unopened execution boundary"):
        freeze_locked_o4b_streaming_execution_plan(
            suite_plan,
            corpus_freeze,
            availability_manifest,
            availability_report,
            inventory_manifest,
            inventory_report,
            pe_retention_config,
            validation_pe_promotion,
            suite_root / "execution",
            shard_manifest,
            report_path,
            "abc123",
        )
def test_freeze_locked_suite_rejects_incomplete_validation_and_unsafe_output(
    tmp_path,
) -> None:
    evidence = tmp_path / "validation_evidence.json"
    config = tmp_path / "suite.yaml"
    _complete_validation_evidence(evidence)
    _locked_suite_config(config)
    report = json.loads(evidence.read_text(encoding="utf-8"))
    report["publication_ready"] = False
    evidence.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="complete validation-freeze"):
        freeze_locked_evaluation_suite_plan(
            evidence, config, tmp_path / "results", "abc123", tmp_path / "plan.json"
        )

    _complete_validation_evidence(evidence)
    text = config.read_text(encoding="utf-8").replace(
        "search/raw.json", "../escaped.json"
    )
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative path"):
        freeze_locked_evaluation_suite_plan(
            evidence, config, tmp_path / "results", "abc123", tmp_path / "plan.json"
        )
