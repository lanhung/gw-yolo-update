from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.evaluation_lock import (
    freeze_evaluation_corpus,
    freeze_locked_evaluation_suite_plan,
    finalize_locked_evaluation_suite_receipt,
    open_evaluation_corpus_once,
    validate_locked_evaluation_suite_access,
    validate_locked_evaluation_suite_input,
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


def _locked_suite_config(path) -> None:
    outputs = {
        "raw_candidate_search": "search/raw.json",
        "mask_candidate_search": "search/mask.json",
        "paired_raw_mask_search": "search/paired.json",
        "locked_ood_transfer": "robustness/ood.json",
        "dingo_batch": "pe/dingo.json",
        "amplfi_batch": "pe/amplfi.json",
        "joint_pe": "pe/joint.json",
        "catalog_diagnostic": "catalog/diagnostic.json",
        "suite_receipt": "suite.json",
    }
    inputs = {
        "raw_test_time_slide_report": "inputs/raw-slides.json",
        "mask_test_time_slide_report": "inputs/mask-slides.json",
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
        "  schema: locked_suite_v1\n"
        "  corpus_label: GWTC-5.0_O4b_locked_suite_v1\n"
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
        "    minimum_paired_pe_injections: 100\n"
        "    minimum_locked_ood_rows: 500\n"
        "    bootstrap_replicates: 10000\n"
        "    bootstrap_seed: 20260722\n"
        "    pe_credible_level: 0.9\n"
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
        "GWTC-5.0_O4b_locked_suite_v1",
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
        "joint_pe": "locked_joint_paired_pe_complete",
        "catalog_diagnostic": "locked_gwtc5_catalog_diagnostic",
    }
    expected_inputs = {
        "raw_candidate_search": {
            "time_slide": "raw_test_time_slide_report",
            "injection_ranking": "raw_test_injection_ranking_report",
        },
        "mask_candidate_search": {
            "time_slide": "mask_test_time_slide_report",
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
    receipt = finalize_locked_evaluation_suite_receipt(
        plan_path, access_path, plan["outputs"]["suite_receipt"]
    )
    assert receipt["passed"] is True
    assert receipt["all_predeclared_outputs_present"] is True
    assert len(receipt["outputs"]) == 8


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
