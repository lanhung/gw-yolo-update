from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.cli import main
from gwyolo.io import file_sha256
from gwyolo.publication import (
    run_publication_evidence_audit,
    run_publication_result_registry,
)


def _write_protocol(path: Path, phase: str = "validation_freeze") -> None:
    path.write_text(
        f"""
publication_evidence:
  schema: publication_evidence_v1
  protocol: hand_calculated_protocol
  phase: {phase}
  groups: [data, search]
  requirements:
    - id: data_gate
      group: data
      description: hand-calculated data gate
      checks:
        - {{field: passed, op: equals, value: true}}
        - {{field: rows, op: at_least, value: 4}}
        - {{field: seeds, op: length_at_least, value: 2}}
        - {{field: overlaps, op: all_empty}}
      replay_artifacts:
        - {{path_field: manifest.path, sha256_field: manifest.sha256}}
    - id: search_gate
      group: search
      checks:
        - {{field: exposure_years, op: greater_than, value: 1}}
""".lstrip(),
        encoding="utf-8",
    )


def _write_data_evidence(path: Path, manifest: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "rows": 4,
                "seeds": [11, 12],
                "overlaps": {"train:val": []},
                "manifest": {"path": str(manifest), "sha256": file_sha256(manifest)},
            }
        ),
        encoding="utf-8",
    )


def test_publication_evidence_audit_counts_only_replayed_passing_gates(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.yaml"
    manifest = tmp_path / "manifest.jsonl"
    evidence = tmp_path / "data.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    _write_protocol(protocol)
    manifest.write_text('{"split":"train"}\n', encoding="utf-8")
    _write_data_evidence(evidence, manifest)

    report = run_publication_evidence_audit(
        protocol,
        [f"data_gate={evidence}"],
        output,
        markdown,
    )

    assert report["publication_ready"] is False
    assert report["scientific_claim_allowed"] is False
    assert report["summary"] == {
        "required_total": 2,
        "required_passed": 1,
        "required_pending": 1,
        "required_failed": 0,
        "completion_percent": 50.0,
    }
    assert report["groups"]["data"]["required_passed"] == 1
    assert report["requirements"][0]["artifact_replay"][0]["passed"] is True
    assert report["requirements"][1]["state"] == "pending"
    assert "Required gates passed: **1/2**" in markdown.read_text(encoding="utf-8")


def test_publication_evidence_audit_fails_changed_artifact_and_require_ready(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.yaml"
    manifest = tmp_path / "manifest.jsonl"
    evidence = tmp_path / "data.json"
    search = tmp_path / "search.json"
    output = tmp_path / "audit.json"
    _write_protocol(protocol)
    manifest.write_text("original\n", encoding="utf-8")
    _write_data_evidence(evidence, manifest)
    manifest.write_text("changed\n", encoding="utf-8")
    search.write_text(json.dumps({"exposure_years": 2.0}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="1/2 gates"):
        run_publication_evidence_audit(
            protocol,
            [f"data_gate={evidence}", f"search_gate={search}"],
            output,
            require_ready=True,
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["required_failed"] == 1
    assert report["requirements"][0]["artifact_replay"][0]["passed"] is False


def test_publication_evidence_cli_marks_locked_package_complete_without_authorizing_claim(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.yaml"
    manifest = tmp_path / "manifest.jsonl"
    evidence = tmp_path / "data.json"
    search = tmp_path / "search.json"
    output = tmp_path / "audit.json"
    _write_protocol(protocol, "locked_final")
    manifest.write_text("frozen\n", encoding="utf-8")
    _write_data_evidence(evidence, manifest)
    search.write_text(json.dumps({"exposure_years": 2.0}), encoding="utf-8")

    assert (
        main(
            [
                "publication-evidence-audit",
                "--config",
                str(protocol),
                "--evidence",
                f"data_gate={evidence}",
                "--evidence",
                f"search_gate={search}",
                "--output",
                str(output),
                "--require-ready",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["publication_ready"] is True
    assert report["locked_final_evidence_complete"] is True
    assert report["scientific_claim_allowed"] is False


def _write_registry_ledgers(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "registry-manifest.jsonl"
    manifest.write_text('{"split":"validation"}\n', encoding="utf-8")
    data_evidence = tmp_path / "registry-data.json"
    _write_data_evidence(data_evidence, manifest)

    validation_protocol = tmp_path / "registry-validation.yaml"
    _write_protocol(validation_protocol)
    validation_ledger = tmp_path / "registry-validation-ledger.json"
    run_publication_evidence_audit(
        validation_protocol,
        [f"data_gate={data_evidence}"],
        validation_ledger,
    )

    locked_protocol = tmp_path / "registry-locked.yaml"
    _write_protocol(locked_protocol, "locked_final")
    locked_search = tmp_path / "registry-locked-search.json"
    locked_search.write_text(
        json.dumps(
            {
                "exposure_years": 2.0,
                "endpoint_complete": True,
                "primary_endpoint_result": {
                    "metric": "paired_delta_recovered_vt_at_common_far",
                    "observed_absolute_weighted_efficiency_gain": 0.01,
                    "significant_mask_advantage": False,
                },
            }
        ),
        encoding="utf-8",
    )
    locked_ledger = tmp_path / "registry-locked-ledger.json"
    run_publication_evidence_audit(
        locked_protocol,
        [f"data_gate={data_evidence}", f"search_gate={locked_search}"],
        locked_ledger,
    )
    return validation_ledger, locked_ledger, manifest


def test_publication_result_registry_replays_and_retains_hand_calculated_null_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "d" * 40)
    validation_ledger, locked_ledger, _ = _write_registry_ledgers(tmp_path)
    output = tmp_path / "registry.json"
    csv_output = tmp_path / "registry.csv"
    markdown = tmp_path / "registry.md"

    assert (
        main(
            [
                "publication-result-registry",
                "--ledger",
                str(validation_ledger),
                "--ledger",
                str(locked_ledger),
                "--output",
                str(output),
                "--csv",
                str(csv_output),
                "--markdown",
                str(markdown),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"] == {
        "ledger_count": 2,
        "gate_total": 4,
        "passed": 3,
        "failed": 0,
        "pending": 1,
        "skipped": 0,
        "negative_or_null_outcomes": 1,
        "locked_final_registry_present": True,
        "locked_final_ready": True,
    }
    assert report["requirements_omitted"] == 0
    assert report["negative_and_null_results_retained"] is True
    assert report["scientific_claim_allowed"] is False
    locked_search = next(
        row
        for row in report["rows"]
        if row["phase"] == "locked_final" and row["gate_id"] == "search_gate"
    )
    assert (
        locked_search["result_class"]
        == "gate_passed_null_or_negative_primary_endpoint"
    )
    assert (
        locked_search["outcome"]["primary_endpoint_result"][
            "significant_mask_advantage"
        ]
        is False
    )
    assert "gate_passed_null_or_negative_primary_endpoint" in csv_output.read_text(
        encoding="utf-8"
    )
    assert "Registered gates: **4**" in markdown.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        run_publication_result_registry(
            [validation_ledger, locked_ledger], output, csv_output, markdown
        )


def test_publication_result_registry_fails_closed_on_tamper_or_duplicate_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "e" * 40)
    validation_ledger, locked_ledger, manifest = _write_registry_ledgers(tmp_path)
    manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger replay differs for gate: data_gate"):
        run_publication_result_registry(
            [validation_ledger, locked_ledger],
            tmp_path / "tampered.json",
            tmp_path / "tampered.csv",
            tmp_path / "tampered.md",
        )

    validation_ledger, _, _ = _write_registry_ledgers(tmp_path / "duplicate")
    with pytest.raises(ValueError, match="duplicate phase: validation_freeze"):
        run_publication_result_registry(
            [validation_ledger, validation_ledger],
            tmp_path / "duplicate.json",
            tmp_path / "duplicate.csv",
            tmp_path / "duplicate.md",
        )


def test_publication_result_registry_requires_full_commit_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_ledger, _, _ = _write_registry_ledgers(tmp_path)
    monkeypatch.delenv("GWYOLO_CODE_COMMIT", raising=False)
    with pytest.raises(ValueError, match="full GWYOLO_CODE_COMMIT"):
        run_publication_result_registry(
            [validation_ledger],
            tmp_path / "missing-commit.json",
            tmp_path / "missing-commit.csv",
            tmp_path / "missing-commit.md",
        )


@pytest.mark.parametrize(
    "binding",
    ["undeclared=/tmp/report.json", "data_gate=/tmp/a.json"],
)
def test_publication_evidence_rejects_unknown_or_duplicate_binding(
    tmp_path: Path, binding: str
) -> None:
    protocol = tmp_path / "protocol.yaml"
    _write_protocol(protocol)
    bindings = [binding]
    if binding.startswith("data_gate"):
        bindings.append(binding)
    with pytest.raises(ValueError):
        run_publication_evidence_audit(protocol, bindings, tmp_path / "audit.json")


def test_official_validation_protocol_rejects_undersized_independent_endpoint(
    tmp_path: Path,
) -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "publication_validation_evidence.yaml"
    )
    component_reports = {}
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
        component_reports[label] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    calibration = tmp_path / "candidate-calibration.jsonl"
    arrivals = tmp_path / "arrivals.jsonl"
    calibration.write_text('{"gps_block":"g1"}\n', encoding="utf-8")
    arrivals.write_text('{"injection_id":"i1"}\n', encoding="utf-8")
    evidence = tmp_path / "endpoint.json"

    def write_endpoint(rows: int, calibration_blocks: int, injection_blocks: int) -> None:
        evidence.write_text(
            json.dumps(
                {
                    "status": "frozen_gps_and_purpose_disjoint_validation_endpoint",
                    "passed": True,
                    "rows": rows,
                    "candidate_calibration_unique_gps_blocks": calibration_blocks,
                    "injection_validation_unique_gps_blocks": injection_blocks,
                    "purpose_gps_block_overlap": 0,
                    "test_rows_read": 0,
                    "test_evaluation": None,
                    "scientific_claim_allowed": False,
                    "candidate_calibration_background_manifest_path": str(calibration),
                    "candidate_calibration_background_manifest_sha256": file_sha256(
                        calibration
                    ),
                    "injection_arrival_manifest_path": str(arrivals),
                    "injection_arrival_manifest_sha256": file_sha256(arrivals),
                    "component_reports": component_reports,
                }
            ),
            encoding="utf-8",
        )

    write_endpoint(rows=2, calibration_blocks=1, injection_blocks=1)
    failed = run_publication_evidence_audit(
        protocol,
        [f"independent_validation_endpoint={evidence}"],
        tmp_path / "failed-audit.json",
    )
    gate = next(
        row
        for row in failed["requirements"]
        if row["id"] == "independent_validation_endpoint"
    )
    assert gate["state"] == "failed"
    assert all(item["passed"] for item in gate["artifact_replay"])
    assert {
        row["field"] for row in gate["checks"] if row["passed"] is False
    } == {
        "rows",
        "candidate_calibration_unique_gps_blocks",
        "injection_validation_unique_gps_blocks",
    }

    write_endpoint(rows=3000, calibration_blocks=25, injection_blocks=25)
    passed = run_publication_evidence_audit(
        protocol,
        [f"independent_validation_endpoint={evidence}"],
        tmp_path / "passed-audit.json",
    )
    gate = next(
        row
        for row in passed["requirements"]
        if row["id"] == "independent_validation_endpoint"
    )
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 8


def test_official_validation_protocol_requires_automatic_bound_raw_mask_receipt(
    tmp_path: Path,
) -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "publication_validation_evidence.yaml"
    )
    evidence = tmp_path / "raw-mask.json"
    evidence.write_text(
        json.dumps(
            {
                "status": (
                    "validation_only_paired_raw_mask_candidate_calibration_comparison"
                ),
                "passed": True,
                "mask_locked_test_arm_eligible": True,
                "locked_test_prerequisites_satisfied": False,
                "test_rows_read": 0,
                "scientific_claim_allowed": False,
                "code_commit": "old",
            }
        ),
        encoding="utf-8",
    )
    failed = run_publication_evidence_audit(
        protocol,
        [f"paired_raw_mask_vt={evidence}"],
        tmp_path / "failed-raw-mask-audit.json",
    )
    gate = next(
        row for row in failed["requirements"] if row["id"] == "paired_raw_mask_vt"
    )
    assert gate["state"] == "failed"

    artifacts = {}
    for label in (
        "raw_mask_endpoint",
        "automatic_mask_audit",
        "automatic_mask_manifest",
        "overlap_config",
        "gate_config",
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        artifacts[label] = {"path": str(path), "sha256": file_sha256(path)}
    evidence.write_text(
        json.dumps(
            {
                "status": "bound_validation_raw_mask_automatic_evidence",
                "passed": True,
                "mask_locked_test_arm_eligible": True,
                "functional_raw_mask_endpoint_passed": True,
                "automatic_mask_policy_passed": True,
                "human_annotation_required": False,
                "human_annotation_used": False,
                "human_ground_truth_claimed": False,
                "pixel_accuracy_claim_allowed": False,
                "automatic_glitch_masks_are_pseudo_labels": True,
                "negative_and_null_masks_retained": True,
                "background_dependence_audits": {
                    arm: {
                        "status": (
                            "detector_set_candidate_background_dependence_audit_v1"
                        ),
                        "passed": True,
                    }
                    for arm in ("raw", "mask")
                },
                "injection_bootstrap_independence": {
                    "status": "injection_bootstrap_independence_audit_v1",
                    "passed": True,
                    "method": (
                        "gps_block_then_paired_injection_hierarchical_bootstrap_v1"
                    ),
                    "physical_groups": 25,
                },
                "validation_only": True,
                "checks": {
                    "minimum_rows": True,
                    "minimum_unique_glitches": True,
                    "minimum_gps_blocks": True,
                    "minimum_labels": True,
                    "chirp_masks_replayed": True,
                    "nonempty_glitch_masks": True,
                    "automatic_replay": True,
                },
                "locked_test_prerequisites_satisfied": False,
                "test_rows_read": 0,
                "scientific_claim_allowed": False,
                "code_commit": "new",
                **artifacts,
            }
        ),
        encoding="utf-8",
    )
    passed = run_publication_evidence_audit(
        protocol,
        [f"paired_raw_mask_vt={evidence}"],
        tmp_path / "passed-raw-mask-audit.json",
    )
    gate = next(
        row for row in passed["requirements"] if row["id"] == "paired_raw_mask_vt"
    )
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 5


def test_official_validation_protocol_requires_variable_detector_calibration(
    tmp_path: Path,
) -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "publication_validation_evidence.yaml"
    )
    artifacts = {}
    for label in (
        "independent_validation_endpoint",
        "candidate_pipeline",
        "calibration",
        "validation_time_slide",
        "background_manifest",
        "candidate_manifest",
        "schedule",
        "validation_injection_ranking",
        "model_selection",
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        artifacts[label] = {"path": str(path), "sha256": file_sha256(path)}

    evidence = tmp_path / "candidate-search.json"
    valid = {
        "status": "frozen_validation_candidate_search_calibration_endpoint_bound",
        "passed": True,
        "publication_calibration_eligible": True,
        "slide_schedule_audit": {
            "passed": True,
            "schedule_kind": "variable_detector_set_block_permutation",
        },
        "background_dependence_audit": {
            "status": "detector_set_candidate_background_dependence_audit_v1",
            "passed": True,
            "gates": {"cluster_bootstrap_complete": True},
            "multiway_cluster_bootstrap": {"replicates": 10000},
            "unique_gps_blocks": 25,
            "unique_shifts": 25,
            "background_manifest": artifacts["background_manifest"],
            "candidate_manifest": artifacts["candidate_manifest"],
            "schedule": artifacts["schedule"],
        },
        "expanded_background_lineage": {"authorization": "purpose_disjoint"},
        "target_far_per_year": 0.1,
        "target_far_has_at_least_one_expected_background_count": True,
        "bootstrap_replicates": 10000,
        "validation_purpose_gps_block_overlap": 0,
        "independent_validation_rows": 3000,
        "test_rows_read": 0,
        "test_evaluation": None,
        "validation_injection_diagnostic": {"status": "validation_only"},
        "injection_bootstrap_independence": {
            "status": "injection_bootstrap_independence_audit_v1",
            "passed": True,
            "method": "gps_block_then_paired_injection_hierarchical_bootstrap_v1",
            "physical_groups": 25,
        },
        "scientific_claim_allowed": False,
        "code_commit": "variable-detector",
        "independent_validation_endpoint": artifacts[
            "independent_validation_endpoint"
        ],
        "candidate_pipeline": artifacts["candidate_pipeline"],
        "calibration": artifacts["calibration"],
        "validation_time_slide": artifacts["validation_time_slide"],
        "validation_injection_ranking": artifacts["validation_injection_ranking"],
        "model_selection": artifacts["model_selection"],
    }
    evidence.write_text(json.dumps(valid), encoding="utf-8")
    passed = run_publication_evidence_audit(
        protocol,
        [f"continuous_candidate_calibration={evidence}"],
        tmp_path / "passed-variable-detector-audit.json",
    )
    gate = next(
        row
        for row in passed["requirements"]
        if row["id"] == "continuous_candidate_calibration"
    )
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 9

    legacy = json.loads(json.dumps(valid))
    legacy["slide_schedule_audit"]["schedule_kind"] = "gps_block_permutation"
    legacy["background_dependence_audit"][
        "status"
    ] = "candidate_background_dependence_audit_v1"
    legacy["background_dependence_audit"]["three_way_cluster_bootstrap"] = legacy[
        "background_dependence_audit"
    ].pop("multiway_cluster_bootstrap")
    legacy.pop("expanded_background_lineage")
    evidence.write_text(json.dumps(legacy), encoding="utf-8")
    failed = run_publication_evidence_audit(
        protocol,
        [f"continuous_candidate_calibration={evidence}"],
        tmp_path / "failed-legacy-detector-audit.json",
    )
    gate = next(
        row
        for row in failed["requirements"]
        if row["id"] == "continuous_candidate_calibration"
    )
    assert gate["state"] == "failed"
    assert {
        row["field"] for row in gate["checks"] if row["passed"] is False
    } == {
        "slide_schedule_audit.schedule_kind",
        "background_dependence_audit.status",
        "background_dependence_audit.multiway_cluster_bootstrap.replicates",
        "expanded_background_lineage",
    }


def test_official_validation_protocol_requires_all_calibration_detector_strata(
    tmp_path: Path,
) -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "publication_validation_evidence.yaml"
    )
    artifacts = {}
    for label in ("plan", "baseline"):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        artifacts[label] = {"path": str(path), "sha256": file_sha256(path)}
    evidence = tmp_path / "calibration-robustness.json"
    report = {
        "status": "completed_validation_calibration_perturbation_robustness",
        "passed": True,
        "physical_time_domain_perturbation": True,
        "fresh_time_frequency_transform": True,
        "detector_strata": {"H1+L1": {"scenario_count": 7}},
        "required_detector_subsets": ["H1+L1"],
        "required_detector_subsets_covered": True,
        "minimum_injections_per_detector_subset": 25,
        "required_detector_subset_minimums_passed": True,
        "test_rows_read": 0,
        "scenario_threshold_refits": 0,
        "scenario_count": 7,
        "injection_bootstrap_independence": {
            "status": "injection_bootstrap_independence_audit_v1",
            "passed": True,
            "method": "gps_block_then_paired_injection_hierarchical_bootstrap_v1",
            "physical_groups": 25,
        },
        "scientific_claim_allowed": False,
        "plan": artifacts["plan"],
        "baseline_calibration": artifacts["baseline"],
    }
    evidence.write_text(json.dumps(report), encoding="utf-8")
    failed = run_publication_evidence_audit(
        protocol,
        [f"calibration_perturbation_robustness={evidence}"],
        tmp_path / "failed-single-pair-calibration.json",
    )
    gate = next(
        row
        for row in failed["requirements"]
        if row["id"] == "calibration_perturbation_robustness"
    )
    assert gate["state"] == "failed"
    assert {
        row["field"] for row in gate["checks"] if row["passed"] is False
    } == {"detector_strata", "required_detector_subsets"}

    subsets = ["H1+L1", "H1+V1", "L1+V1", "H1+L1+V1"]
    report["detector_strata"] = {
        subset: {"scenario_count": 7} for subset in subsets
    }
    report["required_detector_subsets"] = subsets
    evidence.write_text(json.dumps(report), encoding="utf-8")
    passed = run_publication_evidence_audit(
        protocol,
        [f"calibration_perturbation_robustness={evidence}"],
        tmp_path / "passed-detector-set-calibration.json",
    )
    gate = next(
        row
        for row in passed["requirements"]
        if row["id"] == "calibration_perturbation_robustness"
    )
    assert gate["state"] == "passed"


def test_official_validation_protocol_requires_hard_endpoint_scaling_binding(
    tmp_path: Path,
) -> None:
    protocol = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "publication_validation_evidence.yaml"
    )
    evidence = tmp_path / "scaling.json"
    raw = {
        "status": "completed_group_safe_physical_overlap_data_scaling_curve",
        "passed": True,
        "minimum_seeds": 5,
        "paired_seeds": [1, 2, 3, 4, 5],
        "promotion_checks": {
            "fixed_epochs": {"material_positive_gain": True},
            "fixed_optimizer_updates": {"material_positive_gain": True},
        },
        "test_rows_read": 0,
        "code_commit": "diagnostic-only",
    }
    evidence.write_text(json.dumps(raw), encoding="utf-8")
    failed = run_publication_evidence_audit(
        protocol,
        [f"group_safe_data_scaling={evidence}"],
        tmp_path / "raw-scaling-audit.json",
    )
    gate = next(
        row
        for row in failed["requirements"]
        if row["id"] == "group_safe_data_scaling"
    )
    assert gate["state"] == "failed"

    artifacts = {}
    for label in ("subset", "diagnostic", "hard-subset", "hard-bundle"):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        artifacts[label] = {"path": str(path), "sha256": file_sha256(path)}
    bound = {
        **raw,
        "code_commit": "bound-hard-endpoint",
        "hard_endpoint_binding": {
            "passed": True,
            "all_scaling_cells_replayed": True,
            "required_strata": [
                "low_network_snr",
                "missing_detector",
                "o3b_transfer",
                "rare_glitch_family",
            ],
        },
        "hard_endpoint_kind": "predeclared_validation_hard_subset",
        "scale_promotion_authorized": False,
        "diagnosis": "domain_transfer_limited_do_not_scale_same_distribution",
        "subset_report_path": artifacts["subset"]["path"],
        "subset_report_sha256": artifacts["subset"]["sha256"],
        "scaling_diagnostic": artifacts["diagnostic"],
        "hard_subset": artifacts["hard-subset"],
        "hard_endpoint_bundle": artifacts["hard-bundle"],
    }
    evidence.write_text(json.dumps(bound), encoding="utf-8")
    passed = run_publication_evidence_audit(
        protocol,
        [f"group_safe_data_scaling={evidence}"],
        tmp_path / "bound-scaling-audit.json",
    )
    gate = next(
        row
        for row in passed["requirements"]
        if row["id"] == "group_safe_data_scaling"
    )
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 4
