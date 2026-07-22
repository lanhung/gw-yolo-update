from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.cli import main
from gwyolo.io import file_sha256
from gwyolo.publication import run_publication_evidence_audit


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


def test_official_validation_protocol_requires_authorized_raw_mask_receipt(
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
        "authorization",
        "parent_plan",
        "merge_report",
        "raw_calibration",
        "mask_calibration",
        "paired_comparison",
        "mask_validation",
        "mask_timing",
    ):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label}), encoding="utf-8")
        artifacts[label] = {"path": str(path), "sha256": file_sha256(path)}
    evidence.write_text(
        json.dumps(
            {
                "status": "completed_validation_only_raw_mask_continuous_background",
                "passed": True,
                "mask_locked_test_arm_eligible": True,
                "validation_calibration_frozen": True,
                "background_plan_authorization_id": "authorization-id",
                "background_plan_purpose_disjoint": True,
                "background_plan_capacity_authorized": True,
                "locked_test_prerequisites_satisfied": False,
                "test_rows_read": 0,
                "scientific_claim_allowed": False,
                "code_commit": "new",
                "inputs": {
                    "background_plan_authorization": artifacts["authorization"],
                    "parent_plan": artifacts["parent_plan"],
                },
                "merge_report": artifacts["merge_report"],
                "calibrations": {
                    "raw": artifacts["raw_calibration"],
                    "mask": artifacts["mask_calibration"],
                },
                "paired_validation_comparison": artifacts["paired_comparison"],
                "mask_validation_receipt": artifacts["mask_validation"],
                "mask_timing_receipt": artifacts["mask_timing"],
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
    assert len(gate["artifact_replay"]) == 8
