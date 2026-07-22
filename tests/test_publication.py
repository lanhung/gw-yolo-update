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
