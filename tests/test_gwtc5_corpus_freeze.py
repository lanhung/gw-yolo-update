from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.evaluation_lock import freeze_gwtc5_locked_corpus_contract
from gwyolo.publication import run_publication_evidence_audit


SUITE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/locked_evaluation_suite_gwtc5.yaml"
)
LEDGER_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/publication_validation_evidence.yaml"
)
DETECTOR_SUBSETS = ("H1+L1", "H1+V1", "L1+V1", "H1+L1+V1")
SOURCE_FAMILIES = ("BBH", "BNS", "NSBH")
STRESS_STRATA = (
    "glitch_overlap",
    "missing_detector",
    "calibration_perturbation",
    "waveform_systematics",
    "high_mass_unequal_mass",
    "high_spin_precessing",
)


def _write_manifest(path: Path, rows: int = 3000, **updates: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            row = {
                "split": "test",
                "injection_id": f"locked-injection-{index}",
                "waveform_id": f"locked-waveform-{index}",
                "gps_block": f"locked-block-{index // 10}",
                "source_family": SOURCE_FAMILIES[index % len(SOURCE_FAMILIES)],
                "observing_run": "O4b",
                "detector_subset": DETECTOR_SUBSETS[index % len(DETECTOR_SUBSETS)],
                "catalog_release": "GWTC-5.0",
                "stress_strata": list(STRESS_STRATA),
                **updates,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_gwtc5_freeze_binds_exact_suite_and_ledger_checks_live_access_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "frozen-commit")
    manifest = tmp_path / "gwtc5-o4b-test.jsonl"
    _write_manifest(manifest)
    output = tmp_path / "gwtc5-unopened.json"
    access = tmp_path / "gwtc5-access.json"

    report = freeze_gwtc5_locked_corpus_contract(
        manifest,
        SUITE_CONFIG,
        output,
        access,
    )
    assert report["corpus_label"] == "GWTC-5.0_O4b_locked_suite_v2"
    assert report["rows"] == 3000
    assert report["test_strain_rows_read"] == 0
    assert report["candidate_scores_inspected"] is False
    assert report["required_detector_subsets_covered"] is True
    assert not access.exists()

    first = run_publication_evidence_audit(
        LEDGER_CONFIG,
        [f"locked_corpus_unopened={output}"],
        tmp_path / "ledger-before-access.json",
    )
    gate = next(
        row for row in first["requirements"] if row["id"] == "locked_corpus_unopened"
    )
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 2

    access.write_text(json.dumps({"status": "opened"}), encoding="utf-8")
    second = run_publication_evidence_audit(
        LEDGER_CONFIG,
        [f"locked_corpus_unopened={output}"],
        tmp_path / "ledger-after-access.json",
    )
    gate = next(
        row for row in second["requirements"] if row["id"] == "locked_corpus_unopened"
    )
    assert gate["state"] == "failed"
    failed_fields = {row["field"] for row in gate["checks"] if not row["passed"]}
    assert failed_fields == {"access_log_path"}


def test_gwtc5_freeze_rejects_undersized_or_non_o4b_inventory(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.jsonl"
    _write_manifest(manifest, rows=10)
    with pytest.raises(ValueError, match="injection floor"):
        freeze_gwtc5_locked_corpus_contract(
            manifest,
            SUITE_CONFIG,
            tmp_path / "small.json",
            tmp_path / "access.json",
        )

    _write_manifest(manifest, observing_run="O4a")
    with pytest.raises(ValueError, match="non-O4b"):
        freeze_gwtc5_locked_corpus_contract(
            manifest,
            SUITE_CONFIG,
            tmp_path / "wrong-run.json",
            tmp_path / "access.json",
        )


def test_gwtc5_freeze_rejects_selection_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "scores-exposed.jsonl"
    _write_manifest(manifest, candidate_score=0.2)
    with pytest.raises(ValueError, match="selection/result fields"):
        freeze_gwtc5_locked_corpus_contract(
            manifest,
            SUITE_CONFIG,
            tmp_path / "must-not-exist.json",
            tmp_path / "access.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()
