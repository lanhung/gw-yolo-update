from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from gwyolo.evaluation_lock import freeze_gwtc5_locked_corpus_contract
from gwyolo.io import atomic_write_json, file_sha256
from gwyolo.locked_injections import run_gwtc5_locked_injection_inventory
from gwyolo.publication import run_publication_evidence_audit


ROOT = Path(__file__).resolve().parents[1]
SUITE_CONFIG = ROOT / "configs/locked_evaluation_suite_gwtc5.yaml"
POPULATION_CONFIG = ROOT / "configs/gwtc5_locked_injection_population.yaml"
LEDGER_CONFIG = ROOT / "configs/publication_validation_evidence.yaml"


@pytest.fixture(scope="module")
def planned_inventory(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("gwtc5-freeze")
    access_log = root / "gwtc5-access.json"
    distribution = (
        ("H1+L1", 250),
        ("H1+V1", 250),
        ("L1+V1", 1000),
        ("H1+L1+V1", 2500),
    )
    required = ("H1+L1", "H1+V1", "L1+V1", "H1+L1+V1")
    rows = []
    index = 0
    for subset, count in distribution:
        ifos = subset.split("+")
        for _ in range(count):
            gps = 1_410_000_000 + index * 4096
            rows.append(
                {
                    "availability_id": f"availability-{index:05d}",
                    "split": "test",
                    "observing_run": "O4b",
                    "catalog_release": "GWTC-5.0",
                    "gps_start": gps,
                    "gps_end": gps + 4096,
                    "gps_block": f"O4b:{gps}:4096",
                    "available_ifos": ifos,
                    "compatible_detector_subsets": [
                        value
                        for value in required
                        if set(value.split("+")) <= set(ifos)
                    ],
                    "sources": {},
                }
            )
            index += 1
    availability = root / "availability.jsonl"
    availability.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    availability_report = root / "availability-report.json"
    atomic_write_json(
        availability_report,
        {
            "status": "score_blind_gwtc5_o4b_availability_inventory",
            "passed": True,
            "manifest_path": str(availability.resolve()),
            "manifest_sha256": file_sha256(availability),
            "suite_config_path": str(SUITE_CONFIG.resolve()),
            "suite_config_sha256": file_sha256(SUITE_CONFIG),
            "access_log_path": str(access_log.resolve()),
            "candidate_catalog_queried": False,
            "candidate_scores_inspected": False,
            "event_level_parameters_inspected": False,
            "test_strain_files_downloaded": 0,
            "test_strain_bytes_read": 0,
            "test_strain_rows_read": 0,
            "availability_blocks": len(rows),
        },
    )
    plan_root = root / "plan"
    plan = run_gwtc5_locked_injection_inventory(
        availability,
        availability_report,
        SUITE_CONFIG,
        POPULATION_CONFIG,
        access_log,
        plan_root,
    )
    planned_rows = [
        json.loads(line)
        for line in Path(plan["manifest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    strata: Counter[str] = Counter()
    approximants = set()
    for row in planned_rows:
        family = row["source_family"]
        primary = row["waveform_approximant"]
        strata[f"{family}:primary:{primary}"] = 3
        approximants.add(primary)
        alternative = row.get("alternative_waveform_approximant")
        if alternative:
            strata[f"{family}:alternative:{alternative}"] = 3
            approximants.add(alternative)
    waveform_report = root / "waveform-validation.json"
    runtime_requirements = root / "waveform-requirements.txt"
    runtime_requirements.write_text("pycbc==test\nlalsuite==test\n", encoding="utf-8")
    frozen_packages = ["lalsuite==test", "pycbc==test"]
    frozen_text = "\n".join(frozen_packages) + "\n"
    runtime_receipt = root / "waveform-runtime-receipt.json"
    atomic_write_json(
        runtime_receipt,
        {
            "status": "verified_isolated_waveform_runtime",
            "passed": True,
            "code_commit": "frozen-commit",
            "python_executable": "/test/waveform/python",
            "requirements_path": str(runtime_requirements.resolve()),
            "requirements_sha256": file_sha256(runtime_requirements),
            "pycbc_version": "test",
            "lalsuite_version": "test",
            "approximants": {name: {} for name in sorted(approximants)},
            "pip_freeze": frozen_packages,
            "pip_freeze_sha256": hashlib.sha256(frozen_text.encode()).hexdigest(),
        },
    )
    cases = [
        {
            "source_family": key.split(":", 2)[0],
            "waveform_role": key.split(":", 2)[1],
            "approximant": key.split(":", 2)[2],
            "passed": True,
        }
        for key, count in sorted(strata.items())
        for _ in range(count)
    ]
    atomic_write_json(
        waveform_report,
        {
            "passed": True,
            "validation_scope": "external_reference_waveform_equivalence",
            "selection_mode": "family_approximant",
            "include_alternatives": True,
            "recipe_manifest_path": str(Path(plan["manifest_path"]).resolve()),
            "recipe_manifest_sha256": file_sha256(plan["manifest_path"]),
            "approximants": sorted(approximants),
            "selected_cases": len(cases),
            "case_strata": dict(sorted(strata.items())),
            "versions": {"pycbc": "test", "lalsuite": "test"},
            "runtime_receipt_bound": True,
            "runtime_receipt_path": str(runtime_receipt.resolve()),
            "runtime_receipt_sha256": file_sha256(runtime_receipt),
            "requirements_sha256": file_sha256(runtime_requirements),
            "pip_freeze_sha256": hashlib.sha256(frozen_text.encode()).hexdigest(),
            "code_commit": "frozen-commit",
            "environment": {"python_executable": "/test/waveform/python"},
            "cases": cases,
        },
    )
    return {
        "root": root,
        "access_log": access_log,
        "manifest": Path(plan["manifest_path"]),
        "inventory_report": plan_root / "gwtc5_locked_injection_inventory_report.json",
        "waveform_report": waveform_report,
    }


def test_gwtc5_freeze_binds_physical_producer_and_live_access_log(
    planned_inventory: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "frozen-commit")
    access = planned_inventory["access_log"]
    if access.exists():
        access.unlink()
    output = planned_inventory["root"] / "gwtc5-unopened.json"
    report = freeze_gwtc5_locked_corpus_contract(
        planned_inventory["manifest"],
        planned_inventory["inventory_report"],
        planned_inventory["waveform_report"],
        SUITE_CONFIG,
        output,
        access,
    )
    assert report["rows"] == 4000
    assert report["test_strain_rows_read"] == 0
    assert report["inventory_producer_bound"] is True
    assert report["physical_stress_predicates_passed"] is True
    assert report["waveform_runtime_validation_bound"] is True
    assert report["one_injection_per_frozen_gps_block"] is True
    assert report["pre_access_vt_weights_assigned"] is False
    assert report["post_access_dq_replacement_allowed"] is False

    first = run_publication_evidence_audit(
        LEDGER_CONFIG,
        [f"locked_corpus_unopened={output}"],
        planned_inventory["root"] / "ledger-before-access.json",
    )
    gate = next(
        row for row in first["requirements"] if row["id"] == "locked_corpus_unopened"
    )
    assert gate["state"] == "passed"
    assert len(gate["artifact_replay"]) == 8

    access.write_text(json.dumps({"status": "opened"}), encoding="utf-8")
    try:
        second = run_publication_evidence_audit(
            LEDGER_CONFIG,
            [f"locked_corpus_unopened={output}"],
            planned_inventory["root"] / "ledger-after-access.json",
        )
        gate = next(
            row
            for row in second["requirements"]
            if row["id"] == "locked_corpus_unopened"
        )
        assert gate["state"] == "failed"
        failed_fields = {row["field"] for row in gate["checks"] if not row["passed"]}
        assert failed_fields == {"access_log_path"}
    finally:
        access.unlink()


def test_gwtc5_freeze_rejects_manifest_tampering(
    planned_inventory: dict[str, Path], tmp_path: Path
) -> None:
    rows = planned_inventory["manifest"].read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["candidate_score"] = 0.2
    rows[0] = json.dumps(row, sort_keys=True)
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source-safe injection producer"):
        freeze_gwtc5_locked_corpus_contract(
            tampered,
            planned_inventory["inventory_report"],
            planned_inventory["waveform_report"],
            SUITE_CONFIG,
            tmp_path / "must-not-exist.json",
            planned_inventory["access_log"],
        )


def test_gwtc5_freeze_rejects_unbound_producer_report(
    planned_inventory: dict[str, Path], tmp_path: Path
) -> None:
    report = json.loads(
        planned_inventory["inventory_report"].read_text(encoding="utf-8")
    )
    report["physical_stress_predicates_passed"] = False
    invalid = tmp_path / "invalid-producer.json"
    atomic_write_json(invalid, report)
    with pytest.raises(ValueError, match="source-safe injection producer"):
        freeze_gwtc5_locked_corpus_contract(
            planned_inventory["manifest"],
            invalid,
            planned_inventory["waveform_report"],
            SUITE_CONFIG,
            tmp_path / "must-not-exist.json",
            planned_inventory["access_log"],
        )
