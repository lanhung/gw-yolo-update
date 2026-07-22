from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import atomic_write_json, file_sha256, load_yaml
from gwyolo.locked_injections import (
    audit_gwtc5_locked_injection_rows,
    run_gwtc5_locked_injection_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_CONFIG = ROOT / "configs/locked_evaluation_suite_gwtc5.yaml"
POPULATION_CONFIG = ROOT / "configs/gwtc5_locked_injection_population.yaml"


def _availability_bundle(root: Path) -> tuple[Path, Path, Path]:
    access_log = root / "gwtc5-access.json"
    distribution = (
        ("H1+L1", 250),
        ("H1+V1", 250),
        ("L1+V1", 1000),
        ("H1+L1+V1", 2500),
    )
    rows = []
    index = 0
    required = ("H1+L1", "H1+V1", "L1+V1", "H1+L1+V1")
    for subset, count in distribution:
        ifos = subset.split("+")
        compatible = [
            value for value in required if set(value.split("+")) <= set(ifos)
        ]
        for _ in range(count):
            gps = 1_400_000_000 + index * 4096
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
                    "compatible_detector_subsets": compatible,
                    "sources": {},
                }
            )
            index += 1
    manifest = root / "availability.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = root / "availability-report.json"
    atomic_write_json(
        report,
        {
            "status": "score_blind_gwtc5_o4b_availability_inventory",
            "passed": True,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": file_sha256(manifest),
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
    return manifest, report, access_log


@pytest.fixture(scope="module")
def locked_plan(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("locked-plan")
    availability, availability_report, access_log = _availability_bundle(root)
    output = root / "planned"
    report = run_gwtc5_locked_injection_inventory(
        availability,
        availability_report,
        SUITE_CONFIG,
        POPULATION_CONFIG,
        access_log,
        output,
    )
    return {
        "root": root,
        "availability": availability,
        "availability_report": availability_report,
        "access_log": access_log,
        "output": output,
        "report": report,
    }


def test_locked_injection_plan_is_deterministic_physical_and_score_blind(
    locked_plan: dict[str, object],
) -> None:
    report = locked_plan["report"]
    assert isinstance(report, dict)
    assert report["rows"] == 4000
    assert report["minimum_usable_after_dq"] == 3000
    assert report["candidate_scores_inspected"] is False
    assert report["test_strain_rows_read"] == 0
    assert report["pre_access_vt_weights_assigned"] is False
    audit = report["audit"]
    assert audit["source_family_counts"] == {"BBH": 2000, "BNS": 1000, "NSBH": 1000}
    assert audit["detector_subset_counts"] == {
        "H1+L1": 250,
        "H1+L1+V1": 2500,
        "H1+V1": 250,
        "L1+V1": 1000,
    }
    assert audit["stress_stratum_counts"]["high_mass_unequal_mass"] == 600
    assert audit["stress_stratum_counts"]["high_spin_precessing"] == 600
    assert audit["one_injection_per_frozen_gps_block"] is True
    rows = [
        json.loads(line)
        for line in Path(report["manifest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["pre_access_vt_weight"] is None for row in rows)
    high_spin = next(row for row in rows if "high_spin_precessing" in row["stress_strata"])
    assert abs(high_spin["spin_1x"]) + abs(high_spin["spin_1y"]) > 0.5
    assert high_spin["waveform_approximant"] == "IMRPhenomXPHM"

    second = Path(locked_plan["root"]) / "planned-second"
    replay = run_gwtc5_locked_injection_inventory(
        locked_plan["availability"],
        locked_plan["availability_report"],
        SUITE_CONFIG,
        POPULATION_CONFIG,
        locked_plan["access_log"],
        second,
    )
    assert replay["manifest_sha256"] == report["manifest_sha256"]


def test_locked_physical_audit_rejects_fabricated_stress_labels(
    locked_plan: dict[str, object],
) -> None:
    report = locked_plan["report"]
    rows = [
        json.loads(line)
        for line in Path(report["manifest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    availability_rows = [
        json.loads(line)
        for line in Path(locked_plan["availability"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    high_spin = next(row for row in rows if "high_spin_precessing" in row["stress_strata"])
    high_spin["spin_1x"] = 0.0
    high_spin["spin_1y"] = 0.0
    suite = load_yaml(SUITE_CONFIG)["locked_evaluation_suite"]
    population = load_yaml(POPULATION_CONFIG)["gwtc5_locked_injection_population"]
    with pytest.raises(ValueError, match="high-spin precessing label is not physical"):
        audit_gwtc5_locked_injection_rows(rows, availability_rows, suite, population)


def test_locked_injection_plan_fails_after_access_log(
    locked_plan: dict[str, object],
) -> None:
    access_log = Path(locked_plan["access_log"])
    access_log.write_text('{"status":"opened"}\n', encoding="utf-8")
    try:
        with pytest.raises(FileExistsError, match="access log"):
            run_gwtc5_locked_injection_inventory(
                locked_plan["availability"],
                locked_plan["availability_report"],
                SUITE_CONFIG,
                POPULATION_CONFIG,
                access_log,
                Path(locked_plan["root"]) / "must-not-exist",
            )
    finally:
        access_log.unlink()
