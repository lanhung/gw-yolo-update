from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.detector_validation_data import (
    export_network_numeric_validation_background,
    freeze_source_disjoint_detector_acquisition_plan,
    plan_detector_stratified_validation_injections,
)
from gwyolo.io import file_sha256


def _write_source(
    path: Path,
    available_ifos: list[str],
    event_gps: float,
) -> None:
    model_ifos = ["H1", "L1", "V1"]
    availability = np.asarray(
        [ifo in available_ifos for ifo in model_ifos], dtype=np.uint8
    )
    raw = np.zeros((3, 32), dtype=np.float32)
    for index, valid in enumerate(availability):
        if valid:
            raw[index] = np.linspace(index + 1, index + 2, 32)
    np.savez_compressed(
        path,
        raw_strain=raw,
        ifos=np.asarray(model_ifos),
        sample_rate=np.asarray(4),
        event_gps=np.asarray(event_gps),
        detector_availability=availability,
    )


def _write_inputs(
    tmp_path: Path,
    subsets: list[list[str]],
    duplicate_first_block: bool = False,
) -> tuple[Path, Path]:
    rows = []
    for index, subset in enumerate(subsets):
        source = tmp_path / f"source-{index}.npz"
        _write_source(source, subset, 1000.0 + index * 20)
        rows.append(
            {
                "split": "val",
                "aligned_network_context": True,
                "glitch_id": f"g{index}",
                "network_gps_block": f"O3:block-{index}",
                "available_ifos": subset,
                "observing_run": "O3",
                "path": str(source),
                "sha256": file_sha256(source),
            }
        )
    if duplicate_first_block:
        source = tmp_path / "duplicate.npz"
        _write_source(source, subsets[0], 1000.5)
        rows.append(
            {
                **rows[0],
                "glitch_id": "g-duplicate",
                "path": str(source),
                "sha256": file_sha256(source),
            }
        )
    manifest = tmp_path / "network.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": (
                    "verified_group_safe_gravityspy_aligned_network_corpus"
                ),
                "passed": True,
                "scientific_claim_allowed": False,
                "validation_manifest_sha256": file_sha256(manifest),
                "split_audit": {
                    "cross_split_overlaps": {
                        "glitch_id": [],
                        "network_gps_block": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, audit


def test_detector_validation_background_exports_one_real_bank_per_block(
    tmp_path: Path,
) -> None:
    subsets = [
        ["H1", "L1"],
        ["H1", "V1"],
        ["L1", "V1"],
        ["H1", "L1", "V1"],
    ]
    manifest, audit = _write_inputs(
        tmp_path,
        subsets,
        duplicate_first_block=True,
    )

    result = export_network_numeric_validation_background(
        manifest,
        audit,
        tmp_path / "output",
        analysis_duration_seconds=4.0,
        minimum_per_detector_subset=1,
        require_ready=True,
    )

    assert result["passed"] is True
    assert result["source_rows"] == 5
    assert result["selected_rows"] == 4
    assert result["unique_network_gps_blocks"] == 4
    assert result["detector_subset_counts"] == {
        "H1+L1": 1,
        "H1+V1": 1,
        "L1+V1": 1,
        "H1+L1+V1": 1,
    }
    exported = [
        json.loads(line)
        for line in Path(result["manifest_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in exported:
        assert row["physical_signal_present"] is False
        assert row["physical_signal_projection_required"] is True
        with np.load(row["background_bank"]["path"], allow_pickle=False) as arrays:
            noise = np.asarray(arrays["noise"])
            assert noise.shape[0] == len(row["ifos"])
            assert np.all(np.any(noise != 0, axis=1))

    plan = plan_detector_stratified_validation_injections(
        result["manifest_path"],
        tmp_path / "output" / "detector_validation_background_report.json",
        tmp_path / "plan",
        injections_per_detector_subset=2,
    )
    assert plan["passed"] is True
    assert plan["rows"] == 8
    assert plan["detector_subset_counts"] == {
        "H1+L1": 2,
        "H1+V1": 2,
        "L1+V1": 2,
        "H1+L1+V1": 2,
    }
    recipes = [
        json.loads(line)
        for line in Path(plan["manifest_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len({row["injection_id"] for row in recipes}) == 8
    assert len({row["waveform_id"] for row in recipes}) == 8
    assert all(row["split"] == "val" for row in recipes)


def test_detector_validation_background_reports_subset_deficit(
    tmp_path: Path,
) -> None:
    manifest, audit = _write_inputs(
        tmp_path,
        [["H1", "L1"], ["H1", "V1"], ["H1", "L1", "V1"]],
    )

    result = export_network_numeric_validation_background(
        manifest,
        audit,
        tmp_path / "diagnostic",
        minimum_per_detector_subset=1,
    )

    assert result["passed"] is False
    assert result["detector_subset_deficits"]["L1+V1"] == 1
    with pytest.raises(RuntimeError, match="below frozen subset floors"):
        export_network_numeric_validation_background(
            manifest,
            audit,
            tmp_path / "required",
            minimum_per_detector_subset=1,
            require_ready=True,
        )


def test_detector_validation_background_rejects_changed_numeric_source(
    tmp_path: Path,
) -> None:
    manifest, audit = _write_inputs(tmp_path, [["H1", "L1"]])
    row = json.loads(manifest.read_text(encoding="utf-8"))
    Path(row["path"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="source hash changed"):
        export_network_numeric_validation_background(
            manifest,
            audit,
            tmp_path / "tampered",
            required_detector_subsets=["H1+L1"],
            minimum_per_detector_subset=1,
        )


def test_detector_acquisition_plan_excludes_frozen_sources_and_prior_plan(
    tmp_path: Path,
) -> None:
    def pair(index: int) -> dict:
        gps = 1000 + index * 100
        return {
            "pair_id": f"O3-{gps}-L1-V1",
            "run": "O3",
            "gps_start": gps,
            "detectors": {
                ifo: {
                    "detector": ifo,
                    "gps_start": gps,
                    "sample_rate": 4096,
                    "hdf5_url": f"https://gwosc.org/{ifo}-{gps}.hdf5",
                    "detail_url": f"https://gwosc.org/api/{ifo}-{gps}",
                }
                for ifo in ("L1", "V1")
            },
        }

    inventory = tmp_path / "inventory.json"
    pairs = [pair(index) for index in range(4)]
    inventory.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "locked_evaluation_data": False,
                "run": "O3",
                "detectors": ["L1", "V1"],
                "sample_rate_khz": 4,
                "seed": 1,
                "selected_pairs": 4,
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )
    frozen = tmp_path / "frozen.jsonl"
    frozen.write_text(
        json.dumps(
            {
                "network_strain_sources": {
                    ifo: {
                        "gps_start": 1000,
                        "hdf5_url": pairs[0]["detectors"][ifo]["hdf5_url"],
                    }
                    for ifo in ("L1", "V1")
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    excluded = tmp_path / "excluded.json"
    excluded.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "locked_evaluation_data": False,
                "run": "O3",
                "pairs": [pairs[1]],
            }
        ),
        encoding="utf-8",
    )

    result = freeze_source_disjoint_detector_acquisition_plan(
        inventory,
        [frozen],
        tmp_path / "selected.json",
        target_pairs=2,
        seed=123,
        exclusion_plan_paths=[excluded],
    )

    assert result["selected_pairs"] == 2
    assert {row["gps_start"] for row in result["pairs"]} == {1200, 1300}
    assert result["candidate_scores_inspected"] is False
    assert result["test_data_opened"] is False
