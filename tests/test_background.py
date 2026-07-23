from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

import pytest

from gwyolo.background import (
    _assign_blocks_hash_threshold,
    parse_gps_block_identity,
    plan_background_windows,
    run_background_purpose_partition,
    run_batch_background_plan,
    run_disjoint_background_subset,
    run_background_plan,
    validate_source_verification,
)
from gwyolo.io import file_sha256


def _write_quality_file(path: Path, gps_start: int, duration: int, bad_second: int | None = None) -> None:
    with h5py.File(path, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("GPSstart", data=gps_start)
        meta.create_dataset("Duration", data=duration)
        quality = handle.create_group("quality")
        simple = quality.create_group("simple")
        dq = np.full(duration, 7, dtype=np.int32)
        if bad_second is not None:
            dq[bad_second] = 0
        simple.create_dataset("DQmask", data=dq)
        injections = quality.create_group("injections")
        injections.create_dataset("Injmask", data=np.full(duration, 3, dtype=np.int32))


def test_gps_block_identity_preserves_observing_run_qualification() -> None:
    assert parse_gps_block_identity("gps:1000:64") == ("gps", 1000.0, 64.0)
    assert parse_gps_block_identity("O3b:1000:64") == ("O3b", 1000.0, 64.0)
    with pytest.raises(ValueError, match="unsupported GPS block"):
        parse_gps_block_identity("development:1000:64")


def test_background_windows_use_common_dq_and_disjoint_blocks(tmp_path) -> None:
    h1 = tmp_path / "h1.hdf5"
    l1 = tmp_path / "l1.hdf5"
    _write_quality_file(h1, 1000, 64)
    _write_quality_file(l1, 1000, 64, bad_second=10)
    rows, report = plan_background_windows(
        {"H1": h1, "L1": l1},
        window_duration=4,
        stride=4,
        block_duration=16,
        required_dq_bits=1,
        excluded_intervals=[(1032, 1036)],
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=3,
    )
    starts = {row["gps_start"] for row in rows}
    assert 1008 not in starts  # L1 DQ failure at GPS 1010 removes the whole window.
    assert 1032 not in starts  # Explicit event exclusion.
    assert report["windows"] == 14
    assert report["unique_gps_blocks"] == 4
    assert report["passed"]
    assert all(not values for values in report["cross_split_block_overlaps"].values())
    assert sum(item["live_time_seconds"] for item in report["splits"].values()) == 56


def test_hash_threshold_split_is_stable_under_incremental_shards() -> None:
    first = [f"gps:{index}:256" for index in range(20)]
    second = [f"gps:{index}:256" for index in range(20, 40)]
    combined = _assign_blocks_hash_threshold(first + second, 0.2, 0.2, 7)
    incremental = {
        **_assign_blocks_hash_threshold(first, 0.2, 0.2, 7),
        **_assign_blocks_hash_threshold(second, 0.2, 0.2, 7),
    }
    assert incremental == combined


def test_background_live_time_uses_interval_union(tmp_path) -> None:
    h1 = tmp_path / "h1.hdf5"
    _write_quality_file(h1, 2000, 16)
    _, report = plan_background_windows(
        {"H1": h1},
        window_duration=8,
        stride=4,
        block_duration=16,
        validation_fraction=0,
        test_fraction=0,
    )
    assert report["windows"] == 3
    assert report["splits"]["train"]["live_time_seconds"] == 16


def test_background_excludes_windows_without_full_preprocessing_context(tmp_path) -> None:
    h1 = tmp_path / "h1.hdf5"
    _write_quality_file(h1, 1000, 64)
    rows, report = plan_background_windows(
        {"H1": h1},
        window_duration=8,
        stride=8,
        block_duration=64,
        required_context_duration=32,
        validation_fraction=0,
        test_fraction=0,
    )
    assert [row["gps_start"] for row in rows] == [1016, 1024, 1032, 1040]
    assert report["required_context_duration"] == 32


def test_background_dq_gate_covers_full_whitening_context(tmp_path) -> None:
    h1 = tmp_path / "h1.hdf5"
    _write_quality_file(h1, 1000, 64, bad_second=30)
    rows, report = plan_background_windows(
        {"H1": h1},
        window_duration=8,
        stride=8,
        block_duration=64,
        required_context_duration=16,
        validation_fraction=0,
        test_fraction=0,
    )
    starts = {row["gps_start"] for row in rows}
    assert 1024 not in starts
    assert 1032 not in starts
    assert 1016 in starts
    assert 1040 in starts
    assert report["rejection_counts"]["required_dq_bits_missing_in_context"] == 2


def test_background_run_requires_hash_matched_verified_sources(tmp_path: Path) -> None:
    h1 = tmp_path / "h1.hdf5"
    _write_quality_file(h1, 3000, 16)
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "status": "verified",
                "passed": True,
                "event": "unit-test",
                "detectors": {"H1": {"passed": True, "sha256": file_sha256(h1)}},
            }
        ),
        encoding="utf-8",
    )
    report = run_background_plan(
        {"H1": h1},
        tmp_path / "background",
        verification_path,
        window_duration=8,
        stride=8,
        block_duration=16,
        validation_fraction=0,
        test_fraction=0,
    )
    assert report["source_verification"]["detector_sha256"] == {
        "H1": file_sha256(h1)
    }
    first_row = json.loads(
        (tmp_path / "background" / "background_windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert first_row["source_files"]["H1"]["verification_report_sha256"]

    with h1.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        validate_source_verification({"H1": h1}, verification_path)


def test_batch_background_uses_global_disjoint_block_split(tmp_path: Path) -> None:
    files = []
    for pair_index, gps in enumerate((1000, 2000)):
        for ifo in ("H1", "L1"):
            path = tmp_path / f"{ifo}-{gps}.hdf5"
            _write_quality_file(path, gps, 64)
            files.append(
                {
                    "pair_id": f"pair-{pair_index}",
                    "run": "O4a",
                    "gps_start": gps,
                    "detector": ifo,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "verification": {"passed": True},
                }
            )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "status": "verified_development_strain_batch",
                "passed": True,
                "run": "O4a",
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(
        json.dumps(
            {
                "status": "development_catalog_event_exclusions",
                "run": "O4a",
                "padding_seconds": 16,
                "events": 0,
                "intervals": [],
            }
        ),
        encoding="utf-8",
    )
    result = run_batch_background_plan(
        batch,
        exclusions,
        tmp_path / "planned",
        window_duration=8,
        stride=8,
        block_duration=16,
        required_context_duration=8,
        required_injection_bits=3,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=3,
    )
    assert result["source_pairs"] == 2
    assert len(result["source_batch_report_sha256s"]) == 1
    assert result["windows"] == 16
    assert result["unique_gps_blocks"] == 8
    assert all(not values for values in result["cross_split_block_overlaps"].values())
    rows = [
        json.loads(line)
        for line in (tmp_path / "planned" / "background_windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["pair_id"] for row in rows} == {"pair-0", "pair-1"}
    assert {row["observing_run"] for row in rows} == {"O4a"}


def test_batch_background_merges_reports_before_global_split(tmp_path: Path) -> None:
    reports = []
    for pair_index, gps in enumerate((3000, 4000)):
        files = []
        for ifo in ("H1", "L1"):
            path = tmp_path / f"multi-{ifo}-{gps}.hdf5"
            _write_quality_file(path, gps, 64)
            files.append(
                {
                    "pair_id": f"multi-pair-{pair_index}",
                    "run": "O4a",
                    "gps_start": gps,
                    "detector": ifo,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "verification": {"passed": True},
                }
            )
        report = tmp_path / f"batch-{pair_index}.json"
        report.write_text(
            json.dumps(
                {
                    "status": "verified_development_strain_batch",
                    "passed": True,
                    "run": "O4a",
                    "files": files,
                }
            )
        )
        reports.append(report)
    exclusions = tmp_path / "multi-exclusions.json"
    exclusions.write_text(
        json.dumps(
            {
                "status": "development_catalog_event_exclusions",
                "run": "O4a",
                "padding_seconds": 16,
                "events": 0,
                "intervals": [],
            }
        )
    )
    result = run_batch_background_plan(
        reports,
        exclusions,
        tmp_path / "merged",
        window_duration=8,
        stride=8,
        block_duration=16,
        required_context_duration=8,
        required_injection_bits=3,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=3,
    )
    assert result["source_pairs"] == 2
    assert len(result["source_batch_report_sha256s"]) == 2
    assert result["unique_gps_blocks"] == 8
    assert all(not values for values in result["cross_split_block_overlaps"].values())


def test_disjoint_background_subset_excludes_declared_gps_groups(tmp_path: Path) -> None:
    manifest = tmp_path / "background.jsonl"
    rows = [
        {
            "window_id": "old",
            "split": "val",
            "gps_block": "block-old",
            "gps_start": 0.0,
            "gps_end": 8.0,
        },
        {
            "window_id": "fresh-a",
            "split": "val",
            "gps_block": "block-fresh",
            "gps_start": 8.0,
            "gps_end": 16.0,
        },
        {
            "window_id": "fresh-b",
            "split": "val",
            "gps_block": "block-fresh",
            "gps_start": 16.0,
            "gps_end": 24.0,
        },
        {
            "window_id": "train",
            "split": "train",
            "gps_block": "block-train",
            "gps_start": 24.0,
            "gps_end": 32.0,
        },
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = tmp_path / "background-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_multi_segment_development_background",
                "passed": True,
                "split_strategy": "hash_threshold_v1",
                "split_seed": 7,
                "manifest_sha256": file_sha256(manifest),
            }
        ),
        encoding="utf-8",
    )
    exclusion = tmp_path / "exclude.jsonl"
    exclusion.write_text(
        json.dumps({"gps_block": "block-old"})
        + "\n"
        + json.dumps({"gps_block": "block-not-present"})
        + "\n",
        encoding="utf-8",
    )

    result = run_disjoint_background_subset(
        manifest, report, [exclusion], tmp_path / "disjoint", "val"
    )

    selected = [
        json.loads(line)
        for line in Path(result["manifest_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["window_id"] for row in selected] == ["fresh-a", "fresh-b"]
    assert result["windows"] == 2
    assert result["unique_gps_blocks"] == 1
    assert result["excluded_source_split_windows"] == 1
    assert result["selected_exclusion_gps_block_overlap"] == 0
    assert result["splits"]["val"]["live_time_seconds"] == 16.0
    assert result["splits"]["test"]["windows"] == 0
    assert result["exclusion_manifests"][0]["sha256"] == file_sha256(exclusion)


def test_background_purpose_partition_is_complete_and_group_disjoint(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "validation-background.jsonl"
    rows = [
        {
            "window_id": f"window-{index}",
            "split": "val",
            "gps_block": f"block-{index}",
            "gps_start": float(index * 8),
            "gps_end": float(index * 8 + 8),
        }
        for index in range(20)
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    source_report = tmp_path / "validation-background-report.json"
    source_report.write_text(
        json.dumps(
            {
                "status": "verified_group_disjoint_development_background_subset",
                "passed": True,
                "split_strategy": "hash_threshold_v1",
                "split_seed": 5,
                "manifest_sha256": file_sha256(manifest),
            }
        ),
        encoding="utf-8",
    )

    result = run_background_purpose_partition(
        manifest, source_report, tmp_path / "purposes", injection_fraction=0.5, seed=7
    )

    assert result["passed"]
    assert result["purpose_gps_block_overlap"] == 0
    assert result["complete_source_gps_block_coverage"] is True
    assert result["source_unique_gps_blocks"] == 20
    purpose_blocks = []
    total_windows = 0
    total_live_seconds = 0.0
    for purpose in ("candidate_calibration", "injection_validation"):
        summary = result["purposes"][purpose]
        purpose_rows = [
            json.loads(line)
            for line in Path(summary["manifest_path"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        blocks = {row["gps_block"] for row in purpose_rows}
        purpose_blocks.append(blocks)
        assert blocks
        assert summary["manifest_sha256"] == file_sha256(summary["manifest_path"])
        assert summary["report_sha256"] == file_sha256(summary["report_path"])
        total_windows += summary["windows"]
        total_live_seconds += summary["live_time_seconds"]
    assert purpose_blocks[0].isdisjoint(purpose_blocks[1])
    assert purpose_blocks[0] | purpose_blocks[1] == {row["gps_block"] for row in rows}
    assert total_windows == 20
    assert total_live_seconds == 160.0
