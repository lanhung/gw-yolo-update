from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.gravityspy import (
    _effective_network_detector_availability,
    evict_gravityspy_verified_sources,
    gravityspy_weak_mask,
    index_gravityspy_csv,
    match_glitch_to_strain_file,
    materialize_gravityspy_network_strain,
    merge_gravityspy_numeric_manifests,
    merge_gravityspy_network_numeric_manifests,
    plan_gravityspy_network_strain,
    plan_gravityspy_network_recovery,
    select_gravityspy_source_files,
    select_gravityspy_network_source_components,
    shard_gravityspy_network_strain_plan,
    shard_gravityspy_strain_plan,
    split_gravityspy_anchors,
)
from gwyolo.io import file_sha256


def test_gravityspy_weak_mask_is_detector_local_and_hand_calculable() -> None:
    mask = gravityspy_weak_mask(
        "L1",
        ("H1", "L1", "V1"),
        (4.0, 8.0),
        frequency_bins=5,
        time_bins=8,
        fmin=0.0,
        fmax=40.0,
        duration=2.0,
        peak_frequency=20.0,
        quality_factor=2.0,
        output_duration=8.0,
    )
    # Frequencies 10, 20, 30 and times -1, 0, 1 are included for both Q planes.
    assert mask.shape == (3, 2, 5, 8)
    assert int(mask.sum()) == 3 * 3 * 2
    assert int(mask[0].sum()) == 0
    assert int(mask[1].sum()) == 18
    assert int(mask[2].sum()) == 0


def test_network_detector_availability_downgrades_only_valid_companions() -> None:
    quality = {
        "H1": {"usable": True},
        "L1": {"usable": False, "reason": "nonfinite_strain_context"},
        "V1": {"usable": True},
    }
    availability, reason = _effective_network_detector_availability(
        quality, ("H1", "L1", "V1"), "H1"
    )
    assert availability.tolist() == [1, 0, 1]
    assert reason is None
    availability, reason = _effective_network_detector_availability(
        {"H1": {"usable": True}, "L1": {"usable": False}},
        ("H1", "L1", "V1"),
        "H1",
    )
    assert availability.tolist() == [1, 0, 0]
    assert reason == "fewer_than_two_usable_detectors"
    _, reason = _effective_network_detector_availability(
        quality, ("H1", "L1", "V1"), "L1"
    )
    assert reason == "event_ifo_unusable"


def test_gravityspy_network_plan_preserves_real_detector_availability(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "anchors.jsonl"
    rows = [
        {
            "glitch_id": "g-full",
            "split": "train",
            "observing_run": "O3a",
            "ifo": "H1",
            "event_time": 1100.0,
            "network_gps_block": "O3a:1",
        },
        {
            "glitch_id": "g-no-companion",
            "split": "train",
            "observing_run": "O3a",
            "ifo": "H1",
            "event_time": 2100.0,
            "network_gps_block": "O3a:2",
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def fake_results(endpoint):
        ifo = endpoint.split("detector=")[1].split("&")[0]
        records = []
        if ifo in {"H1", "L1"}:
            records.append(
                {
                    "detector": ifo,
                    "sample_rate_kHz": 4,
                    "gps_start": 1000,
                    "hdf5_url": f"https://example/{ifo}-1000-512.hdf5",
                    "detail_url": f"https://example/{ifo}/detail",
                }
            )
        if ifo == "H1":
            records.append(
                {
                    "detector": ifo,
                    "sample_rate_kHz": 4,
                    "gps_start": 2000,
                    "hdf5_url": f"https://example/{ifo}-2000-4096.hdf5",
                    "detail_url": f"https://example/{ifo}/detail2",
                }
            )
        return records, {"api_results_count": len(records), "api_pages": 1}

    monkeypatch.setattr("gwyolo.gravityspy._api_results", fake_results)
    report = plan_gravityspy_network_strain(
        source, tmp_path / "network", context_duration=64, minimum_detectors=2
    )
    assert report["planned_rows"] == 1
    assert report["rejected_rows"] == 1
    assert report["detector_subset_counts"] == {"H1L1": 1}
    planned = json.loads(Path(report["manifest_path"]).read_text().strip())
    assert planned["available_ifos"] == ["H1", "L1"]
    assert planned["detector_availability"] == [1, 1, 0]
    assert set(planned["network_strain_sources"]) == {"H1", "L1"}
    assert report["network_coherence_claim_allowed"] is False


def test_gravityspy_network_plan_rejects_duplicate_glitch_identity(tmp_path) -> None:
    source = tmp_path / "duplicate.jsonl"
    row = {
        "glitch_id": "same",
        "observing_run": "O3a",
        "ifo": "H1",
        "event_time": 1000.0,
        "network_gps_block": "block",
    }
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate glitch"):
        plan_gravityspy_network_strain(source, tmp_path / "network")


def test_gravityspy_network_materialization_keeps_aligned_detector_planes(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """physical_training:
  model_ifos: [H1, L1, V1]
  q_values: [4]
  target_sample_rate: 64
  tensor:
    frequency_bins: 8
    time_bins: 8
    fmin: 4
    fmax: 30
"""
    )
    sources = {
        ifo: {
            "detector": ifo,
            "observing_run": "O2",
            "gps_start": 1000,
            "duration": 4096,
            "sample_rate": 64,
            "hdf5_url": f"https://example/{ifo}-1000-4096.hdf5",
            "detail_url": f"https://example/{ifo}/detail",
        }
        for ifo in ("H1", "L1", "V1")
    }
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        json.dumps(
            {
                "glitch_id": "g-network",
                "split": "val",
                "network_gps_block": "O2:block",
                "observing_run": "O2",
                "ifo": "H1",
                "event_time": 1100.0,
                "duration": 0.2,
                "peak_frequency": 20.0,
                "q_value": 4.0,
                "context_duration": 4.0,
                "available_ifos": ["H1", "L1", "V1"],
                "detector_availability": [1, 1, 1],
                "network_strain_sources": sources,
            }
        )
        + "\n"
    )
    downloaded = {}

    def fake_download(url, path, workers):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(url.encode())
        downloaded[url] = path
        return {"path": str(path)}

    def fake_verify(path, detail, chunk_samples):
        return {
            "passed": True,
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": Path(path).stat().st_size,
        }

    def fake_segment(path, event_time, context_duration):
        if "/V1/" in str(path):
            return {
                "strain": np.full(256, np.nan),
                "sample_rate": 64,
                "quality": {
                    "DQmask": np.ones(4, dtype=np.int64),
                    "Injmask": np.zeros(4, dtype=np.int64),
                },
            }
        phase = 0.0 if "/H1/" in str(path) else 0.5
        return {
            "strain": np.sin(np.linspace(phase, phase + 20, 256)) * 1e-21,
            "sample_rate": 64,
            "quality": {
                "DQmask": np.ones(4, dtype=np.int64),
                "Injmask": np.zeros(4, dtype=np.int64),
            },
        }

    monkeypatch.setattr("gwyolo.gravityspy.download_resumable", fake_download)
    monkeypatch.setattr("gwyolo.gravityspy.verify_hdf5_against_detail", fake_verify)
    monkeypatch.setattr("gwyolo.gravityspy._api_json", lambda _: {})
    monkeypatch.setattr("gwyolo.gravityspy.read_hdf5_segment", fake_segment)
    report = materialize_gravityspy_network_strain(
        plan, config, tmp_path / "cache", tmp_path / "output", output_duration=2.0
    )
    assert report["rows"] == 1
    assert report["verified_files"] == 3
    assert report["detector_subset_counts"] == {"H1L1": 1}
    assert report["planned_detector_subset_counts"] == {"H1L1V1": 1}
    assert report["runtime_detector_downgraded_rows"] == 1
    assert report["unusable_detector_reason_counts"] == {
        "nonfinite_strain_context": 1
    }
    row = json.loads(Path(report["manifest_path"]).read_text().strip())
    with np.load(row["path"], allow_pickle=False) as arrays:
        assert arrays["features"].shape == (3, 1, 8, 8)
        assert arrays["raw_strain"].shape == (3, 128)
        assert arrays["detector_availability"].tolist() == [1, 1, 0]
        assert np.count_nonzero(arrays["features"][:2]) > 0
        assert np.count_nonzero(arrays["features"][2]) == 0
        assert np.count_nonzero(arrays["glitch_mask"][0]) > 0
        assert np.count_nonzero(arrays["glitch_mask"][1:]) == 0

    def reject_redundant_download(*_args, **_kwargs):
        raise AssertionError("completed materialization reacquired source strain")

    monkeypatch.setattr(
        "gwyolo.gravityspy.download_resumable", reject_redundant_download
    )
    resumed = materialize_gravityspy_network_strain(
        plan, config, tmp_path / "cache", tmp_path / "output", output_duration=2.0
    )
    assert resumed == report


def test_completed_gravityspy_network_materialization_rejects_changed_sample(
    tmp_path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """physical_training:
  model_ifos: [H1, L1, V1]
  q_values: [4]
  target_sample_rate: 64
  tensor: {frequency_bins: 8, time_bins: 8, fmin: 4, fmax: 30}
"""
    )
    sources = {
        ifo: {
            "detector": ifo,
            "observing_run": "O2",
            "hdf5_url": f"https://example/{ifo}.hdf5",
            "detail_url": f"https://example/{ifo}/detail",
        }
        for ifo in ("H1", "L1")
    }
    plan = tmp_path / "plan.jsonl"
    plan.write_text(
        json.dumps(
            {
                "glitch_id": "g",
                "split": "train",
                "network_gps_block": "O2:block",
                "observing_run": "O2",
                "ifo": "H1",
                "event_time": 1100.0,
                "duration": 0.2,
                "peak_frequency": 20.0,
                "q_value": 4.0,
                "context_duration": 4.0,
                "available_ifos": ["H1", "L1"],
                "detector_availability": [1, 1, 0],
                "network_strain_sources": sources,
            }
        )
        + "\n"
    )

    def fake_download(url, path, workers):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(url.encode())
        return {"path": str(path)}

    monkeypatch.setattr("gwyolo.gravityspy.download_resumable", fake_download)
    monkeypatch.setattr(
        "gwyolo.gravityspy.verify_hdf5_against_detail",
        lambda path, detail, chunk_samples: {
            "passed": True,
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": Path(path).stat().st_size,
        },
    )
    monkeypatch.setattr("gwyolo.gravityspy._api_json", lambda _: {})
    monkeypatch.setattr(
        "gwyolo.gravityspy.read_hdf5_segment",
        lambda *_args: {
            "strain": np.sin(np.linspace(0, 20, 256)) * 1e-21,
            "sample_rate": 64,
            "quality": {
                "DQmask": np.ones(4, dtype=np.int64),
                "Injmask": np.zeros(4, dtype=np.int64),
            },
        },
    )
    report = materialize_gravityspy_network_strain(
        plan, config, tmp_path / "cache", tmp_path / "output", output_duration=2.0
    )
    row = json.loads(Path(report["manifest_path"]).read_text().strip())
    Path(row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="sample changed"):
        materialize_gravityspy_network_strain(
            plan, config, tmp_path / "cache", tmp_path / "output", output_duration=2.0
        )


def test_gravityspy_network_recovery_plan_selects_only_verified_rejections(
    tmp_path,
) -> None:
    source = tmp_path / "source.jsonl"
    source_rows = [
        {"glitch_id": "accepted", "network_strain_shard": 0, "split": "train"},
        {"glitch_id": "retry", "network_strain_shard": 0, "split": "train"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    output = tmp_path / "completed"
    output.mkdir()
    sample = output / "accepted.npz"
    np.savez(sample, features=np.asarray([1], dtype=np.float32))
    record = {
        **source_rows[0],
        "path": str(sample),
        "sha256": file_sha256(sample),
    }
    manifest = output / "gravityspy_network_numeric_manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    identity = {
        "source_manifest_sha256": file_sha256(source),
        "config_hash": "config",
        "shard": 0,
    }
    partial = output / "materialization_partial.json"
    partial.write_text(
        json.dumps(
            {
                "run_identity": identity,
                "verified_sources": {},
                "records": [record],
                "rejected": [
                    {
                        "glitch_id": "retry",
                        "reason": "fewer_than_two_usable_detectors",
                    }
                ],
            }
        )
    )
    report = output / "gravityspy_network_numeric_report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_gravityspy_aligned_network_numeric_weak_masks",
                "run_identity": identity,
                "manifest_path": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "rows": 1,
                "shard": 0,
                "requested_rows": 2,
                "rejected_rows": 1,
            }
        )
    )
    state = output / "materialization_state.json"
    state.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_identity": identity,
                "completed_rows": 1,
                "rejected_rows": 1,
                "requested_rows": 2,
                "report_sha256": file_sha256(report),
            }
        )
    )
    recovery = plan_gravityspy_network_recovery(
        source, [report], tmp_path / "recovery"
    )
    assert recovery["source_rows_accounted"] == 2
    assert recovery["recovery_rows"] == recovery["unique_recovery_glitches"] == 1
    assert recovery["adds_independent_physical_examples"] is False
    recovered = json.loads(Path(recovery["manifest_path"]).read_text().strip())
    assert recovered["glitch_id"] == "retry"
    assert recovered["recovery_parent_shard"] == 0


def test_gravityspy_network_shards_keep_shared_sources_together(tmp_path) -> None:
    manifest = tmp_path / "network-plan.jsonl"
    rows = []
    source_pairs = [
        ("h1-a", "l1-a"),
        ("h1-a", "l1-b"),
        ("h1-c", "l1-c"),
    ]
    for index, (h1, l1) in enumerate(source_pairs):
        rows.append(
            {
                "glitch_id": f"g-{index}",
                "network_strain_sources": {
                    "H1": {"hdf5_url": h1},
                    "L1": {"hdf5_url": l1},
                },
            }
        )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = shard_gravityspy_network_strain_plan(
        manifest, tmp_path / "shards", files_per_shard=3, seed=4
    )
    assert report["rows"] == 3
    assert report["unique_source_files"] == 5
    assert report["connected_components"] == 2
    assert report["shards"] == 2
    assert report["all_source_files_assigned_once"]
    sharded = [
        json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()
    ]
    by_glitch = {row["glitch_id"]: row["network_strain_shard"] for row in sharded}
    assert by_glitch["g-0"] == by_glitch["g-1"]
    assert by_glitch["g-2"] != by_glitch["g-0"]


def test_network_source_selection_adds_new_gps_components_by_label_deficit(
    tmp_path,
) -> None:
    def row(glitch_id, label, block, first, second):
        return {
            "glitch_id": glitch_id,
            "split": "train",
            "ml_label": label,
            "network_gps_block": block,
            "observing_run": "O3a",
            "ifo": "H1",
            "available_ifos": ["H1", "L1"],
            "network_strain_sources": {
                "H1": {"hdf5_url": first},
                "L1": {"hdf5_url": second},
            },
        }

    existing = tmp_path / "existing.jsonl"
    existing.write_text(
        json.dumps(row("old-a", "A", "old-block", "old-h", "old-l")) + "\n"
    )
    candidates = [
        row("new-a", "A", "block-a", "a-h", "a-l"),
        row("new-b1", "B", "block-b1", "b-h", "b-l"),
        row("new-b2", "B", "block-b2", "b-h", "b-l"),
        row("same-old-block", "B", "old-block", "c-h", "c-l"),
        row("same-old-source", "B", "block-c", "old-h", "d-l"),
    ]
    manifest = tmp_path / "network.jsonl"
    manifest.write_text("".join(json.dumps(value) + "\n" for value in candidates))
    report = select_gravityspy_network_source_components(
        manifest,
        tmp_path / "selected",
        per_label=2,
        maximum_source_files=4,
        seed=4,
        existing_manifest_path=existing,
    )
    assert report["target_met"]
    assert report["selected_source_files"] == 4
    assert report["selected_unique_glitches"] == 3
    assert report["selected_unique_network_gps_blocks"] == 3
    assert report["combined_label_counts"] == {"A": 2, "B": 2}
    selected = {
        json.loads(line)["glitch_id"]
        for line in Path(report["manifest_path"]).read_text().splitlines()
    }
    assert selected == {"new-a", "new-b1", "new-b2"}


def test_network_numeric_merge_requires_aligned_hash_verified_rows(tmp_path) -> None:
    reports = []
    for index in range(2):
        sample = tmp_path / f"network-{index}.npz"
        np.savez(sample, detector_availability=np.asarray([1, 1, 0]))
        manifest = tmp_path / f"network-{index}.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "glitch_id": f"g-{index}",
                    "split": "val",
                    "network_gps_block": f"block-{index}",
                    "ml_label": "Blip" if index == 0 else "Koi_Fish",
                    "observing_run": "O2" if index == 0 else "O3a",
                    "ifo": "H1" if index == 0 else "L1",
                    "available_ifos": ["H1", "L1"],
                    "network_strain_sources": {
                        "H1": {"hdf5_url": f"https://example/H1-{index}.hdf5"},
                        "L1": {"hdf5_url": f"https://example/L1-{index}.hdf5"},
                    },
                    "aligned_network_context": True,
                    "human_pixel_mask": False,
                    "path": str(sample),
                    "sha256": file_sha256(sample),
                }
            )
            + "\n"
        )
        report = tmp_path / f"network-report-{index}.json"
        report.write_text(
            json.dumps(
                {
                    "status": "verified_gravityspy_aligned_network_numeric_weak_masks",
                    "manifest_path": str(manifest),
                    "manifest_sha256": file_sha256(manifest),
                    "rows": 1,
                    "shard": index,
                }
            )
        )
        reports.append(report)
    merged = merge_gravityspy_network_numeric_manifests(
        reports, tmp_path / "merged-network", "val"
    )
    assert merged["rows"] == merged["unique_glitch_ids"] == 2
    assert merged["detector_subset_counts"] == {"H1L1": 2}
    assert merged["labels"] == {"Blip": 1, "Koi_Fish": 1}
    assert merged["runs"] == {"O2": 1, "O3a": 1}
    assert merged["event_ifos"] == {"H1": 1, "L1": 1}
    assert merged["available_ifos"] == {"H1": 2, "L1": 2}
    assert merged["unique_source_files"] == 4
    assert merged["weak_masks"] == 2
    assert merged["network_coherence_claim_allowed"] is False


def test_gravityspy_numeric_merge_verifies_unique_split_rows(tmp_path) -> None:
    reports = []
    for index in range(2):
        sample = tmp_path / f"sample-{index}.npz"
        np.savez(sample, features=np.asarray([index], dtype=np.float32))
        manifest = tmp_path / f"manifest-{index}.jsonl"
        row = {
            "glitch_id": f"g-{index}",
            "split": "train",
            "path": str(sample),
            "sha256": file_sha256(sample),
            "network_gps_block": f"block-{index}",
            "ml_label": "Blip",
            "observing_run": "O3a",
            "ifo": "H1",
            "human_pixel_mask": False,
        }
        manifest.write_text(json.dumps(row) + "\n")
        report = tmp_path / f"report-{index}.json"
        report.write_text(
            json.dumps(
                {
                    "status": "verified_gravityspy_numeric_weak_masks",
                    "manifest_path": str(manifest),
                    "manifest_sha256": file_sha256(manifest),
                    "rows": 1,
                }
            )
        )
        reports.append(report)
    result = merge_gravityspy_numeric_manifests(reports, tmp_path / "merged", "train")
    assert "exact_command" in result and "environment" in result
    assert result["rows"] == result["unique_glitch_ids"] == 2
    assert result["weak_masks"] == 2
    assert result["human_pixel_masks"] == 0
    assert result["labels"] == {"Blip": 2}


def test_gravityspy_source_selection_fills_deficits_with_whole_files(tmp_path) -> None:
    plan = tmp_path / "plan.jsonl"
    rows = []
    specifications = {
        "a": ["Blip", "Blip", "Blip"],
        "b": ["Tomte", "Tomte", "Blip"],
        "c": ["Koi_Fish", "Koi_Fish"],
    }
    for source, labels in specifications.items():
        for index, label in enumerate(labels):
            rows.append(
                {
                    "glitch_id": f"{source}-{index}",
                    "split": "train",
                    "ml_label": label,
                    "network_gps_block": f"block-{source}-{index}",
                    "observing_run": "O3a",
                    "ifo": "H1",
                    "strain_source": {"hdf5_url": f"https://example/{source}.hdf5"},
                }
            )
    plan.write_text("".join(json.dumps(row) + "\n" for row in rows))
    existing = tmp_path / "existing.jsonl"
    existing.write_text(json.dumps({**rows[0], "path": "unused.npz"}) + "\n")
    report = select_gravityspy_source_files(
        plan,
        tmp_path / "selection",
        per_label=2,
        maximum_files=2,
        existing_manifest_path=existing,
    )
    assert "exact_command" in report and "environment" in report
    assert report["target_met"]
    assert report["selected_source_files"] == 2
    assert report["selected_rows"] == 5
    assert report["combined_label_counts"] == {"Blip": 2, "Koi_Fish": 2, "Tomte": 2}
    selected = [
        json.loads(line)
        for line in Path(report["manifest_path"]).read_text().splitlines()
    ]
    assert {row["strain_source"]["hdf5_url"] for row in selected} == {
        "https://example/b.hdf5",
        "https://example/c.hdf5",
    }


def test_gravityspy_source_eviction_requires_verified_numeric_output(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    source = cache / "source.hdf5"
    source.write_bytes(b"official source")
    sample = tmp_path / "sample.npz"
    np.savez(sample, features=np.asarray([1], dtype=np.float32))
    manifest = tmp_path / "gravityspy_numeric_manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(sample), "sha256": file_sha256(sample)}) + "\n"
    )
    identity = {"shard": 1}
    report = tmp_path / "gravityspy_numeric_report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_gravityspy_numeric_weak_masks",
                "manifest_path": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "rows": 1,
                "verified_files": 1,
                "run_identity": identity,
            }
        )
    )
    (tmp_path / "materialization_partial.json").write_text(
        json.dumps(
            {
                "run_identity": identity,
                "verified_sources": {
                    "https://gwosc/source.hdf5": {
                        "path": str(source),
                        "sha256": file_sha256(source),
                    }
                },
            }
        )
    )
    result = evict_gravityspy_verified_sources(report, cache, tmp_path / "eviction.json")
    assert result["status"] == "complete"
    assert result["evicted_files"] == 1
    assert result["evicted_bytes"] == len(b"official source")
    assert not source.exists()
    assert sample.exists()


def test_glitch_strain_match_requires_full_context_in_one_file() -> None:
    records = [{"gps_start": 1000, "duration": 100, "hdf5_url": "example"}]
    assert match_glitch_to_strain_file(1050, records, 64) == records[0]
    assert match_glitch_to_strain_file(1010, records, 64) is None
    assert match_glitch_to_strain_file(1090, records, 64) is None


def test_glitch_strain_shards_never_split_a_source_file(tmp_path) -> None:
    manifest = tmp_path / "plan.jsonl"
    rows = []
    for file_index in range(5):
        for glitch_index in range(2):
            rows.append(
                {
                    "glitch_id": f"g-{file_index}-{glitch_index}",
                    "network_gps_block": f"b-{file_index}-{glitch_index}",
                    "event_time": 1000 + file_index * 10 + glitch_index,
                    "ml_label": "Blip",
                    "observing_run": "O3a",
                    "ifo": "H1",
                    "strain_source": {"hdf5_url": f"https://example/{file_index}.hdf5"},
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = shard_gravityspy_strain_plan(manifest, tmp_path / "out", files_per_shard=2)
    assert "exact_command" in report and "environment" in report
    assert report["shards"] == 3
    assert report["all_rows_preserved"]
    output = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assignments = {}
    for row in output:
        assignments.setdefault(row["strain_source"]["hdf5_url"], set()).add(
            row["strain_shard"]
        )
    assert all(len(shards) == 1 for shards in assignments.values())
    assert [row["rows"] for row in report["shard_summaries"]] == [4, 4, 2]


def test_gravityspy_split_keeps_network_gps_blocks_together(tmp_path) -> None:
    manifest = tmp_path / "anchors.jsonl"
    rows = []
    for block in range(40):
        for ifo in ("H1", "L1"):
            rows.append(
                {
                    "glitch_id": f"{ifo}-{block}",
                    "gravityspy_id": f"{ifo}-{block}",
                    "ifo": ifo,
                    "observing_run": "O3a",
                    "event_time": 1000000000.0 + block * 64,
                    "ml_label": "Blip",
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = split_gravityspy_anchors(manifest, tmp_path / "split", seed=9)
    assert report["passed"]
    assert all(
        count == 0
        for pair in report["cross_split_overlaps"].values()
        for count in pair.values()
    )
    assignments = {}
    for split in ("train", "val", "test"):
        for line in Path(report["manifests"][split]["path"]).read_text().splitlines():
            row = json.loads(line)
            assignments.setdefault(row["network_gps_block"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in assignments.values())


def test_gravityspy_index_filters_and_groups(tmp_path) -> None:
    path = tmp_path / "H1_O1.csv"
    fieldnames = [
        "event_time",
        "ifo",
        "duration",
        "peak_frequency",
        "snr",
        "q_value",
        "gravityspy_id",
        "ml_label",
        "ml_confidence",
        "url1",
        "url2",
        "url3",
        "url4",
    ]
    rows = [
        ["100.5", "H1", "1", "80", "10", "8", "a", "Blip", "0.95", "1", "2", "3", "4"],
        ["110.5", "H1", "1", "90", "11", "9", "b", "Blip", "0.99", "1", "2", "3", "4"],
        ["200.5", "H1", "2", "40", "12", "5", "c", "Chirp", "0.99", "1", "2", "3", "4"],
        ["300.5", "H1", "2", "50", "13", "6", "d", "Tomte", "0.70", "1", "2", "3", "4"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    selected, report = index_gravityspy_csv(path, "H1_O1.csv", 0.9, 10, 1)
    assert {item["gravityspy_id"] for item in selected} == {"a", "b"}
    assert {item["gps_block"] for item in selected} == {"H1:64:64"}
    assert report["raw_rows"] == 4
    assert report["selected_rows"] == 2
    assert report["unique_gps_blocks"] == 1
