from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.gravityspy import (
    _effective_network_detector_availability,
    audit_gravityspy_network_materialization_progress,
    evict_gravityspy_verified_sources,
    forecast_gravityspy_network_family_capacity,
    gravityspy_weak_mask,
    index_gravityspy_csv,
    match_glitch_to_strain_file,
    materialize_gravityspy_network_strain,
    merge_gravityspy_numeric_manifests,
    merge_gravityspy_network_numeric_manifests,
    plan_gravityspy_network_strain,
    plan_gravityspy_network_recovery,
    resplit_gravityspy_network_numeric_corpus,
    select_gravityspy_source_files,
    select_gravityspy_network_source_components,
    shard_gravityspy_network_strain_plan,
    shard_gravityspy_strain_plan,
    split_gravityspy_anchors,
)
from gwyolo.io import canonical_hash, file_sha256, load_yaml


def test_network_family_capacity_forecast_separates_floor_and_plan_ceiling(
    tmp_path,
) -> None:
    def sources(component: str) -> dict[str, dict[str, str]]:
        return {
            "H1": {"hdf5_url": f"https://example/{component}-h1.hdf5"},
            "L1": {"hdf5_url": f"https://example/{component}-l1.hdf5"},
        }

    rows = []
    family_components = {
        "A": [f"a-{index}" for index in range(5)],
        "B": [f"b-{index}" for index in range(6)],
        "C": ["c-shared"] * 6,
        "D": [f"d-{index}" for index in range(4)],
    }
    for family, components in family_components.items():
        for index, component in enumerate(components):
            sample = tmp_path / f"{family}-{index}.npz"
            np.savez(sample, identity=np.asarray([family, index]))
            rows.append(
                {
                    "glitch_id": f"{family}-{index}",
                    "ml_label": family,
                    "network_gps_block": f"block-{component}",
                    "network_strain_sources": sources(component),
                    "path": str(sample),
                    "sha256": file_sha256(sample),
                }
            )
    manifest = tmp_path / "materialized.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = tmp_path / "materialized-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_merged_gravityspy_aligned_network_numeric_split",
                "manifest_path": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "rows": len(rows),
            }
        )
    )
    pending = [
        {
            "glitch_id": "A-pending",
            "ml_label": "A",
            "network_gps_block": "block-a-pending",
            "network_strain_sources": sources("a-pending"),
            "network_strain_shard": 1,
        },
        {
            "glitch_id": "C-pending",
            "ml_label": "C",
            "network_gps_block": "block-c-pending",
            "network_strain_sources": sources("c-pending"),
            "network_strain_shard": 1,
        },
        {
            "glitch_id": "D-pending",
            "ml_label": "D",
            "network_gps_block": "block-d-pending",
            "network_strain_sources": sources("d-pending"),
            "network_strain_shard": 1,
        },
        {
            "glitch_id": "D-rejected",
            "ml_label": "D",
            "network_gps_block": "block-d-rejected",
            "network_strain_sources": sources("d-rejected"),
            "network_strain_shard": 0,
        },
    ]
    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps(row) + "\n" for row in pending))
    report_payload = json.loads(report.read_text())
    report_payload.update(
        {
            "status": "verified_gravityspy_aligned_network_numeric_weak_masks",
            "shard": 0,
            "run_identity": {"source_manifest_sha256": file_sha256(plan)},
        }
    )
    report.write_text(json.dumps(report_payload))
    config = tmp_path / "promotion.yaml"
    config.write_text(
        "overlap_sampling_promotion:\n"
        "  minimum_validation_rows_per_family: 5\n"
    )

    result = forecast_gravityspy_network_family_capacity(
        [report], [plan], config, tmp_path / "capacity.json"
    )

    assert result["families_with_current_shortfall"] == ["A", "C", "D"]
    assert result["families_impossible_even_if_all_pending_rows_are_usable"] == [
        "D"
    ]
    assert result["families"]["A"]["current"]["rows"] == 5
    assert result["accounted_rejected_rows"] == 1
    assert result["families"]["D"]["accounted_rejected_rows"] == 1
    assert result["families"]["A"]["all_pending_usable_ceiling"][
        "labelwise_group_safe_split_feasible"
    ]
    assert not result["families"]["C"]["current"][
        "labelwise_group_safe_split_feasible"
    ]
    assert result["families"]["C"]["all_pending_usable_ceiling"][
        "minimum_feasible_validation_rows"
    ] == 6
    assert not result["passed"]
    assert result["bounded_expansion_required"]
    assert not result["model_selection_authorized"]
    required_output = tmp_path / "required-capacity.json"
    with pytest.raises(ValueError, match="family capacity is not ready"):
        forecast_gravityspy_network_family_capacity(
            [report],
            [plan],
            config,
            required_output,
            require_ready=True,
        )
    assert required_output.is_file()


def test_network_materialization_progress_counts_only_completed_physical_rows(
    tmp_path,
) -> None:
    plan = tmp_path / "plan.jsonl"

    def source(url: str, ifo: str) -> dict[str, str]:
        return {"hdf5_url": url, "detector": ifo}

    planned = [
        {
            "split": "train",
            "network_strain_shard": 0,
            "glitch_id": "g0",
            "network_strain_sources": {
                "H1": source("https://example/h0.hdf5", "H1"),
                "L1": source("https://example/l0.hdf5", "L1"),
            },
        },
        {
            "split": "train",
            "network_strain_shard": 0,
            "glitch_id": "g1",
            "network_strain_sources": {
                "H1": source("https://example/h0.hdf5", "H1"),
                "L1": source("https://example/l0.hdf5", "L1"),
            },
        },
        {
            "split": "train",
            "network_strain_shard": 1,
            "glitch_id": "g2",
            "network_strain_sources": {
                "H1": source("https://example/h1.hdf5", "H1"),
                "L1": source("https://example/l1.hdf5", "L1"),
            },
        },
    ]
    plan.write_text("".join(json.dumps(row) + "\n" for row in planned))
    sample = tmp_path / "g0.npz"
    sample.write_bytes(b"verified numeric sample")
    manifest = tmp_path / "shard-0.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "glitch_id": "g0",
                "network_gps_block": "O3a:100:64",
                "observing_run": "O3a",
                "ml_label": "Blip",
                "path": str(sample),
                "sha256": file_sha256(sample),
            }
        )
        + "\n"
    )
    report = tmp_path / "shard-0-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_gravityspy_aligned_network_numeric_weak_masks",
                "shard": 0,
                "run_identity": {"source_manifest_sha256": file_sha256(plan)},
                "requested_rows": 2,
                "rows": 1,
                "rejected_rows": 1,
                "manifest_path": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "detector_subset_counts": {"H1L1": 1},
                "runtime_detector_downgraded_rows": 1,
                "rejection_reason_counts": {"event_ifo_unusable": 1},
                "verified_files": 2,
            }
        )
    )
    result = audit_gravityspy_network_materialization_progress(
        plan,
        [report],
        "train",
        2,
        tmp_path / "progress.json",
    )
    assert result["corpus_complete"] is False
    assert result["completed_shards"] == [0]
    assert result["pending_shards"] == [1]
    assert result["shard_completion_fraction"] == 0.5
    assert result["row_completion_fraction"] == 2 / 3
    assert result["usable_yield_among_accounted"] == 0.5
    assert result["usable_fraction_of_plan"] == 1 / 3
    assert result["unique_usable_glitches"] == 1
    assert result["unique_usable_network_gps_blocks"] == 1
    assert result["planned_unique_source_files"] == 4
    assert result["partial_corpus_may_select_model"] is False


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
    assert set(row["network_strain_sources"]) == {"H1", "L1"}
    assert set(row["planned_network_strain_sources"]) == {"H1", "L1", "V1"}
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


def test_network_materialization_imports_only_equivalent_verified_source_cache(
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
                "glitch_id": "imported-cache",
                "split": "val",
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
    cache = tmp_path / "cache"
    verified_sources = {}
    for ifo, source in sources.items():
        path = cache / "O2" / ifo / f"{ifo}.hdf5"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"verified-{ifo}".encode())
        observed = {
            "filesize_bytes": path.stat().st_size,
            "mean_strain": 0.0,
            "stdev_strain": 1.0,
            "min_strain": -1.0,
            "max_strain": 1.0,
            "nans_fraction": 0.0,
        }
        verified_sources[source["hdf5_url"]] = {
            "passed": True,
            "failures": [],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "detail_url": source["detail_url"],
            "expected": observed,
            "observed": observed,
            "observed_bitsums": {"0": 4},
            "strain_samples": 256,
        }
    legacy = tmp_path / "legacy_partial.json"
    legacy_payload = {
        "run_identity": {
            "code_commit": "legacy-equivalent-transport-only",
            "source_manifest_sha256": file_sha256(plan),
            "config_hash": canonical_hash(load_yaml(config)),
            "output_duration": 2.0,
            "download_workers": 8,
            "chunk_samples": 1_048_576,
            "shard": None,
        },
        "verified_sources": verified_sources,
        "records": [],
        "rejected": [],
    }
    legacy.write_text(json.dumps(legacy_payload), encoding="utf-8")

    def no_network(*_args, **_kwargs):
        raise AssertionError("verified local cache attempted network access")

    monkeypatch.setattr("gwyolo.gravityspy.download_resumable", no_network)
    monkeypatch.setattr("gwyolo.gravityspy._api_json", no_network)
    monkeypatch.setattr("gwyolo.gravityspy.verify_hdf5_against_detail", no_network)
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
        plan,
        config,
        cache,
        tmp_path / "output",
        output_duration=2.0,
        verified_source_inventories=[legacy],
    )
    assert report["rows"] == 1
    assert report["verified_files"] == 2
    assert report["imported_verified_source_inventories"] == [
        {
            "path": str(legacy),
            "sha256": file_sha256(legacy),
            "source_code_commit": "legacy-equivalent-transport-only",
            "imported_urls": sorted(source["hdf5_url"] for source in sources.values()),
        }
    ]

    tampered = tmp_path / "tampered_partial.json"
    legacy_payload["verified_sources"][sources["H1"]["hdf5_url"]]["sha256"] = "0" * 64
    tampered.write_text(json.dumps(legacy_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        materialize_gravityspy_network_strain(
            plan,
            config,
            cache,
            tmp_path / "tampered-output",
            output_duration=2.0,
            verified_source_inventories=[tampered],
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


def test_network_source_selection_targets_family_and_separate_exclusions(
    tmp_path,
) -> None:
    def row(glitch_id: str, label: str, component: str) -> dict:
        return {
            "glitch_id": glitch_id,
            "split": "train",
            "ml_label": label,
            "network_gps_block": f"block-{component}",
            "observing_run": "O3a",
            "ifo": "H1",
            "available_ifos": ["H1", "L1"],
            "network_strain_sources": {
                "H1": {"hdf5_url": f"https://example/{component}-h1.hdf5"},
                "L1": {"hdf5_url": f"https://example/{component}-l1.hdf5"},
            },
        }

    candidates = tmp_path / "target-candidates.jsonl"
    candidates.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in (
                row("h0", "Helix", "h0"),
                row("h1", "Helix", "h1"),
                row("h2", "Helix", "h2"),
                row("b0", "Blip", "b0"),
                row("b1", "Blip", "b1"),
            )
        )
    )
    exclusion = tmp_path / "exclusion.jsonl"
    exclusion.write_text(json.dumps(row("old", "Helix", "h0")) + "\n")

    result = select_gravityspy_network_source_components(
        candidates,
        tmp_path / "target-selection",
        per_label=2,
        maximum_source_files=4,
        seed=3,
        target_labels=["Helix"],
        exclusion_manifest_paths=[exclusion],
    )

    assert result["target_met"]
    assert result["target_labels"] == ["Helix"]
    assert result["selected_label_counts"] == {"Helix": 2}
    assert result["selected_source_files"] == 4
    selected = {
        json.loads(line)["glitch_id"]
        for line in Path(result["manifest_path"]).read_text().splitlines()
    }
    assert selected == {"h1", "h2"}


def test_network_numeric_merge_requires_aligned_hash_verified_rows(tmp_path) -> None:
    reports = []
    for index in range(2):
        sample = tmp_path / f"network-{index}.npz"
        np.savez(
            sample,
            detector_availability=np.asarray([1, 1, 0]),
            test_identity=np.asarray([index]),
        )
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

    # A verified merge report may itself be used as a hash-checked source. This
    # supports a strict base + independent-source union without re-listing every
    # historical shard or weakening duplicate-glitch validation.
    nested_report = tmp_path / "merged-network" / "gravityspy_network_numeric_merge_report.json"
    nested = merge_gravityspy_network_numeric_manifests(
        [nested_report], tmp_path / "nested-network", "val"
    )
    assert nested["rows"] == 2
    assert nested["manifest_sha256"] == merged["manifest_sha256"]


def test_network_numeric_merge_normalizes_legacy_runtime_downgrade(tmp_path) -> None:
    sample = tmp_path / "network.npz"
    np.savez(sample, detector_availability=np.asarray([1, 1, 0]))
    sources = {
        ifo: {"hdf5_url": f"https://example/{ifo}.hdf5"}
        for ifo in ("H1", "L1", "V1")
    }
    manifest = tmp_path / "network.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "glitch_id": "legacy-runtime-downgrade",
                "split": "train",
                "network_gps_block": "O3b:block",
                "ml_label": "Blip",
                "observing_run": "O3b",
                "ifo": "H1",
                "planned_available_ifos": ["H1", "L1", "V1"],
                "available_ifos": ["H1", "L1"],
                "network_strain_sources": sources,
                "data_quality": {
                    "H1": {"usable": True, "reason": None},
                    "L1": {"usable": True, "reason": None},
                    "V1": {"usable": False, "reason": "nonfinite_strain_context"},
                },
                "aligned_network_context": True,
                "human_pixel_mask": False,
                "path": str(sample),
                "sha256": file_sha256(sample),
            }
        )
        + "\n"
    )
    report = tmp_path / "network-report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_gravityspy_aligned_network_numeric_weak_masks",
                "manifest_path": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "rows": 1,
                "shard": 0,
            }
        )
    )
    merged = merge_gravityspy_network_numeric_manifests(
        [report], tmp_path / "merged", "train"
    )
    assert merged["runtime_source_inventory_normalized_rows"] == 1
    assert merged["unique_source_files"] == 2
    assert merged["planned_unique_source_files"] == 3
    row = json.loads(Path(merged["manifest_path"]).read_text().strip())
    assert set(row["network_strain_sources"]) == {"H1", "L1"}
    assert set(row["planned_network_strain_sources"]) == {"H1", "L1", "V1"}
    assert row["runtime_source_inventory_normalized"] is True


def test_network_numeric_resplit_keeps_source_components_disjoint(tmp_path) -> None:
    rows = []
    for index, (component, label, split) in enumerate(
        (
            ("a", "Blip", "train"),
            ("a", "Tomte", "val"),
            ("b", "Blip", "train"),
            ("b", "Tomte", "val"),
        )
    ):
        sample = tmp_path / f"sample-{index}.npz"
        np.savez(
            sample,
            detector_availability=np.asarray([1, 1, 0]),
            test_identity=np.asarray([index]),
        )
        rows.append(
            {
                "glitch_id": f"g-{index}",
                "split": split,
                "network_gps_block": f"block-{index}",
                "ml_label": label,
                "observing_run": "O3a",
                "ifo": "H1",
                "available_ifos": ["H1", "L1"],
                "network_strain_sources": {
                    "H1": {"hdf5_url": f"https://example/{component}-shared.hdf5"},
                    "L1": {"hdf5_url": f"https://example/{component}-{index}.hdf5"},
                },
                "aligned_network_context": True,
                "human_pixel_mask": False,
                "path": str(sample),
                "sha256": file_sha256(sample),
            }
        )
    reports = []
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        manifest = tmp_path / f"historical-{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
        report = tmp_path / f"historical-{split}.json"
        report.write_text(
            json.dumps(
                {
                    "status": "verified_merged_gravityspy_aligned_network_numeric_split",
                    "split": split,
                    "manifest_path": str(manifest),
                    "manifest_sha256": file_sha256(manifest),
                    "rows": len(selected),
                }
            )
        )
        reports.append(report)
    result = resplit_gravityspy_network_numeric_corpus(
        reports, tmp_path / "resplit", validation_fraction=0.25, seed=7
    )
    assert result["passed"]
    assert result["actual_validation_rows"] == 2
    assert all(not values for values in result["cross_split_overlaps"].values())
    assert result["actual_validation_label_counts"] == {"Blip": 1, "Tomte": 1}


def test_network_numeric_resplit_enforces_frozen_family_floor(tmp_path) -> None:
    rows = []
    for label in ("Blip", "Helix"):
        for index in range(6):
            sample = tmp_path / f"{label}-{index}.npz"
            np.savez(sample, identity=np.asarray([label, index]))
            rows.append(
                {
                    "glitch_id": f"{label}-{index}",
                    "split": "train" if index % 2 == 0 else "val",
                    "network_gps_block": f"{label}-block-{index}",
                    "ml_label": label,
                    "observing_run": "O3a",
                    "ifo": "H1",
                    "available_ifos": ["H1", "L1"],
                    "network_strain_sources": {
                        "H1": {
                            "hdf5_url": f"https://example/{label}-{index}-h1.hdf5"
                        },
                        "L1": {
                            "hdf5_url": f"https://example/{label}-{index}-l1.hdf5"
                        },
                    },
                    "aligned_network_context": True,
                    "human_pixel_mask": False,
                    "path": str(sample),
                    "sha256": file_sha256(sample),
                }
            )
    reports = []
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        manifest = tmp_path / f"floor-{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
        report = tmp_path / f"floor-{split}.json"
        report.write_text(
            json.dumps(
                {
                    "status": "verified_merged_gravityspy_aligned_network_numeric_split",
                    "split": split,
                    "manifest_path": str(manifest),
                    "manifest_sha256": file_sha256(manifest),
                    "rows": len(selected),
                }
            )
        )
        reports.append(report)

    result = resplit_gravityspy_network_numeric_corpus(
        reports,
        tmp_path / "floor-resplit",
        validation_fraction=0.2,
        seed=9,
        minimum_validation_rows_per_family=5,
    )

    assert result["passed"]
    assert result["minimum_validation_rows_per_family"] == 5
    assert result["actual_validation_label_counts"] == {"Blip": 5, "Helix": 5}
    train_report = json.loads(Path(result["reports"]["train"]["path"]).read_text())
    assert train_report["labels"] == {"Blip": 1, "Helix": 1}


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
