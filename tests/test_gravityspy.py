from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from gwyolo.gravityspy import (
    gravityspy_weak_mask,
    index_gravityspy_csv,
    match_glitch_to_strain_file,
    merge_gravityspy_numeric_manifests,
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
    assert result["rows"] == result["unique_glitch_ids"] == 2
    assert result["weak_masks"] == 2
    assert result["human_pixel_masks"] == 0
    assert result["labels"] == {"Blip": 2}


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
