from __future__ import annotations

import csv
import json
from pathlib import Path

from gwyolo.gravityspy import (
    index_gravityspy_csv,
    match_glitch_to_strain_file,
    split_gravityspy_anchors,
)


def test_glitch_strain_match_requires_full_context_in_one_file() -> None:
    records = [{"gps_start": 1000, "duration": 100, "hdf5_url": "example"}]
    assert match_glitch_to_strain_file(1050, records, 64) == records[0]
    assert match_glitch_to_strain_file(1010, records, 64) is None
    assert match_glitch_to_strain_file(1090, records, 64) is None


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
