from __future__ import annotations

import csv

from gwyolo.gravityspy import index_gravityspy_csv


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
