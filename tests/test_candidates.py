from __future__ import annotations

import numpy as np

from gwyolo.candidates import build_candidate_time_slides, extract_temporal_clusters


def test_temporal_clusters_preserve_multiple_candidates_and_refine_peak() -> None:
    chirp = np.zeros((2, 1, 1, 8), dtype=np.float32)
    glitch = np.zeros_like(chirp)
    chirp[0, 0, 0] = [0.0, 0.1, 0.6, 0.9, 0.2, 0.0, 0.7, 0.0]
    chirp[1, 0, 0] = [0.0, 0.0, 0.0, 0.8, 0.7, 0.0, 0.0, 0.0]
    glitch[0, 0, 0, 3] = 0.25
    rows = extract_temporal_clusters(chirp, glitch, ["H1", "L1"], 1000, 8, 0.5)
    assert len(rows) == 3
    h1 = [row for row in rows if row["ifo"] == "H1"]
    assert len(h1) == 2
    assert h1[0]["start_bin"] == 2
    assert h1[0]["stop_bin_exclusive"] == 4
    assert h1[0]["peak_bin"] == 3
    assert h1[0]["gps_start"] == 1002
    assert h1[0]["gps_end"] == 1004
    assert h1[0]["chirp_glitch_margin"] == np.float32(0.9) - np.float32(0.25)
    assert 1003.0 < h1[0]["gps_peak"] < 1004.0
    assert h1[0]["timing_uncertainty_floor_seconds"] == 0.5


def test_candidate_time_slides_use_all_candidates_but_cluster_network_events() -> None:
    windows = [
        {
            "window_id": f"w{index}",
            "split": "val",
            "gps_start": index * 8,
            "gps_end": (index + 1) * 8,
            "gps_block": f"b{index}",
        }
        for index in range(3)
    ]
    candidates = [
        {
            "candidate_id": "h1-a",
            "window_id": "w0",
            "split": "val",
            "ifo": "H1",
            "gps_peak": 1.0,
            "chirp_score": 0.8,
            "glitch_score_at_peak": 0.1,
            "bin_width_seconds": 0.01,
        },
        {
            "candidate_id": "h1-b",
            "window_id": "w0",
            "split": "val",
            "ifo": "H1",
            "gps_peak": 1.02,
            "chirp_score": 0.6,
            "glitch_score_at_peak": 0.1,
            "bin_width_seconds": 0.01,
        },
        {
            "candidate_id": "l1",
            "window_id": "w1",
            "split": "val",
            "ifo": "L1",
            "gps_peak": 9.005,
            "chirp_score": 0.7,
            "glitch_score_at_peak": 0.2,
            "bin_width_seconds": 0.01,
        },
    ]
    rows, report = build_candidate_time_slides(
        candidates,
        windows,
        "val",
        "H1",
        "L1",
        slide_count=1,
        step_seconds=8,
        coincidence_window_seconds=0.03,
        cluster_window_seconds=0.1,
    )
    assert report["equivalent_live_time_seconds"] == 16
    assert report["slide_exposure"][0]["raw_coincidences"] == 2
    assert report["slide_exposure"][0]["clustered_candidates"] == 1
    assert len(rows) == 1
    assert rows[0]["ranking_score"] == 0.7
    assert rows[0]["peak_separation_seconds"] < 0.01
