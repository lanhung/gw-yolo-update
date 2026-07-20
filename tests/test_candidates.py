from __future__ import annotations

import numpy as np

from gwyolo.candidates import extract_temporal_clusters


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
