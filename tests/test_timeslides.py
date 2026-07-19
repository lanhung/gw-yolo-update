from __future__ import annotations

import pytest

from gwyolo.timeslides import build_window_time_slides


def _row(index: int) -> dict:
    start = 1000 + 8 * index
    row = {
        "window_id": f"w{index}",
        "split": "val",
        "gps_start": start,
        "gps_end": start + 8,
        "gps_block": f"b{index}",
        "valid_ifos": ["H1", "L1"],
        "chirp_scores": {"H1": 0.1 + index / 10, "L1": 0.4 + index / 10},
        "glitch_scores": {"H1": 0.05, "L1": 0.06},
    }
    row["peak_times"] = {
        "chirp": {
            "H1": {"gps": start + 4.00},
            "L1": {"gps": start + 4.01},
        }
    }
    return row


def test_noncyclic_window_slides_rank_and_exposure_by_hand() -> None:
    rows, report = build_window_time_slides(
        [_row(0), _row(1), _row(2)], "val", "H1", "L1", 2, 8
    )
    assert len(rows) == 3
    assert report["equivalent_live_time_seconds"] == 24
    assert report["slide_exposure"][0]["coincident_windows"] == 2
    assert report["slide_exposure"][1]["coincident_windows"] == 1
    first = rows[0]
    assert first["source_window_ids"] == {"H1": "w0", "L1": "w1"}
    assert first["ranking_score"] == pytest.approx(min(0.1, 0.5))
    assert first["offset_seconds"] == {"H1": 0.0, "L1": 8}


def test_window_slides_reject_subwindow_step_and_missing_ifo() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        build_window_time_slides([_row(0)], "val", "H1", "L1", 1, 1)
    row = _row(0)
    row["valid_ifos"] = ["H1"]
    with pytest.raises(ValueError, match="lack required"):
        build_window_time_slides([row], "val", "H1", "L1", 1, 8)


def test_peak_coincidence_filters_candidates_without_reducing_exposure() -> None:
    rows = [_row(0), _row(1)]
    rows[1]["peak_times"]["chirp"]["L1"]["gps"] += 0.2
    candidates, report = build_window_time_slides(
        rows, "val", "H1", "L1", 1, 8, coincidence_window_seconds=0.05
    )
    assert candidates == []
    assert report["equivalent_live_time_seconds"] == 8
    assert report["slide_exposure"][0]["coincident_candidates"] == 0
