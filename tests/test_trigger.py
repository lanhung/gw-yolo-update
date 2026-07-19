from __future__ import annotations

import numpy as np

from gwyolo.trigger import network_ranking, probability_summaries


def test_network_ranking_uses_second_loudest_valid_ifo() -> None:
    result = network_ranking(
        {"H1": 0.9, "L1": 0.8, "V1": 0.2},
        {"H1": 0.1, "L1": 0.3, "V1": 0.7},
        ["H1", "L1"],
    )
    assert result["ranking_score"] == 0.8
    assert result["maximum_glitch_score"] == 0.3
    assert result["chirp_glitch_margin"] == 0.5
    assert result["network_mode"] == "coincident"


def test_single_ifo_ranking_is_explicitly_diagnostic() -> None:
    result = network_ranking({"H1": 0.7}, {"H1": 0.2}, ["H1"])
    assert result["ranking_score"] == 0.7
    assert result["network_mode"] == "single_ifo_diagnostic"


def test_probability_summary_peak_time_and_network_score_by_hand() -> None:
    probabilities = np.zeros((2, 2, 1, 1, 4), dtype=np.float32)
    probabilities[0, 0, 0, 0] = [0.1, 0.8, 0.2, 0.1]
    probabilities[0, 1, 0, 0] = [0.1, 0.2, 0.7, 0.1]
    probabilities[1, 0, 0, 0] = [0.3, 0.1, 0.1, 0.1]
    probabilities[1, 1, 0, 0] = [0.1, 0.4, 0.1, 0.1]
    result = probability_summaries(
        probabilities, ("H1", "L1"), ["H1", "L1"], 1000.0, 8.0
    )
    assert result["ranking_score"] == np.float32(0.7)
    assert result["peak_times"]["chirp"]["H1"]["gps"] == 1003.0
    assert result["peak_times"]["chirp"]["L1"]["gps"] == 1005.0
