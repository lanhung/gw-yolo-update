from __future__ import annotations

from gwyolo.trigger import network_ranking


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
