import math

from gwyolo.search import calibrate_threshold, evaluate_search, far_upper_limit_zero_count


def test_zero_count_far_limit_is_poisson_2p3_over_time():
    assert math.isclose(far_upper_limit_zero_count(10.0), 0.2302585093, rel_tol=1e-9)


def test_threshold_is_calibrated_only_from_background():
    result = calibrate_threshold([9.0, 8.0, 7.0, 1.0], live_time_years=10.0, target_far_per_year=0.2)
    assert result["threshold"] == 8.0
    assert result["background_count"] == 2
    assert result["far_per_year"] == 0.2


def test_search_evaluation_reports_weighted_vt():
    result = evaluate_search(
        threshold=8.0,
        background_scores=[9.0, 7.0],
        background_live_time_years=20.0,
        injections=[
            {"ranking_score": 10.0, "vt_weight": 2.0},
            {"ranking_score": 7.0, "vt_weight": 1.0},
            {"ranking_score": 9.0, "vt_weight": 3.0},
        ],
    )
    assert result["background"]["far_per_year"] == 0.05
    assert result["injections"]["recovered"] == 2
    assert result["injections"]["recovered_vt"] == 5.0
    assert result["injections"]["weighted_efficiency"] == 5.0 / 6.0
