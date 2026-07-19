import math

from gwyolo.search import (
    calibrate_threshold,
    compare_search_methods,
    evaluate_search,
    far_upper_limit_zero_count,
    paired_vt_comparison,
)


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
    assert result["injections"]["efficiency_wilson_95"][0] < 2 / 3


def test_paired_vt_delta_matches_hand_calculation():
    result = paired_vt_comparison(
        [
            {"raw": 10, "clean": 6, "vt_weight": 2, "stratum": "overlap"},
            {"raw": 7, "clean": 7, "vt_weight": 1, "stratum": "overlap"},
            {"raw": 9, "clean": 3, "vt_weight": 3, "stratum": "clean"},
        ],
        threshold_a=8,
        threshold_b=5,
        score_field_a="raw",
        score_field_b="clean",
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["method_a"]["recovered_vt"] == 5
    assert result["method_b"]["recovered_vt"] == 3
    assert result["delta_recovered_vt_b_minus_a"] == -2
    assert result["relative_delta"] == -2 / 5
    assert result["strata"]["overlap"]["weighted_efficiency_b"] == 1.0


def test_search_comparison_calibrates_each_method_on_validation_only():
    validation = [
        {"raw": 9, "clean": 6},
        {"raw": 8, "clean": 5},
        {"raw": 7, "clean": 4},
        {"raw": 1, "clean": 1},
    ]
    result = compare_search_methods(
        validation,
        [{"raw": 9, "clean": 6}, {"raw": 7, "clean": 4}],
        [
            {"raw": 10, "clean": 6, "vt_weight": 2},
            {"raw": 7, "clean": 7, "vt_weight": 1},
            {"raw": 9, "clean": 3, "vt_weight": 3},
        ],
        validation_live_time_years=10,
        test_live_time_years=20,
        target_far_per_year=0.2,
        score_field_a="raw",
        score_field_b="clean",
        bootstrap_replicates=20,
        seed=1,
    )
    assert result["calibrations"]["raw"]["threshold"] == 8
    assert result["calibrations"]["clean"]["threshold"] == 5
    assert result["test_evaluations"]["raw"]["background"]["far_per_year"] == 0.05
    assert result["test_evaluations"]["clean"]["background"]["far_per_year"] == 0.05
