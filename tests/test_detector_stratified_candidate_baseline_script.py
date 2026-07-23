from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "run_detector_stratified_candidate_baseline.sh"


def test_detector_stratified_baseline_is_validation_only_and_score_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = [
        "candidate-search-validation-pipeline",
        "candidate-search-validation-detector-set-recalibrate",
        "detector_stratified_candidate_calibration.yaml",
        "within_block_gps_order_v1",
        "publication_calibration_eligible",
        "variable_detector_set_block_permutation",
        "final_search_far_claim_allowed",
        "test_rows_read",
    ]
    for token in required:
        assert token in source
    assert "--target-far-per-year 0.1" not in source
    assert "--split test" not in source
    assert "candidate_scores_inspected" in source


def test_detector_stratified_policy_predeclares_robustness_exposure() -> None:
    policy = (
        ROOT / "configs" / "detector_stratified_candidate_calibration.yaml"
    ).read_text(encoding="utf-8")
    assert "target_far_per_year: 1000.0" in policy
    assert "zero_count_confidence: 0.90" in policy
    assert "exposure_safety_factor: 1.5" in policy
    for subset in ("H1+L1", "H1+V1", "L1+V1", "H1+L1+V1"):
        assert subset in policy


def test_clean_detector_set_bootstrap_gate_requires_noninferiority() -> None:
    source = (
        ROOT / "scripts" / "run_detector_set_clean_bootstrap_gate.sh"
    ).read_text(encoding="utf-8")
    assert "minimum_clean_chirp_iou_retention" in source
    assert "training_dropout_probability" in source
    assert "minimum_available_detectors" in source
    assert "test_rows_read" in source
    assert "failed non-inferiority" in source
