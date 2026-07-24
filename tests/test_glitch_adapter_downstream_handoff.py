from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_glitch_adapter_can_reach_every_primary_validation_consumer() -> None:
    direct_consumers = (
        "run_detector_stratified_candidate_baseline.sh",
        "run_numeric_raw_mask_detector_set_successor.sh",
        "run_mask_deglitch_validation.sh",
        "run_promoted_paired_pe_smoke.sh",
    )
    for name in direct_consumers:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert 'elif arm == "glitch_adapter"' in source

    resolver_consumers = (
        "run_promoted_candidate_validation.sh",
        "queue_calibration_robustness_validation.sh",
        "run_candidate_background_range.sh",
        "run_mask_conditioned_background_range.sh",
    )
    for name in resolver_consumers:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "resolve_promoted_overlap_model.sh" in source

    for name in (
        "run_post_five_seed_candidate_publication_chain.sh",
        "run_mask_publication_queue.sh",
        "queue_detector_stratified_calibration_robustness.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "ADAPTER_CONFIG" in source


def test_adapter_five_seed_receipt_can_start_candidate_validation() -> None:
    source = (
        ROOT / "scripts/run_post_five_seed_candidate_publication_chain.sh"
    ).read_text(encoding="utf-8")
    assert "completed_glitch_adapter_five_seed_gate" in source
    assert "completed_glitch_adapter_negative_one_seed" in source
    assert "completed_glitch_adapter_negative_five_seed" in source
