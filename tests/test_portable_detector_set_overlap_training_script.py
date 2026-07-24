from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_portable_detector_set_overlap_training.sh"
)


def test_portable_detector_set_training_is_group_safe_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "detector-set-training-bundle-import" in source
    assert "verified_portable_detector_set_training_preflight" in source
    assert 'allowed_subsets = {"H1L1", "H1L1V1", "H1V1", "L1V1"}' in source
    assert "cross_split_overlaps" in source
    assert "physical-overlap-finetune" in source
    assert 'CUDA_VISIBLE_DEVICES="$assigned_gpu"' in source
    assert "validation_selected_real_glitch_overlap_finetune" in source
    assert "checkpoint_selection_metric" in source
    assert "glitch_adapter_only" in source
    assert '"test_rows_read": 0' in source
    assert '"test_evaluation": None' in source
    assert "same_distribution_data_scaling_claim_allowed" in source
    assert "detector_complete_clean_training_authorized" in source
