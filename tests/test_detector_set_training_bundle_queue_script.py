from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "queue_detector_set_training_bundle.sh"
)


def test_detector_set_bundle_queue_is_fail_closed_and_portable() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'while [[ ! -s "$OVERLAP_RECEIPT" ]]' in source
    assert "detector_set_training_bundle_queue_upstream_incomplete" in source
    assert "detector-set-training-bundle-export" in source
    assert "--clean-train-manifest" in source
    assert "--clean-validation-manifest" in source
    assert "--pretrained-checkpoint" in source
    assert "finetune=$FINETUNE_CONFIG" in source
    assert "overlap_factory=$OVERLAP_CONFIG" in source
    assert '"test_rows_read": 0' in source
    assert "detector_complete_clean_training_authorized" in source
