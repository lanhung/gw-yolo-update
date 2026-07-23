from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_detector_validation_acquisition_metadata.sh"
)


def test_detector_acquisition_metadata_is_preaccess_and_score_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "gwosc-run-plan",
        "detector-validation-acquisition-plan",
        "FROZEN_TRAIN_MANIFEST",
        "FROZEN_VALIDATION_MANIFEST",
        "--detectors H1 V1",
        "--detectors L1 V1",
        "--exclude-plan",
        '"candidate_scores_inspected"',
        '"test_data_opened"',
    ):
        assert token in source
    assert "gwosc-batch-download" not in source
    assert "O4b" not in source
    assert "evaluation-corpus-open-once" not in source
