from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_detector_validation_data_plan.sh"
)


def test_detector_validation_data_plan_is_score_blind_and_preaccess() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "detector-validation-background-export",
        "detector-validation-injection-plan",
        "MINIMUM_PER_DETECTOR_SUBSET",
        "INJECTIONS_PER_DETECTOR_SUBSET",
        '"candidate_scores_inspected"',
        '"physical_signal_projection_required"',
        '"test_rows_read"',
    ):
        assert token in source
    assert "trigger-score" not in source
    assert "injection-score" not in source
    assert "evaluation-corpus-open-once" not in source
