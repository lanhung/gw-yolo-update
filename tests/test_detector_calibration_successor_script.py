from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "queue_detector_stratified_calibration_robustness.sh"
)


def test_detector_calibration_successor_freezes_plan_after_physical_audit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert source.index("PHYSICAL_MATERIALIZATION_AUDIT") < source.index(
        "calibration-perturbation-plan-freeze"
    )
    assert source.index("calibration-perturbation-plan-freeze") < source.index(
        "queue_calibration_robustness_validation.sh"
    )
    for token in (
        '"publication_calibration_eligible"',
        '"candidate_scores_inspected"',
        '"test_rows_read"',
        "H1+L1+V1",
        "counts[subset] < 25",
    ):
        assert token in source
    assert "evaluation-corpus-open-once" not in source
    assert "O4b" not in source
