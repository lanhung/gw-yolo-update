from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_calibration_robustness_validation.sh"
)


def test_calibration_robustness_runner_is_fixed_threshold_and_validation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "calibration-timing-transfer-compatibility-audit",
        "calibration-perturbation-plan",
        "calibration-timing-compatibility-report",
        "candidate-block-permutations",
        "calibration-perturbation-scenario-freeze",
        "calibration-perturbation-evaluate",
        "BASELINE_CALIBRATION_REPORT",
        '"scenario_threshold_refits"',
        '"test_rows_read"',
        "--required-split val",
        "--scenario-receipt",
    ):
        assert token in source
    assert "candidate-search-calibrate" not in source
    assert "evaluation-corpus-open-once" not in source
    assert "--required-split test" not in source
