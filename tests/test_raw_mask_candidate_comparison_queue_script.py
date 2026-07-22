from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_raw_mask_candidate_comparison_queue.sh"
)


def test_raw_mask_candidate_comparison_queue_is_fail_closed_and_validation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "UPSTREAM_RECEIPT",
        "UPSTREAM_PID",
        "UPSTREAM_IDENTITY",
        "candidate-search-raw-mask-compare",
        "completed_validation_only_raw_mask_candidate_background",
        "validation_only_paired_raw_mask_candidate_calibration_comparison",
        'report.get("scientific_claim_allowed") is not False',
        'report.get("locked_test_allowed") is not False',
        'report.get("locked_test_prerequisites_satisfied") is not False',
        'report.get("test_rows_read") != 0',
    ):
        assert token in source
    assert "evaluation-corpus-open-once" not in source
    assert "O4b" not in source
    assert "GWTC-5" not in source
