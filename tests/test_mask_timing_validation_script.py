from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_mask_timing_validation.sh"


def test_mask_timing_runner_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_mask_timing_runner_binds_immutable_checkout_and_receipts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert "MASK_VALIDATION_RECEIPT" in source
    assert "PIPELINE_REPORT" in source
    assert "mask-timing-validation" in source
    assert "--required-split test" not in source
