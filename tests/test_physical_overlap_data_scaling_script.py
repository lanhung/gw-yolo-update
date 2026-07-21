from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_physical_overlap_data_scaling.sh"


def test_overlap_data_scaling_script_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_overlap_data_scaling_script_runs_both_controls_and_never_test() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "physical-overlap-scale-subsets" in source
    assert "physical-overlap-scale-summarize" in source
    assert "fixed_epochs fixed_optimizer_updates" in source
    assert "exactly five seeds" in source
    assert "--include-full" in source
    assert "--required-split test" not in source
    assert "test_rows_read" in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
