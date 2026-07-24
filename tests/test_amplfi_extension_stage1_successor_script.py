from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_amplfi_extension_stage1_successor.sh"
)


def test_amplfi_extension_stage1_successor_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_amplfi_extension_stage1_successor_freezes_bank_before_training() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("amplfi-training-bank-freeze") < source.index(
        "run_amplfi_publication_stage1.sh"
    )
    assert "BACKGROUND_BANK_REPORT" in source
    assert "while kill -0" not in source
    assert "required-split test" not in source
