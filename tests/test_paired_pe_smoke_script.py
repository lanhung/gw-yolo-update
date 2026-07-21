from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_paired_pe_smoke_fails_before_work_when_paths_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/run_paired_pe_smoke.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "GWYOLO_PYTHON" in completed.stderr
