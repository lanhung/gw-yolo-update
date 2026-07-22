from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_family_capacity_independent_overlap.sh"


def test_family_capacity_independent_overlap_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_family_capacity_independent_overlap_rebuilds_against_final_split() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "expected one passing family-safe model branch" in source
    assert "run_independent_pe_overlap.sh" in source
    assert "training_overlap_manifest_sha256" in source
    assert "validation_glitch_manifest_sha256" in source
    assert "five_seed_stability" in source
    assert "test_rows_read" in source
    assert "required-split test" not in source
