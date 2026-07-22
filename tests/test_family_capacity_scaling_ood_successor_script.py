from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_family_capacity_scaling_ood_successor.sh"
)


def test_family_capacity_successor_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_family_capacity_successor_preserves_scientific_boundaries() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "expected exactly one completed family-safe branch" in source
    assert "run_network_ood_validation.sh" in source
    assert "run_physical_overlap_data_scaling.sh" in source
    assert "run_physical_overlap_scaling_hard_endpoint.sh" in source
    assert 'if [[ "$five_seed_passed" == True ]]' in source
    assert "not_authorized_by_five_seed_gate" in source
    assert "test_rows_read" in source
    assert "FAR/IFAR/<VT>" in source
    assert "locked evaluation" in source
    assert "required-split test" not in source
