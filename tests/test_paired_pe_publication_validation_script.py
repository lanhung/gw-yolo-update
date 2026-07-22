from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_paired_pe_publication_validation.sh"


def test_paired_pe_publication_validation_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_paired_pe_publication_validation_is_serial_and_validation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    steps = (
        "run_promoted_paired_pe_smoke.sh",
        "run_dingo_official_native_paired_smoke.sh",
        "run_amplfi_within_backend_paired_smoke.sh",
        "run_paired_pe_portfolio_validation.sh",
    )
    positions = [source.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "PE_VALIDATION_LIMIT must be at least 100" in source
    assert "PE_BOOTSTRAP_REPLICATES must be at least 10000" in source
    assert 'summary.get("test_rows_read") != 0' in source
    assert "absolute_cross_backend_comparison_allowed" in source
    assert "--required-split test" not in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
