from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_locked_pe_endpoints.sh"


def test_locked_pe_endpoints_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_locked_pe_endpoints_uses_predeclared_portfolio_output() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"paired_pe_portfolio"' in source
    assert "pe-robustness-portfolio-evaluate-locked" in source
    assert "pe-robustness-joint-evaluate-locked" not in source
    assert "pe-backend-bind-locked" in source
    assert "VALIDATION_PE_PROMOTION_REPORT" in source
