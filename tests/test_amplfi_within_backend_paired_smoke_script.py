from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_amplfi_within_backend_paired_smoke.sh"
)


def test_amplfi_within_backend_smoke_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_amplfi_within_backend_smoke_is_strict_validation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "paired_pe_native_inputs_smoke_complete" in source
    assert "amplfi-common-batch" in source
    assert source.count("--required-split val") == 1
    assert "pe-robustness-evaluate" in source
    assert "--within-backend-only" in source
    assert "within_backend_provenance_gate" in source
    assert "bootstrap_replicates" in source
    assert "evaluation_tier" in source
    assert "minimum_publication_validation_injections" in source
    assert "test_rows_read" in source
    assert "pe-robustness-joint-evaluate" not in source
    assert "dingo-common-batch" not in source
    assert "--required-split test" not in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
