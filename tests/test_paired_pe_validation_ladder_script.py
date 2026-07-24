from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_paired_pe_validation_ladder.sh"


def test_paired_pe_validation_ladder_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_paired_pe_validation_ladder_is_bounded_then_publication_scale() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    steps = (
        "run_promoted_paired_pe_smoke.sh",
        "run_dingo_official_native_paired_smoke.sh",
        "run_amplfi_within_backend_paired_smoke.sh",
        "run_paired_pe_portfolio_validation.sh",
        "run_paired_pe_publication_validation.sh",
    )
    positions = [source.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "PE_SMOKE_LIMIT must be in [1, 99]" in source
    assert "PE_VALIDATION_LIMIT must be at least 100" in source
    assert "PE_BOOTSTRAP_REPLICATES must be at least 10000" in source
    assert 'prior_summary="$SMOKE_PORTFOLIO_OUTPUT_ROOT/' in source
    assert "MODEL_SELECTION_TRAIN_OVERLAP_MANIFEST" in source
    assert "MODEL_SELECTION_VALIDATION_OVERLAP_MANIFEST" in source
    assert "MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST" in source
    assert "while kill -0" not in source
    assert "--required-split test" not in source

