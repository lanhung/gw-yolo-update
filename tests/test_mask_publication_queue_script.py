from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_mask_publication_queue.sh"


def test_mask_publication_queue_is_fail_closed_and_dependency_ordered() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "FIVE_SEED_UPSTREAM_PID",
        "INDEPENDENT_PE_UPSTREAM_PID",
        "BACKGROUND_UPSTREAM_PID",
        "run_mask_deglitch_validation.sh",
        "run_mask_timing_validation.sh",
        "coherent_background_scale_allowed",
        "RUN_MASK_BACKGROUND",
        "run_mask_conditioned_background_range.sh",
        "test_rows_read",
        "locked_test_allowed",
    ):
        assert token in source
    assert "evaluation-corpus-open-once" not in source
    assert "O4b" not in source
    assert "--test-fraction" not in source
