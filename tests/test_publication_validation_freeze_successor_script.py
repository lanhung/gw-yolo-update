from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_publication_validation_freeze_successor.sh"
)


def test_publication_validation_freeze_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_publication_validation_freeze_never_opens_locked_data() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "run_publication_validation_ledger.sh" in source
    assert "locked-evaluation-suite-freeze" in source
    assert "required_passed\") != 10" in source
    assert "access_log.exists()" in source
    assert "evaluation-corpus-open-once" not in source
    assert "run_locked_search_endpoints.sh" not in source
    assert "required-split test" not in source
