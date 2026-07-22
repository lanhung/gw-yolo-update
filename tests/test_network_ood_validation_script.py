from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_network_ood_validation.sh"


def test_network_ood_script_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode != 0
    assert "TASK_PYTHON" in completed.stderr


def test_network_ood_script_binds_corpus_commit_and_auxiliary_boundary() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "GRAVITYSPY_CORPUS_AUDIT" in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert "unknown_scores_used_for_selection" in source
    assert "heldout_scores_used_for_method_or_fit_selection" in source
    assert 'report.get("device") != "cuda"' in source
    assert "cannot veto a" in source
    assert "network_ood_validation_receipt.json" in source
    assert "detector-set-ood-validation-bind" in source
    assert "network_ood_validation_endpoint.json" in source
    assert "test_rows_read" in source
