from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_network_ood_endpoint_bind.sh"


def test_network_ood_endpoint_bind_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_network_ood_endpoint_bind_uses_exact_checkout_and_no_test_data() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert "detector-set-ood-validation-bind" in source
    assert "SOURCE_OOD_RECEIPT" in source
    assert "GRAVITYSPY_CORPUS_AUDIT" in source
    assert "--required-split test" not in source
