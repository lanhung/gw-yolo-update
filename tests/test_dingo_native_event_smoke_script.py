from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/run_dingo_official_native_event_smoke.sh"


def test_native_dingo_event_smoke_fails_closed_without_environment() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_native_dingo_event_smoke_is_synthetic_and_test_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "synthetic_zero_event.hdf5" in source
    assert "synthetic_runtime_smoke_only" in source
    assert "DINGO 0.5.8" in source
    assert "--num-samples 4" in source
    assert "--num-gnpe-iterations 1" in source
    assert '"scientific_claim_allowed": False' in source
    assert '"test_rows_read": 0' in source
    assert "--required-split test" not in source


def test_native_dingo_event_smoke_embedded_python_compiles() -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert len(snippets) == 2
    for snippet in snippets:
        completed = subprocess.run(
            [sys.executable, "-c", f"compile({snippet!r}, '<embedded>', 'exec')"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
