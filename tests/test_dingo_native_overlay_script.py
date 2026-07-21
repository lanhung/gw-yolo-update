from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/setup_dingo_native_overlay.sh"


def test_dingo_native_overlay_fails_closed_without_environment() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_dingo_native_overlay_freezes_source_runtime_and_no_test_rows() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "04c8ab3ec694410ad85466ea6bfdc6aa2274ac14" in source
    assert "v0.5.8" in source
    assert "EXPECTED_NATIVE_VERSION:-0.5.8" in source
    assert "--no-deps" in source
    assert "--no-build-isolation" in source
    assert "--ignore-installed" in source
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" in source
    assert '"${base_sites[@]}"' in source
    assert "test_rows_read\": 0" in source
    assert "scientific_claim_allowed\": False" in source
    assert "torch.cuda.is_available" in source
    assert "status --porcelain" in source


def test_dingo_native_overlay_embedded_python_compiles() -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert len(snippets) == 4
    for snippet in snippets:
        completed = subprocess.run(
            [sys.executable, "-c", f"compile({snippet!r}, '<embedded>', 'exec')"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
