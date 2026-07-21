from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dingo_native_fallback.sh"


def test_native_fallback_is_authorization_bound_and_test_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "PRIMARY_FAILURE_RECEIPT",
        "COMPATIBILITY_ADJUDICATION_REPORT",
        "dingo_native_runtime_fallback_authorized",
        "EXPECTED_NATIVE_DINGO_VERSION:-0.5.8",
        "model_substitution_allowed",
        "verified_official_dingo_native_runtime_dual_model_load",
        "test_rows_read",
    ):
        assert token in source


def test_native_fallback_embedded_python_compiles() -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert len(snippets) == 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_native_fallback_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr
