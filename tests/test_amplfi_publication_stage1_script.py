from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_amplfi_publication_stage1.sh"


def test_amplfi_stage1_is_capacity_selection_and_model_load_gated() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "verified_capacity_ready_amplfi_training_background",
        "publication_stage_1",
        "--minimum-publication-epochs 100",
        "--minimum-validation-points 50",
        "publication_eligible",
        "run_pe_model_load_smoke.py",
        "pe-backend-model-freeze",
        "verified_amplfi_publication_stage1_model",
        "test_rows_read",
    ):
        assert token in source


def test_amplfi_stage1_embedded_python_compiles() -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert len(snippets) == 3
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_amplfi_stage1_fails_closed_without_environment() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr
