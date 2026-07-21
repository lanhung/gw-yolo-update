from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_amplfi_background_acquisition.sh"


def test_amplfi_acquisition_streams_exports_and_evicts_before_merge() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    positions = [
        source.index("gwosc-batch-download"),
        source.index("background-batch-plan"),
        source.index("amplfi-background-export"),
        source.index("amplfi-background-source-evict"),
        source.index("amplfi-background-capacity-audit"),
    ]
    assert positions == sorted(positions)
    for token in (
        "--test-fraction 0",
        "hash_threshold_v1",
        "verified_capacity_ready_amplfi_training_background",
        "test_rows_read",
        "PAIRS_PER_SHARD * SHARD_COUNT != 80",
    ):
        assert token in source


def test_amplfi_acquisition_embedded_python_compiles() -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert len(snippets) == 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_amplfi_acquisition_fails_closed_without_environment() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr
