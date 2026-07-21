from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_background_acquisition_range.sh"


def test_background_acquisition_range_is_retryable_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "gwosc-plan-shard" in source
    assert "gwosc-batch-download" in source
    assert "background-batch-plan" in source
    assert "hash_threshold_v1" in source
    assert "exhausted retries without a batch report" in source
    assert "--verified-source-inventory" in source
    assert 'TEST_FRACTION=${TEST_FRACTION:-0}' in source


def test_background_acquisition_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 3
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_background_acquisition_rejects_nonzero_test_fraction(tmp_path: Path) -> None:
    environment = os.environ.copy()
    for name in (
        "TASK_PYTHON",
        "TASK_CODE_DIR",
        "GWYOLO_CODE_COMMIT",
        "PARENT_PLAN",
        "EVENT_EXCLUSIONS",
        "CACHE_ROOT",
        "OUTPUT_ROOT",
        "SHARD_STOP_EXCLUSIVE",
    ):
        environment[name] = str(tmp_path / name.lower())
    environment["SHARD_STOP_EXCLUSIVE"] = "10"
    environment["TEST_FRACTION"] = "0.2"
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "must keep test_fraction=0" in completed.stderr
