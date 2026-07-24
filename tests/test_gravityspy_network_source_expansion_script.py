from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_gravityspy_network_source_expansion.sh"
)


def test_source_expansion_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_source_expansion_retries_a_transient_shard_failure(tmp_path: Path) -> None:
    for name in (
        "train-shards.jsonl",
        "val-shards.jsonl",
        "train-merge.json",
        "val-merge.json",
        "config.yaml",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    call_log = tmp_path / "calls.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >>\"$CALL_LOG\"\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": str(fake_python),
            "TRAIN_SHARD_MANIFEST": str(tmp_path / "train-shards.jsonl"),
            "VAL_SHARD_MANIFEST": str(tmp_path / "val-shards.jsonl"),
            "TRAIN_SHARD_COUNT": "1",
            "VAL_SHARD_COUNT": "1",
            "TRAIN_EXISTING_MERGE_REPORT": str(tmp_path / "train-merge.json"),
            "VAL_EXISTING_MERGE_REPORT": str(tmp_path / "val-merge.json"),
            "CONFIG": str(tmp_path / "config.yaml"),
            "CACHE_ROOT": str(tmp_path / "cache"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "GWYOLO_CODE_COMMIT": "commit",
            "MAX_ATTEMPTS": "2",
            "RETRY_DELAY_SECONDS": "0",
            "MINIMUM_FREE_KB": "1",
            "CALL_LOG": str(call_log),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "train-shard-0 exhausted bounded materialization retries" in completed.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert all("gravityspy-network-strain-materialize" in call for call in calls)
