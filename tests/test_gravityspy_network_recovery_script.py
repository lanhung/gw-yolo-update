from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_gravityspy_network_recovery.sh"


def _base_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "TRAIN_SOURCE_MANIFEST": str(tmp_path / "train-source.jsonl"),
            "VAL_SOURCE_MANIFEST": str(tmp_path / "val-source.jsonl"),
            "CONFIG": str(tmp_path / "config.yaml"),
            "CACHE_ROOT": str(tmp_path / "cache"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "GWYOLO_CODE_COMMIT": "commit",
        }
    )
    return environment


def test_network_recovery_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_network_recovery_merged_mode_requires_both_merge_reports(
    tmp_path: Path,
) -> None:
    environment = _base_environment(tmp_path)
    environment["REPORT_MODE"] = "merged"
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "TRAIN_MERGE_REPORT" in completed.stderr


def test_network_recovery_merged_mode_hash_verifies_source_reports() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'report.get("status")' in source
    assert 'report.get("split") != split' in source
    assert 'digest(source) != item.get("sha256")' in source
    assert 'if ! report_output=$("$TASK_PYTHON"' in source
    assert "failed to resolve completed shard reports" in source


def test_network_recovery_propagates_merged_source_hash_failure(
    tmp_path: Path,
) -> None:
    environment = _base_environment(tmp_path)
    for name in ("train-source.jsonl", "val-source.jsonl", "config.yaml"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    source_report = tmp_path / "source-report.json"
    source_report.write_text("{}\n", encoding="utf-8")
    for split in ("train", "val"):
        merge = tmp_path / f"{split}-merge.json"
        merge.write_text(
            (
                '{"status":"verified_merged_gravityspy_aligned_network_numeric_split",'
                f'"split":"{split}","source_reports":['
                f'{{"path":"{source_report}","sha256":"wrong"}}]}}\n'
            ),
            encoding="utf-8",
        )
        environment[f"{split.upper()}_MERGE_REPORT"] = str(merge)
    environment["REPORT_MODE"] = "merged"
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "failed to resolve completed shard reports from train merge" in completed.stderr
