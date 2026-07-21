from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_gravityspy_validation_fallback.sh"
)


def test_validation_fallback_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_validation_fallback_waits_and_reuses_completed_legacy_shards() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'while kill -0 "$WAIT_PID"' in source
    assert 'if [[ -s "$legacy_report" ]]' in source
    assert "gravityspy-network-strain-materialize" in source
    assert "gravityspy-network-numeric-merge" in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert '--split val' in source
    assert '--split test' not in source
    assert 'fallback OUTPUT_ROOT must be separate' in source


def test_validation_fallback_rejects_nonpositive_shard_count(tmp_path: Path) -> None:
    environment = os.environ.copy()
    for name in (
        "TASK_PYTHON",
        "TASK_CODE_DIR",
        "GWYOLO_CODE_COMMIT",
        "SOURCE_MANIFEST",
        "CONFIG",
        "CACHE_ROOT",
        "LEGACY_OUTPUT_PREFIX",
        "OUTPUT_ROOT",
    ):
        environment[name] = str(tmp_path / name.lower())
    environment["SOURCE_SHARD_COUNT"] = "0"
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "must be positive integers" in completed.stderr


def test_validation_fallback_reuses_all_completed_legacy_reports(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    source_manifest = tmp_path / "source.jsonl"
    config = tmp_path / "config.yaml"
    source_manifest.write_text("{}\n", encoding="utf-8")
    config.write_text("physical_training: {}\n", encoding="utf-8")
    legacy_prefix = tmp_path / "legacy-shard"
    legacy_reports = []
    for shard in range(2):
        directory = Path(f"{legacy_prefix}{shard}")
        directory.mkdir()
        report = directory / "gravityspy_network_numeric_report.json"
        report.write_text("{}\n", encoding="utf-8")
        legacy_reports.append(report)

    call_log = tmp_path / "calls.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$*\" >>\"$CALL_LOG\"\n"
        "output=\n"
        "while (($#)); do\n"
        "  if [[ $1 == --output-dir ]]; then output=$2; shift 2; else shift; fi\n"
        "done\n"
        "mkdir -p \"$output\"\n"
        "printf '{}\\n' >\"$output/gravityspy_network_numeric_merge_report.json\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_root = tmp_path / "fallback"
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": str(fake_python),
            "TASK_CODE_DIR": str(repository),
            "GWYOLO_CODE_COMMIT": commit,
            "SOURCE_MANIFEST": str(source_manifest),
            "CONFIG": str(config),
            "CACHE_ROOT": str(tmp_path / "cache"),
            "LEGACY_OUTPUT_PREFIX": str(legacy_prefix),
            "OUTPUT_ROOT": str(output_root),
            "SOURCE_SHARD_COUNT": "2",
            "MINIMUM_FREE_KB": "1",
            "CALL_LOG": str(call_log),
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    calls = call_log.read_text(encoding="utf-8")
    assert calls.count("gravityspy-network-numeric-merge") == 1
    assert "gravityspy-network-strain-materialize" not in calls
    assert all(str(report) in calls for report in legacy_reports)
    assert "--split val" in calls
    assert "--split test" not in calls
