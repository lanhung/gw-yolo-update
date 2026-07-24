from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "queue_consolidated_o4a_background_recovery.sh"
)


def test_recovery_waits_for_all_upstreams_and_preserves_score_blind_rules() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'for pid in $UPSTREAM_PIDS' in source
    assert 'while kill -0 "$pid"' in source
    assert "run_background_acquisition_range.sh" in source
    assert "TEST_FRACTION=0" in source
    assert "DOWNLOAD_ONLY=false" in source
    assert 'RECOVERY_DOWNLOAD_WORKERS:-8' in source
    assert 'RECOVERY_MAX_ATTEMPTS:-20' in source
    assert "candidate_scores_inspected" in source
    assert '"test_rows_read": 0' in source
    assert "reused_existing_primary_report" in source
    assert "consolidated_single_stream_recovery" in source


def test_recovery_rejects_invalid_upstream_pid_before_waiting(
    tmp_path: Path,
) -> None:
    code = tmp_path / "code"
    (code / "src" / "gwyolo").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(code)], check=True)
    subprocess.run(
        ["git", "-C", str(code), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(code), "config", "user.name", "Test"],
        check=True,
    )
    (code / "placeholder").write_text("x", encoding="utf-8")
    subprocess.run(["git", "-C", str(code), "add", "placeholder"], check=True)
    subprocess.run(["git", "-C", str(code), "commit", "-qm", "test"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip()
    environment = os.environ.copy()
    for name in (
        "EXISTING_COMPLETION_REPORT",
        "PARENT_PLAN",
        "EVENT_EXCLUSIONS",
        "PLAN_AUTHORIZATION",
        "PILOT_PLAN",
        "PILOT_REPORT",
        "CACHE_ROOT",
        "OUTPUT_ROOT",
        "RECOVERY_RECEIPT",
    ):
        environment[name] = str(tmp_path / name.lower())
    environment.update(
        {
            "TASK_PYTHON": os.environ.get("PYTHON", "python"),
            "TASK_CODE_DIR": str(code),
            "GWYOLO_CODE_COMMIT": commit,
            "UPSTREAM_PIDS": "not-a-pid",
            "SHARD_STOP_EXCLUSIVE": "220",
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "must contain positive process IDs" in completed.stderr
