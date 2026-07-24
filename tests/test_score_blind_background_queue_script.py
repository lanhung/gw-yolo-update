from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "queue_score_blind_background_after_pilot.sh"
)


def test_score_blind_queue_is_pilot_gated_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'while kill -0 "$PILOT_PID"' in source
    assert "verified_development_strain_batch" in source
    assert "authorized_validation_candidate_continuous_background_plan" in source
    assert 'parent_plan.get("run") != "O4a"' in source
    assert "candidate_scores_inspected" in source
    assert "run_background_acquisition_range.sh" in source
    assert "pilot exited without a completed batch report" in source
    assert "TEST_FRACTION=${TEST_FRACTION:-0}" in source


def test_score_blind_queue_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 1
    compile(snippets[0], f"{SCRIPT.name}:heredoc", "exec")


def test_score_blind_queue_rejects_nonzero_test_fraction(tmp_path: Path) -> None:
    environment = os.environ.copy()
    for name in (
        "TASK_PYTHON",
        "TASK_CODE_DIR",
        "GWYOLO_CODE_COMMIT",
        "PILOT_PLAN",
        "PILOT_REPORT",
        "PLAN_AUTHORIZATION",
        "PARENT_PLAN",
        "EVENT_EXCLUSIONS",
        "CACHE_ROOT",
        "OUTPUT_ROOT",
    ):
        environment[name] = str(tmp_path / name.lower())
    environment["PILOT_PID"] = "123"
    environment["SHARD_STOP_EXCLUSIVE"] = "220"
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


def test_score_blind_queue_executes_only_after_exact_pilot_contract(
    tmp_path: Path,
) -> None:
    code = tmp_path / "code"
    (code / "src/gwyolo").mkdir(parents=True)
    (code / "scripts").mkdir()
    downstream = code / "scripts/run_background_acquisition_range.sh"
    downstream.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nprintf "started\\n" > "$QUEUE_MARKER"\n',
        encoding="utf-8",
    )
    downstream.chmod(0o755)

    def write(name: str, value: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    commit = "abc123"
    parent = write(
        "parent.json",
        {
            "status": "development_acquisition_plan",
            "run": "O4a",
            "locked_evaluation_data": False,
            "selected_pairs": 4,
        },
    )
    pilot_plan = write(
        "pilot-plan.json",
        {
            "status": "development_acquisition_plan",
            "run": "O4a",
            "locked_evaluation_data": False,
            "parent_plan_sha256": digest(parent),
            "shard_index": 0,
            "pairs_per_shard": 4,
            "selected_pairs": 4,
            "detectors": ["H1", "L1"],
            "code_commit": commit,
        },
    )
    files = [
        {
            "pair_id": f"pair-{pair}",
            "detector": detector,
            "verification": {"passed": True},
        }
        for pair in range(4)
        for detector in ("H1", "L1")
    ]
    pilot_report = write(
        "pilot-report.json",
        {
            "status": "verified_development_strain_batch",
            "passed": True,
            "run": "O4a",
            "plan_sha256": digest(pilot_plan),
            "selected_pairs": 4,
            "verified_files": 8,
            "code_commit": commit,
            "files": files,
        },
    )
    authorization = write(
        "authorization.json",
        {
            "status": "authorized_validation_candidate_continuous_background_plan",
            "passed": True,
            "candidate_scores_inspected": False,
            "test_rows_read": 0,
            "authorization_identity": {
                "parent_plan_sha256": digest(parent),
                "selected_pairs": 4,
                "pairs_per_shard": 4,
                "shard_stop_exclusive": 1,
            },
        },
    )
    exclusions = write(
        "event-exclusions.json",
        {
            "status": "development_catalog_event_exclusions",
            "run": "O4a",
            "padding_seconds": 16.0,
        },
    )
    marker = tmp_path / "started.txt"
    environment = {
        **os.environ,
        "TASK_PYTHON": sys.executable,
        "TASK_CODE_DIR": str(code),
        "GWYOLO_CODE_COMMIT": commit,
        "PILOT_PID": "99999999",
        "PILOT_PLAN": str(pilot_plan),
        "PILOT_REPORT": str(pilot_report),
        "PLAN_AUTHORIZATION": str(authorization),
        "PARENT_PLAN": str(parent),
        "EVENT_EXCLUSIONS": str(exclusions),
        "CACHE_ROOT": str(tmp_path / "cache"),
        "OUTPUT_ROOT": str(tmp_path / "full-output"),
        "SHARD_STOP_EXCLUSIVE": "1",
        "QUEUE_MARKER": str(marker),
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "started\n"
