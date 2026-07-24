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
    / "queue_resumable_gwosc_pilot_recovery.sh"
)


def test_recovery_is_bounded_identity_preserving_and_score_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'while kill -0 "$UPSTREAM_PID"' in source
    assert "MAX_RECOVERY_ATTEMPTS" in source
    assert '--download-workers "$DOWNLOAD_WORKERS"' in source
    assert '--chunk-samples "$CHUNK_SAMPLES"' in source
    assert "candidate_scores_inspected" in source
    assert '"test_rows_read": 0' in source
    assert "O4a" in source
    assert "locked_evaluation_data" in source
    assert "verified_resumable_gwosc_pilot_recovery" in source
    assert "pilot_cli_code_commit" in source
    assert "recovery_code_commit" in source
    assert "recovery_script_sha256" in source


def test_recovery_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 3
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:{index}", "exec")


def test_passing_upstream_creates_no_recovery_receipt(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "src/gwyolo").mkdir(parents=True)
    plan = tmp_path / "pilot.json"
    plan.write_text(
        json.dumps(
            {
                "status": "development_acquisition_plan",
                "run": "O4a",
                "locked_evaluation_data": False,
                "shard_index": 0,
                "selected_pairs": 1,
                "detectors": ["H1", "L1"],
                "code_commit": "abc123",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "download"
    output.mkdir()
    report = output / "batch_download_report.json"
    report.write_text(
        json.dumps(
            {
                "status": "verified_development_strain_batch",
                "passed": True,
                "run": "O4a",
                "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "code_commit": "abc123",
                "selected_pairs": 1,
                "verified_files": 2,
                "files": [
                    {
                        "pair_id": "pair-0",
                        "detector": detector,
                        "verification": {"passed": True},
                    }
                    for detector in ("H1", "L1")
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "recovery.json"
    environment = {
        **os.environ,
        "UPSTREAM_PID": "99999999",
        "TASK_PYTHON": sys.executable,
        "TASK_CODE_DIR": str(code),
        "GWYOLO_CODE_COMMIT": "abc123",
        "RECOVERY_CODE_COMMIT": "recovery456",
        "PILOT_PLAN": str(plan),
        "PILOT_REPORT": str(report),
        "CACHE_ROOT": str(tmp_path / "cache"),
        "PILOT_OUTPUT_DIR": str(output),
        "RECOVERY_RECEIPT": str(receipt),
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "no recovery receipt created" in completed.stdout
    assert not receipt.exists()
