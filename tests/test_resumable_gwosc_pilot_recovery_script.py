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


def test_recovery_runs_cli_and_writes_separate_provenance(
    tmp_path: Path,
) -> None:
    code = tmp_path / "code"
    package = code / "src/gwyolo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        """
import hashlib
import json
import os
import pathlib
import sys

assert sys.argv[1] == "gwosc-batch-download"
arguments = dict(zip(sys.argv[2::2], sys.argv[3::2]))
plan_path = pathlib.Path(arguments["--plan"])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
output = pathlib.Path(arguments["--output-dir"])
output.mkdir(parents=True, exist_ok=True)
files = [
    {
        "pair_id": "pair-0",
        "detector": detector,
        "verification": {"passed": True},
    }
    for detector in plan["detectors"]
]
report = {
    "status": "verified_development_strain_batch",
    "passed": True,
    "run": "O4a",
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "code_commit": os.environ["GWYOLO_CODE_COMMIT"],
    "selected_pairs": 1,
    "verified_files": 2,
    "files": files,
}
(output / "batch_download_report.json").write_text(
    json.dumps(report), encoding="utf-8"
)
state = {
    "status": "complete",
    "completed_files": 2,
    "requested_files": 2,
    "run_identity": {
        "plan_sha256": report["plan_sha256"],
        "download_workers": int(arguments["--download-workers"]),
        "chunk_samples": int(arguments["--chunk-samples"]),
    },
}
(output / "batch_download_state.json").write_text(
    json.dumps(state), encoding="utf-8"
)
""".lstrip(),
        encoding="utf-8",
    )
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
                "code_commit": "pilot123",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "download"
    receipt = tmp_path / "recovery.json"
    environment = {
        **os.environ,
        "UPSTREAM_PID": "99999999",
        "TASK_PYTHON": sys.executable,
        "TASK_CODE_DIR": str(code),
        "GWYOLO_CODE_COMMIT": "pilot123",
        "RECOVERY_CODE_COMMIT": "recovery456",
        "PILOT_PLAN": str(plan),
        "PILOT_REPORT": str(output / "batch_download_report.json"),
        "CACHE_ROOT": str(tmp_path / "cache"),
        "PILOT_OUTPUT_DIR": str(output),
        "RECOVERY_RECEIPT": str(receipt),
        "DOWNLOAD_WORKERS": "8",
        "CHUNK_SAMPLES": "1048576",
        "MAX_RECOVERY_ATTEMPTS": "1",
        "RETRY_DELAY_SECONDS": "0",
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(receipt.read_text(encoding="utf-8"))
    assert result["status"] == "verified_resumable_gwosc_pilot_recovery"
    assert result["pilot_cli_code_commit"] == "pilot123"
    assert result["recovery_code_commit"] == "recovery456"
    assert result["recovery_attempts_executed"] == 1
    assert result["selected_pairs"] == 1
    assert result["verified_files"] == 2
    assert result["test_rows_read"] == 0
    assert result["candidate_scores_inspected"] is False
    assert result["recovery_script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()
