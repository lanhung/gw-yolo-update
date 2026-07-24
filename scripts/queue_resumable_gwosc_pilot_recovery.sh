#!/usr/bin/env bash
set -euo pipefail

# Preserve the exact range partition of a score-blind GWOSC pilot when a slow
# endpoint exhausts an earlier bounded retry supervisor. This script is a
# fallback only: a completed upstream pilot is replay-verified and produces no
# recovery receipt, so downstream recovery queues can distinguish a real
# recovery from the already-authoritative path.

required_variables=(
  UPSTREAM_PID
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PILOT_PLAN
  PILOT_REPORT
  CACHE_ROOT
  PILOT_OUTPUT_DIR
  RECOVERY_RECEIPT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

POLL_SECONDS=${POLL_SECONDS:-30}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
CHUNK_SAMPLES=${CHUNK_SAMPLES:-1048576}
MAX_RECOVERY_ATTEMPTS=${MAX_RECOVERY_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-120}

for value in \
  "$UPSTREAM_PID" \
  "$POLL_SECONDS" \
  "$DOWNLOAD_WORKERS" \
  "$CHUNK_SAMPLES" \
  "$MAX_RECOVERY_ATTEMPTS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "PID and recovery controls must be positive integers" >&2
    exit 2
  fi
done
if ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "retry delay must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -x "$TASK_PYTHON" ]] \
  || [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ ! -f "$PILOT_PLAN" ]]; then
  echo "recovery runtime, code checkout, or pilot plan is invalid" >&2
  exit 2
fi
if [[ "$(dirname "$PILOT_REPORT")" != "$PILOT_OUTPUT_DIR" ]]; then
  echo "PILOT_REPORT must live directly in PILOT_OUTPUT_DIR" >&2
  exit 2
fi
if [[ "$RECOVERY_RECEIPT" == "$PILOT_REPORT" ]]; then
  echo "recovery receipt must not overwrite the immutable pilot report" >&2
  exit 2
fi

while kill -0 "$UPSTREAM_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done

verify_pilot() {
  "$TASK_PYTHON" - \
    "$PILOT_PLAN" \
    "$PILOT_REPORT" \
    "$GWYOLO_CODE_COMMIT" \
    "$DOWNLOAD_WORKERS" \
    "$CHUNK_SAMPLES" \
    "$PILOT_OUTPUT_DIR/batch_download_state.json" <<'PY'
import hashlib
import json
import pathlib
import sys


plan_path, report_path, commit, workers, chunk_samples, state_path = sys.argv[1:]
plan_file = pathlib.Path(plan_path)
report_file = pathlib.Path(report_path)
if not report_file.is_file():
    raise SystemExit("completed pilot report is absent")
plan = json.loads(plan_file.read_text(encoding="utf-8"))
report = json.loads(report_file.read_text(encoding="utf-8"))
expected_files = int(plan.get("selected_pairs", -1)) * len(plan.get("detectors", []))
keys = {
    (str(row.get("pair_id")), str(row.get("detector")))
    for row in report.get("files", [])
}
if (
    plan.get("status") != "development_acquisition_plan"
    or plan.get("run") != "O4a"
    or plan.get("locked_evaluation_data") is not False
    or int(plan.get("shard_index", -1)) != 0
    or plan.get("code_commit") != commit
    or report.get("status") != "verified_development_strain_batch"
    or report.get("passed") is not True
    or report.get("run") != "O4a"
    or report.get("plan_sha256")
    != hashlib.sha256(plan_file.read_bytes()).hexdigest()
    or report.get("code_commit") != commit
    or int(report.get("selected_pairs", -1))
    != int(plan.get("selected_pairs", -2))
    or int(report.get("verified_files", -1)) != expected_files
    or len(keys) != expected_files
    or any(
        row.get("verification", {}).get("passed") is not True
        for row in report.get("files", [])
    )
):
    raise SystemExit("pilot report failed exact recovery replay")
state_file = pathlib.Path(state_path)
if state_file.is_file():
    state = json.loads(state_file.read_text(encoding="utf-8"))
    identity = state.get("run_identity", {})
    if (
        identity.get("plan_sha256") != report.get("plan_sha256")
        or int(identity.get("download_workers", -1)) != int(workers)
        or int(identity.get("chunk_samples", -1)) != int(chunk_samples)
    ):
        raise SystemExit("pilot state changed range partition or chunk identity")
PY
}

if [[ -s "$PILOT_REPORT" ]]; then
  verify_pilot
  echo "upstream pilot already passed; no recovery receipt created"
  exit 0
fi

state_path="$PILOT_OUTPUT_DIR/batch_download_state.json"
if [[ -s "$state_path" ]]; then
  "$TASK_PYTHON" - \
    "$state_path" \
    "$PILOT_PLAN" \
    "$DOWNLOAD_WORKERS" \
    "$CHUNK_SAMPLES" <<'PY'
import hashlib
import json
import pathlib
import sys


state_path, plan_path, workers, chunk_samples = sys.argv[1:]
state = json.loads(pathlib.Path(state_path).read_text(encoding="utf-8"))
identity = state.get("run_identity", {})
if (
    identity.get("plan_sha256")
    != hashlib.sha256(pathlib.Path(plan_path).read_bytes()).hexdigest()
    or int(identity.get("download_workers", -1)) != int(workers)
    or int(identity.get("chunk_samples", -1)) != int(chunk_samples)
):
    raise SystemExit("recovery would change the existing range-download identity")
PY
fi

mkdir -p "$CACHE_ROOT" "$PILOT_OUTPUT_DIR" "$(dirname "$RECOVERY_RECEIPT")"
attempts_executed=0
for ((attempt = 1; attempt <= MAX_RECOVERY_ATTEMPTS; attempt++)); do
  attempts_executed=$attempt
  printf '%s pilot-recovery-attempt=%s workers=%s chunk_samples=%s\n' \
    "$(date -u +%FT%TZ)" "$attempt" "$DOWNLOAD_WORKERS" "$CHUNK_SAMPLES"
  if (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli gwosc-batch-download \
      --plan "$PILOT_PLAN" \
      --cache-dir "$CACHE_ROOT" \
      --output-dir "$PILOT_OUTPUT_DIR" \
      --download-workers "$DOWNLOAD_WORKERS" \
      --chunk-samples "$CHUNK_SAMPLES"
  ); then
    break
  fi
  if (( attempt < MAX_RECOVERY_ATTEMPTS )); then
    sleep "$RETRY_DELAY_SECONDS"
  fi
done
if [[ ! -s "$PILOT_REPORT" ]]; then
  echo "score-blind pilot exhausted bounded recovery attempts" >&2
  exit 1
fi
verify_pilot

export attempts_executed DOWNLOAD_WORKERS CHUNK_SAMPLES CACHE_ROOT
export PILOT_OUTPUT_DIR GWYOLO_CODE_COMMIT
"$TASK_PYTHON" - \
  "$PILOT_PLAN" \
  "$PILOT_REPORT" \
  "$RECOVERY_RECEIPT" <<'PY'
import hashlib
import json
import os
import pathlib
import tempfile


plan_path, report_path, target_value = map(pathlib.Path, os.sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
result = {
    "status": "verified_resumable_gwosc_pilot_recovery",
    "passed": True,
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "candidate_scores_inspected": False,
    "run": "O4a",
    "plan_path": str(plan_path.resolve()),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "pilot_report_path": str(report_path.resolve()),
    "pilot_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "selected_pairs": int(report["selected_pairs"]),
    "verified_files": int(report["verified_files"]),
    "download_workers": int(os.environ["DOWNLOAD_WORKERS"]),
    "chunk_samples": int(os.environ["CHUNK_SAMPLES"]),
    "recovery_attempts_executed": int(os.environ["attempts_executed"]),
    "code_commit": os.environ["GWYOLO_CODE_COMMIT"],
    "exact_command": (
        "python -m gwyolo.cli gwosc-batch-download "
        f"--plan {plan_path} --cache-dir {os.environ['CACHE_ROOT']} "
        f"--output-dir {os.environ['PILOT_OUTPUT_DIR']} "
        f"--download-workers {os.environ['DOWNLOAD_WORKERS']} "
        f"--chunk-samples {os.environ['CHUNK_SAMPLES']}"
    ),
}
target = pathlib.Path(target_value)
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(json.dumps(result, indent=2, sort_keys=True))
PY
