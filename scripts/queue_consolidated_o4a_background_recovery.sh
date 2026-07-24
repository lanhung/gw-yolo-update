#!/usr/bin/env bash
set -euo pipefail

# Wait for all existing disjoint O4a acquisition ranges to terminate naturally.
# Reuse a completed primary report when available; otherwise run one consolidated
# score-blind, test-free recovery against the shared immutable cache.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  UPSTREAM_PIDS
  EXISTING_COMPLETION_REPORT
  PARENT_PLAN
  EVENT_EXCLUSIONS
  PLAN_AUTHORIZATION
  PILOT_PLAN
  PILOT_REPORT
  CACHE_ROOT
  OUTPUT_ROOT
  RECOVERY_RECEIPT
  SHARD_STOP_EXCLUSIVE
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
observed_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi
for pid in $UPSTREAM_PIDS; do
  if ! [[ "$pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "UPSTREAM_PIDS must contain positive process IDs" >&2
    exit 2
  fi
done

for pid in $UPSTREAM_PIDS; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${QUEUE_POLL_SECONDS:-30}"
  done
done

write_receipt() {
  local source_report=$1
  local mode=$2
  mkdir -p "$(dirname "$RECOVERY_RECEIPT")"
  "$TASK_PYTHON" - "$RECOVERY_RECEIPT" "$source_report" "$PARENT_PLAN" \
    "$PLAN_AUTHORIZATION" "$GWYOLO_CODE_COMMIT" "$mode" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target, report_path, parent_path, authorization_path, commit, mode = sys.argv[1:]
report = pathlib.Path(report_path).resolve()
parent = pathlib.Path(parent_path).resolve()
authorization = pathlib.Path(authorization_path).resolve()
loaded = json.loads(report.read_text(encoding="utf-8"))
plan = json.loads(parent.read_text(encoding="utf-8"))
auth = json.loads(authorization.read_text(encoding="utf-8"))
if (
    loaded.get("status") != "verified_multi_segment_development_background"
    or loaded.get("passed") is not True
    or loaded.get("split_strategy") != "hash_threshold_v1"
    or int(loaded.get("splits", {}).get("test", {}).get("windows", -1)) != 0
    or any(loaded.get("cross_split_block_overlaps", {}).values())
    or plan.get("run") != "O4a"
    or plan.get("locked_evaluation_data") is not False
    or auth.get("status")
    != "authorized_validation_candidate_continuous_background_plan"
    or auth.get("passed") is not True
    or auth.get("candidate_scores_inspected") is not False
    or int(auth.get("test_rows_read", -1)) != 0
):
    raise SystemExit("consolidated O4a recovery report crossed its score-blind boundary")


def artifact(path):
    value = pathlib.Path(path).resolve()
    return {
        "path": str(value),
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
    }


result = {
    "status": "consolidated_o4a_background_recovery_completed",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "development O4a background acquisition is input evidence; fixed-FAR "
        "scoring and locked evaluation remain pending"
    ),
    "candidate_scores_inspected": False,
    "test_rows_read": 0,
    "test_evaluation": None,
    "completion_mode": mode,
    "background_report": artifact(report),
    "parent_plan": artifact(parent),
    "plan_authorization": artifact(authorization),
    "code_commit": commit,
}
target = pathlib.Path(target)
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
}

if [[ -s "$EXISTING_COMPLETION_REPORT" ]]; then
  write_receipt "$EXISTING_COMPLETION_REPORT" reused_existing_primary_report
  exit 0
fi

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  PARENT_PLAN="$PARENT_PLAN" \
  EVENT_EXCLUSIONS="$EVENT_EXCLUSIONS" \
  PLAN_AUTHORIZATION="$PLAN_AUTHORIZATION" \
  PILOT_PLAN="$PILOT_PLAN" \
  PILOT_REPORT="$PILOT_REPORT" \
  CACHE_ROOT="$CACHE_ROOT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SHARD_START=0 \
  SHARD_STOP_EXCLUSIVE="$SHARD_STOP_EXCLUSIVE" \
  PAIRS_PER_SHARD="${PAIRS_PER_SHARD:-4}" \
  VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.2}" \
  TEST_FRACTION=0 \
  BACKGROUND_SEED="${BACKGROUND_SEED:-20260719}" \
  DOWNLOAD_WORKERS="${RECOVERY_DOWNLOAD_WORKERS:-8}" \
  MINIMUM_FREE_KB="${MINIMUM_FREE_KB:-536870912}" \
  MAX_ATTEMPTS="${RECOVERY_MAX_ATTEMPTS:-20}" \
  RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-120}" \
  DOWNLOAD_ONLY=false \
  bash "$TASK_CODE_DIR/scripts/run_background_acquisition_range.sh"

recovered_report="$OUTPUT_ROOT/merged-background/background_plan_report.json"
if [[ ! -s "$recovered_report" ]]; then
  echo "consolidated O4a recovery ended without a merged background report" >&2
  exit 1
fi
write_receipt "$recovered_report" consolidated_single_stream_recovery
