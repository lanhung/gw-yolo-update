#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PILOT_CODE_COMMIT
  PILOT_PID
  PILOT_PLAN
  PILOT_REPORT
  PLAN_AUTHORIZATION
  PARENT_PLAN
  EVENT_EXCLUSIONS
  CACHE_ROOT
  OUTPUT_ROOT
  SHARD_STOP_EXCLUSIVE
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

PAIRS_PER_SHARD=${PAIRS_PER_SHARD:-4}
POLL_SECONDS=${POLL_SECONDS:-30}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-536870912}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-120}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.2}
TEST_FRACTION=${TEST_FRACTION:-0}
BACKGROUND_SEED=${BACKGROUND_SEED:-20260719}

if ! [[ "$PILOT_PID" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$PAIRS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$SHARD_STOP_EXCLUSIVE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MINIMUM_FREE_KB" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "pilot PID and queue/acquisition controls must be valid integers" >&2
  exit 2
fi
if [[ "$TEST_FRACTION" != "0" && "$TEST_FRACTION" != "0.0" ]]; then
  echo "score-blind validation background must keep test_fraction=0" >&2
  exit 2
fi
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "task Python is absent or not executable: $TASK_PYTHON" >&2
  exit 2
fi
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ ! -f "$TASK_CODE_DIR/scripts/run_background_acquisition_range.sh" ]]; then
  echo "task code directory is invalid: $TASK_CODE_DIR" >&2
  exit 2
fi
for input in "$PILOT_PLAN" "$PLAN_AUTHORIZATION" "$PARENT_PLAN" "$EVENT_EXCLUSIONS"; do
  if [[ ! -f "$input" ]]; then
    echo "required immutable input is absent: $input" >&2
    exit 2
  fi
done
if [[ "$OUTPUT_ROOT" == "$(dirname "$PILOT_REPORT")" ]] \
  || [[ "$OUTPUT_ROOT" == "$(dirname "$(dirname "$PILOT_REPORT")")" ]]; then
  echo "full acquisition output must be separate from the immutable pilot output" >&2
  exit 2
fi

while kill -0 "$PILOT_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
if [[ ! -s "$PILOT_REPORT" ]]; then
  echo "pilot exited without a completed batch report" >&2
  exit 1
fi

"$TASK_PYTHON" - \
  "$PILOT_REPORT" \
  "$PILOT_PLAN" \
  "$PLAN_AUTHORIZATION" \
  "$PARENT_PLAN" \
  "$EVENT_EXCLUSIONS" \
  "$PILOT_CODE_COMMIT" \
  "$PAIRS_PER_SHARD" \
  "$SHARD_STOP_EXCLUSIVE" <<'PY'
import hashlib
import json
import pathlib
import sys


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    pilot_report_path,
    pilot_plan_path,
    authorization_path,
    parent_plan_path,
    event_exclusions_path,
    pilot_code_commit,
    pairs_per_shard,
    shard_stop,
) = sys.argv[1:]
pilot_report = load(pilot_report_path)
pilot_plan = load(pilot_plan_path)
authorization = load(authorization_path)
parent_plan = load(parent_plan_path)
event_exclusions = load(event_exclusions_path)
identity = authorization.get("authorization_identity", {})
parent_hash = digest(parent_plan_path)
pilot_plan_hash = digest(pilot_plan_path)
pairs_per_shard = int(pairs_per_shard)
shard_stop = int(shard_stop)
expected_pilot_files = int(pilot_plan.get("selected_pairs", -1)) * len(
    pilot_plan.get("detectors", [])
)
pilot_keys = {
    (str(row.get("pair_id")), str(row.get("detector")))
    for row in pilot_report.get("files", [])
}

if (
    parent_plan.get("status") != "development_acquisition_plan"
    or parent_plan.get("run") != "O4a"
    or parent_plan.get("locked_evaluation_data") is not False
    or int(parent_plan.get("selected_pairs", -1)) != pairs_per_shard * shard_stop
):
    raise SystemExit("parent plan is not the authorized complete O4a validation range")
if (
    authorization.get("status")
    != "authorized_validation_candidate_continuous_background_plan"
    or authorization.get("passed") is not True
    or authorization.get("candidate_scores_inspected") is not False
    or int(authorization.get("test_rows_read", -1)) != 0
    or identity.get("parent_plan_sha256") != parent_hash
    or int(identity.get("selected_pairs", -1))
    != int(parent_plan["selected_pairs"])
    or int(identity.get("pairs_per_shard", -1)) != pairs_per_shard
    or int(identity.get("shard_stop_exclusive", -1)) != shard_stop
):
    raise SystemExit("background authorization does not bind the requested score-blind range")
if (
    pilot_plan.get("status") != "development_acquisition_plan"
    or pilot_plan.get("run") != "O4a"
    or pilot_plan.get("locked_evaluation_data") is not False
    or pilot_plan.get("parent_plan_sha256") != parent_hash
    or int(pilot_plan.get("shard_index", -1)) != 0
    or int(pilot_plan.get("pairs_per_shard", -1)) != pairs_per_shard
    or int(pilot_plan.get("selected_pairs", -1)) != pairs_per_shard
    or pilot_plan.get("code_commit") != pilot_code_commit
):
    raise SystemExit("pilot plan is not shard zero of the authorized parent")
if (
    pilot_report.get("status") != "verified_development_strain_batch"
    or pilot_report.get("passed") is not True
    or pilot_report.get("run") != "O4a"
    or pilot_report.get("plan_sha256") != pilot_plan_hash
    or int(pilot_report.get("selected_pairs", -1)) != pairs_per_shard
    or int(pilot_report.get("verified_files", -1)) != expected_pilot_files
    or len(pilot_keys) != expected_pilot_files
    or pilot_report.get("code_commit") != pilot_code_commit
    or any(
        row.get("verification", {}).get("passed") is not True
        for row in pilot_report.get("files", [])
    )
):
    raise SystemExit("pilot batch is incomplete, failed, or has another identity")
if (
    event_exclusions.get("status") != "development_catalog_event_exclusions"
    or event_exclusions.get("run") != "O4a"
    or float(event_exclusions.get("padding_seconds", -1)) != 16.0
):
    raise SystemExit("event exclusions are not the frozen O4a +/-16 second list")
PY

export SHARD_START=0
export PAIRS_PER_SHARD
export DOWNLOAD_WORKERS
export MINIMUM_FREE_KB
export MAX_ATTEMPTS
export RETRY_DELAY_SECONDS
export VALIDATION_FRACTION
export TEST_FRACTION
export BACKGROUND_SEED

exec bash "$TASK_CODE_DIR/scripts/run_background_acquisition_range.sh"
