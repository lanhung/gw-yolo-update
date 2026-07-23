#!/usr/bin/env bash
set -euo pipefail

# Post-access only. The one-time opener and streaming worker must already have
# produced an access log and one immutable receipt row for every frozen shard.
# This script never selects a subset: it audits the complete shard inventory,
# runs every predeclared endpoint, and finalizes the suite receipt.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  LOCKED_SUITE_PLAN
  LOCKED_EXECUTION_PLAN
  LOCKED_ACCESS_LOG
  LOCKED_SHARD_RECEIPT_MANIFEST
  STREAMING_COMPLETION_AUDIT_OUTPUT
  RAW_CALIBRATION_REPORT
  MASK_CALIBRATION_REPORT
  VALIDATION_RAW_MASK_COMPARISON_REPORT
  OOD_CONFIG
  VALIDATION_OOD_REPORT
  VALIDATION_PE_PROMOTION_REPORT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required locked reduction variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "locked reduction requires its exact immutable checkout" >&2
  exit 3
fi
inputs=(
  "$TASK_PYTHON"
  "$LOCKED_SUITE_PLAN"
  "$LOCKED_EXECUTION_PLAN"
  "$LOCKED_ACCESS_LOG"
  "$LOCKED_SHARD_RECEIPT_MANIFEST"
  "$RAW_CALIBRATION_REPORT"
  "$MASK_CALIBRATION_REPORT"
  "$VALIDATION_RAW_MASK_COMPARISON_REPORT"
  "$OOD_CONFIG"
  "$VALIDATION_OOD_REPORT"
  "$VALIDATION_PE_PROMOTION_REPORT"
)
for path in "${inputs[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "locked reduction input is absent or empty: $path" >&2
    exit 3
  fi
done

mapfile -t suite_identity < <(
  "$TASK_PYTHON" - "$LOCKED_SUITE_PLAN" "$LOCKED_EXECUTION_PLAN" \
    "$LOCKED_ACCESS_LOG" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

suite_path = pathlib.Path(sys.argv[1]).resolve()
execution_path = pathlib.Path(sys.argv[2]).resolve()
access_path = pathlib.Path(sys.argv[3]).resolve()
commit = sys.argv[4]
suite = json.loads(suite_path.read_text(encoding="utf-8"))
execution = json.loads(execution_path.read_text(encoding="utf-8"))
access = json.loads(access_path.read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
frozen = access.get("frozen_artifacts", {})
if (
    suite.get("status") != "frozen_locked_evaluation_suite_plan"
    or execution.get("status")
    != "frozen_locked_o4b_streaming_execution_plan"
    or access.get("status") != "locked_evaluation_corpus_opened_once"
    or suite.get("code_commit") != commit
    or execution.get("code_commit") != commit
    or access.get("code_commit") != commit
    or frozen.get("locked_suite_plan", {}).get("path") != str(suite_path)
    or frozen.get("locked_suite_plan", {}).get("sha256") != digest(suite_path)
    or frozen.get("locked_execution_plan", {}).get("path")
    != str(execution_path)
    or frozen.get("locked_execution_plan", {}).get("sha256")
    != digest(execution_path)
):
    raise SystemExit("locked reduction plan/access identity failed replay")
print(suite["outputs"]["suite_receipt"])
PY
)
if (( ${#suite_identity[@]} != 1 )); then
  echo "locked reduction did not resolve one suite receipt" >&2
  exit 3
fi
suite_receipt=${suite_identity[0]}

if [[ ! -s "$STREAMING_COMPLETION_AUDIT_OUTPUT" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli locked-o4b-streaming-completion-audit \
      --execution-plan "$LOCKED_EXECUTION_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --receipt-manifest "$LOCKED_SHARD_RECEIPT_MANIFEST" \
      --output "$STREAMING_COMPLETION_AUDIT_OUTPUT"
  )
fi

"$TASK_PYTHON" - "$STREAMING_COMPLETION_AUDIT_OUTPUT" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_locked_o4b_streaming_execution_audit"
    or report.get("passed") is not True
    or report.get("all_predeclared_shards_reduced") is not True
    or report.get("completed_shards") != report.get("expected_shards")
    or report.get("failed_shards") != []
    or report.get("code_commit") != sys.argv[2]
):
    raise SystemExit("locked all-shard completion audit did not pass")
PY

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  LOCKED_SUITE_PLAN="$LOCKED_SUITE_PLAN" \
  LOCKED_ACCESS_LOG="$LOCKED_ACCESS_LOG" \
  RAW_CALIBRATION_REPORT="$RAW_CALIBRATION_REPORT" \
  MASK_CALIBRATION_REPORT="$MASK_CALIBRATION_REPORT" \
  VALIDATION_RAW_MASK_COMPARISON_REPORT="$VALIDATION_RAW_MASK_COMPARISON_REPORT" \
  bash "$TASK_CODE_DIR/scripts/run_locked_search_endpoints.sh"

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  LOCKED_SUITE_PLAN="$LOCKED_SUITE_PLAN" \
  LOCKED_ACCESS_LOG="$LOCKED_ACCESS_LOG" \
  OOD_CONFIG="$OOD_CONFIG" \
  VALIDATION_OOD_REPORT="$VALIDATION_OOD_REPORT" \
  bash "$TASK_CODE_DIR/scripts/run_locked_ood_endpoint.sh"

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  LOCKED_SUITE_PLAN="$LOCKED_SUITE_PLAN" \
  LOCKED_ACCESS_LOG="$LOCKED_ACCESS_LOG" \
  VALIDATION_PE_PROMOTION_REPORT="$VALIDATION_PE_PROMOTION_REPORT" \
  bash "$TASK_CODE_DIR/scripts/run_locked_pe_endpoints.sh"

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  LOCKED_SUITE_PLAN="$LOCKED_SUITE_PLAN" \
  LOCKED_ACCESS_LOG="$LOCKED_ACCESS_LOG" \
  bash "$TASK_CODE_DIR/scripts/run_locked_catalog_endpoint.sh"

if [[ ! -s "$suite_receipt" ]]; then
  mkdir -p "$(dirname "$suite_receipt")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli locked-evaluation-suite-finalize \
      --plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --streaming-completion-audit "$STREAMING_COMPLETION_AUDIT_OUTPUT" \
      --output "$suite_receipt"
  )
fi

"$TASK_PYTHON" - "$suite_receipt" "$STREAMING_COMPLETION_AUDIT_OUTPUT" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1]).resolve()
streaming_path = pathlib.Path(sys.argv[2]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("status") != "completed_locked_evaluation_suite_receipt"
    or receipt.get("passed") is not True
    or receipt.get("all_predeclared_outputs_present") is not True
    or receipt.get("negative_and_null_results_retained") is not True
    or receipt.get("streaming_completion_audit", {}).get("path")
    != str(streaming_path)
    or receipt.get("streaming_completion_audit", {}).get("sha256")
    != hashlib.sha256(streaming_path.read_bytes()).hexdigest()
):
    raise SystemExit("locked suite receipt failed final replay")
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
