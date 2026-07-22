#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  FIVE_SEED_UPSTREAM_PID
  FIVE_SEED_UPSTREAM_IDENTITY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  MODEL_SELECTION_OVERLAP_MANIFEST
  MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  INDEPENDENT_PE_OVERLAP_REPORT
  INDEPENDENT_PE_OVERLAP_AUDIT
  INDEPENDENT_PE_UPSTREAM_PID
  INDEPENDENT_PE_UPSTREAM_IDENTITY
  OVERLAP_MANIFEST
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  MASK_VALIDATION_OUTPUT_ROOT
  MASK_TIMING_OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required mask publication queue variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "mask publication queue requires its declared immutable checkout" >&2
  exit 2
fi

wait_for_artifact() {
  local label=$1
  local artifact=$2
  local pid=$3
  local identity=$4
  while [[ ! -s "$artifact" ]]; do
    if [[ ! -r "/proc/$pid/cmdline" ]] \
      || ! tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$identity"; then
      echo "$label upstream ended without its required artifact: $artifact" >&2
      exit 1
    fi
    sleep 30
  done
}

wait_for_gpu_idle() {
  while :; do
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
}

wait_for_artifact \
  five-seed \
  "$FIVE_SEED_SUMMARY" \
  "$FIVE_SEED_UPSTREAM_PID" \
  "$FIVE_SEED_UPSTREAM_IDENTITY"
wait_for_artifact \
  independent-pe \
  "$INDEPENDENT_PE_OVERLAP_REPORT" \
  "$INDEPENDENT_PE_UPSTREAM_PID" \
  "$INDEPENDENT_PE_UPSTREAM_IDENTITY"
for path in \
  "$TASK_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$MODEL_SELECTION_OVERLAP_MANIFEST" \
  "$MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$INDEPENDENT_PE_OVERLAP_REPORT" \
  "$INDEPENDENT_PE_OVERLAP_AUDIT" \
  "$OVERLAP_MANIFEST" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "mask publication queue input is absent: $path" >&2
    exit 2
  fi
done

mask_receipt="$MASK_VALIDATION_OUTPUT_ROOT/mask_deglitch_validation_receipt.json"
pipeline="$MASK_VALIDATION_OUTPUT_ROOT/pipeline/mask_search_pipeline_report.json"
if [[ ! -s "$mask_receipt" ]]; then
  wait_for_gpu_idle
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    FIVE_SEED_SUMMARY="$FIVE_SEED_SUMMARY" \
    UNIFORM_CONFIG="$UNIFORM_CONFIG" \
    FAMILY_BALANCED_CONFIG="$FAMILY_BALANCED_CONFIG" \
    MODEL_SELECTION_OVERLAP_MANIFEST="$MODEL_SELECTION_OVERLAP_MANIFEST" \
    MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST="$MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST" \
    INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    INDEPENDENT_PE_OVERLAP_REPORT="$INDEPENDENT_PE_OVERLAP_REPORT" \
    INDEPENDENT_OVERLAP_AUDIT="$INDEPENDENT_PE_OVERLAP_AUDIT" \
    OVERLAP_MANIFEST="$OVERLAP_MANIFEST" \
    BACKGROUND_MANIFEST="$BACKGROUND_MANIFEST" \
    INJECTION_MANIFEST="$INJECTION_MANIFEST" \
    OUTPUT_ROOT="$MASK_VALIDATION_OUTPUT_ROOT" \
    bash "$TASK_CODE_DIR/scripts/run_mask_deglitch_validation.sh"
fi
test -s "$mask_receipt"
test -s "$pipeline"

timing_receipt="$MASK_TIMING_OUTPUT_ROOT/mask_timing_validation_receipt.json"
if [[ ! -s "$timing_receipt" ]]; then
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    MASK_VALIDATION_RECEIPT="$mask_receipt" \
    PIPELINE_REPORT="$pipeline" \
    OUTPUT_ROOT="$MASK_TIMING_OUTPUT_ROOT" \
    bash "$TASK_CODE_DIR/scripts/run_mask_timing_validation.sh"
fi

scale_allowed=$(
  "$TASK_PYTHON" - "$timing_receipt" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_validation_only_mask_timing_gate"
    or report.get("test_rows_read") != 0
    or report.get("locked_test_allowed") is not False
):
    raise SystemExit("mask timing receipt failed queue replay")
print("true" if report.get("coherent_background_scale_allowed") is True else "false")
PY
)
if [[ "$scale_allowed" != true ]]; then
  printf '%s\n' "mask timing gate retained a negative result; background scale not started"
  exit 0
fi
if [[ "${RUN_MASK_BACKGROUND:-0}" != 1 ]]; then
  printf '%s\n' "mask timing gate passed; RUN_MASK_BACKGROUND is disabled"
  exit 0
fi

background_required=(
  BACKGROUND_UPSTREAM_ARTIFACT
  BACKGROUND_UPSTREAM_PID
  BACKGROUND_UPSTREAM_IDENTITY
  PARENT_PLAN
  EVENT_EXCLUSIONS
  COHERENCE_CONFIG
  CACHE_ROOT
  MASK_BACKGROUND_OUTPUT_ROOT
  SHARD_STOP_EXCLUSIVE
)
for variable in "${background_required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required mask background queue variable is unset: $variable" >&2
    exit 2
  fi
done
wait_for_artifact \
  background-baseline \
  "$BACKGROUND_UPSTREAM_ARTIFACT" \
  "$BACKGROUND_UPSTREAM_PID" \
  "$BACKGROUND_UPSTREAM_IDENTITY"
env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  MASK_VALIDATION_RECEIPT="$mask_receipt" \
  MASK_TIMING_RECEIPT="$timing_receipt" \
  FIVE_SEED_SUMMARY="$FIVE_SEED_SUMMARY" \
  UNIFORM_CONFIG="$UNIFORM_CONFIG" \
  FAMILY_BALANCED_CONFIG="$FAMILY_BALANCED_CONFIG" \
  PARENT_PLAN="$PARENT_PLAN" \
  EVENT_EXCLUSIONS="$EVENT_EXCLUSIONS" \
  COHERENCE_CONFIG="$COHERENCE_CONFIG" \
  CACHE_ROOT="$CACHE_ROOT" \
  OUTPUT_ROOT="$MASK_BACKGROUND_OUTPUT_ROOT" \
  SHARD_STOP_EXCLUSIVE="$SHARD_STOP_EXCLUSIVE" \
  bash "$TASK_CODE_DIR/scripts/run_mask_conditioned_background_range.sh"
