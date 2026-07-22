#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  LOCKED_SUITE_PLAN
  LOCKED_ACCESS_LOG
  VALIDATION_PE_PROMOTION_REPORT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$LOCKED_SUITE_PLAN" \
  "$LOCKED_ACCESS_LOG" \
  "$VALIDATION_PE_PROMOTION_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 2
fi

mapfile -t locked_outputs < <(
  "$TASK_PYTHON" - "$LOCKED_SUITE_PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if plan.get("status") != "frozen_locked_evaluation_suite_plan":
    raise SystemExit("locked suite plan is invalid")
for key in ("dingo_batch", "amplfi_batch", "joint_pe"):
    print(plan["outputs"][key])
for key in ("dingo_locked_source_batch_report", "amplfi_locked_source_batch_report"):
    print(plan["inputs"][key])
PY
)
if (( ${#locked_outputs[@]} != 5 )); then
  echo "locked suite plan did not resolve the PE outputs" >&2
  exit 2
fi
dingo_output=${locked_outputs[0]}
amplfi_output=${locked_outputs[1]}
joint_output=${locked_outputs[2]}
DINGO_LOCKED_BATCH_REPORT=${locked_outputs[3]}
AMPLFI_LOCKED_BATCH_REPORT=${locked_outputs[4]}
for input in "$DINGO_LOCKED_BATCH_REPORT" "$AMPLFI_LOCKED_BATCH_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "predeclared locked PE source batch is absent: $input" >&2
    exit 2
  fi
done

bind_backend() {
  local backend=$1
  local source_report=$2
  local output=$3
  if [[ ! -s "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src
      "$TASK_PYTHON" -m gwyolo.cli pe-backend-bind-locked \
        --backend "$backend" \
        --batch-report "$source_report" \
        --validation-promotion-report "$VALIDATION_PE_PROMOTION_REPORT" \
        --locked-suite-plan "$LOCKED_SUITE_PLAN" \
        --access-log "$LOCKED_ACCESS_LOG" \
        --output "$output"
    )
  fi
}

bind_backend DINGO "$DINGO_LOCKED_BATCH_REPORT" "$dingo_output"
bind_backend AMPLFI "$AMPLFI_LOCKED_BATCH_REPORT" "$amplfi_output"

if [[ ! -s "$joint_output" ]]; then
  mkdir -p "$(dirname "$joint_output")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src
    "$TASK_PYTHON" -m gwyolo.cli pe-robustness-joint-evaluate-locked \
      --dingo-locked-report "$dingo_output" \
      --amplfi-locked-report "$amplfi_output" \
      --validation-promotion-report "$VALIDATION_PE_PROMOTION_REPORT" \
      --locked-suite-plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --output "$joint_output"
  )
fi
