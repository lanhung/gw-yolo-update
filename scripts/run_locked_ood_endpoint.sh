#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  LOCKED_SUITE_PLAN
  LOCKED_ACCESS_LOG
  OOD_CONFIG
  VALIDATION_OOD_REPORT
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
  "$OOD_CONFIG" \
  "$VALIDATION_OOD_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 2
fi

mapfile -t paths < <(
  "$TASK_PYTHON" - "$LOCKED_SUITE_PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if plan.get("status") != "frozen_locked_evaluation_suite_plan":
    raise SystemExit("locked suite plan is invalid")
print(plan["inputs"]["locked_ood_source_manifest"])
print(plan["inputs"]["locked_ood_score_manifest"])
print(plan["inputs"]["locked_ood_score_report"])
print(plan["outputs"]["locked_ood_transfer"])
PY
)
if (( ${#paths[@]} != 4 )); then
  echo "locked suite plan did not resolve the OOD endpoint" >&2
  exit 2
fi
source_manifest=${paths[0]}
score_manifest=${paths[1]}
score_report=${paths[2]}
endpoint_output=${paths[3]}
if [[ ! -f "$source_manifest" ]]; then
  echo "predeclared locked OOD source manifest is absent: $source_manifest" >&2
  exit 2
fi

if [[ ! -s "$score_report" ]]; then
  mkdir -p "$(dirname "$score_manifest")" "$(dirname "$score_report")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src
    "$TASK_PYTHON" -m gwyolo.cli glitch-ood-score-frozen \
      --config "$OOD_CONFIG" \
      --validation-ood-report "$VALIDATION_OOD_REPORT" \
      --evaluation-manifest "$source_manifest" \
      --output-manifest "$score_manifest" \
      --output-report "$score_report" \
      --required-split test \
      --locked-suite-plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG"
  )
fi

if [[ ! -s "$endpoint_output" ]]; then
  mkdir -p "$(dirname "$endpoint_output")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src
    "$TASK_PYTHON" -m gwyolo.cli ood-abstention-evaluate-locked \
      --validation-ood-report "$VALIDATION_OOD_REPORT" \
      --locked-score-report "$score_report" \
      --locked-score-manifest "$score_manifest" \
      --locked-suite-plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --output "$endpoint_output"
  )
fi
