#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  LOCKED_SUITE_PLAN
  LOCKED_ACCESS_LOG
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
  "$LOCKED_ACCESS_LOG"; do
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
for key in (
    "catalog_source_manifest",
    "catalog_candidate_manifest",
    "catalog_candidate_report",
    "catalog_prediction_manifest",
    "catalog_prediction_report",
):
    print(plan["inputs"][key])
print(plan["outputs"]["catalog_diagnostic"])
search_arm = plan.get("endpoints", {}).get("catalog_search_arm")
if search_arm not in {"raw_candidate_search", "mask_candidate_search"}:
    raise SystemExit("locked suite catalog search arm is invalid")
print(plan["outputs"][search_arm])
PY
)
if (( ${#paths[@]} != 7 )); then
  echo "locked suite plan did not resolve the catalog endpoint" >&2
  exit 2
fi
source_manifest=${paths[0]}
candidate_manifest=${paths[1]}
candidate_report=${paths[2]}
prediction_manifest=${paths[3]}
prediction_report=${paths[4]}
endpoint_output=${paths[5]}
candidate_search_report=${paths[6]}
for input in \
  "$source_manifest" \
  "$candidate_manifest" \
  "$candidate_report" \
  "$candidate_search_report"; do
  if [[ ! -f "$input" ]]; then
    echo "predeclared locked catalog input is absent: $input" >&2
    exit 2
  fi
done

if [[ ! -s "$prediction_report" ]]; then
  mkdir -p "$(dirname "$prediction_manifest")" "$(dirname "$prediction_report")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src
    "$TASK_PYTHON" -m gwyolo.cli catalog-predict-locked \
      --candidate-manifest "$candidate_manifest" \
      --candidate-report "$candidate_report" \
      --locked-suite-plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --prediction-manifest "$prediction_manifest" \
      --prediction-report "$prediction_report"
  )
fi

if [[ ! -s "$endpoint_output" ]]; then
  mkdir -p "$(dirname "$endpoint_output")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src
    "$TASK_PYTHON" -m gwyolo.cli catalog-eval-locked \
      --prediction-manifest "$prediction_manifest" \
      --prediction-report "$prediction_report" \
      --candidate-search-report "$candidate_search_report" \
      --locked-suite-plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --output "$endpoint_output"
  )
fi
