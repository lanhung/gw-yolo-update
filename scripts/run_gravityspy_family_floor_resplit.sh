#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  EXISTING_TRAIN_REPORT
  EXISTING_VALIDATION_REPORT
  INDEPENDENT_OUTPUT_ROOT
  TRAIN_PLAN
  VALIDATION_PLAN
  TRAIN_SHARD_COUNT
  VALIDATION_SHARD_COUNT
  PROMOTION_CONFIG
  CAPACITY_OUTPUT
  TRAIN_EXPANDED_REPORT
  VALIDATION_EXPANDED_REPORT
  RESPLIT_OUTPUT_ROOT
  CORPUS_AUDIT_OUTPUT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  EXISTING_TRAIN_REPORT="$EXISTING_TRAIN_REPORT" \
  EXISTING_VALIDATION_REPORT="$EXISTING_VALIDATION_REPORT" \
  INDEPENDENT_OUTPUT_ROOT="$INDEPENDENT_OUTPUT_ROOT" \
  TRAIN_PLAN="$TRAIN_PLAN" \
  VALIDATION_PLAN="$VALIDATION_PLAN" \
  TRAIN_SHARD_COUNT="$TRAIN_SHARD_COUNT" \
  VALIDATION_SHARD_COUNT="$VALIDATION_SHARD_COUNT" \
  PROMOTION_CONFIG="$PROMOTION_CONFIG" \
  OUTPUT="$CAPACITY_OUTPUT" \
  REQUIRE_READY=1 \
  bash "$TASK_CODE_DIR/scripts/run_gravityspy_family_capacity_forecast.sh"

for input in "$TRAIN_EXPANDED_REPORT" "$VALIDATION_EXPANDED_REPORT"; do
  if [[ ! -s "$input" ]]; then
    echo "expanded merge report is absent: $input" >&2
    exit 2
  fi
done
mkdir -p "$RESPLIT_OUTPUT_ROOT" "$(dirname "$CORPUS_AUDIT_OUTPUT")"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-corpus-resplit \
  --report "$TRAIN_EXPANDED_REPORT" \
  --report "$VALIDATION_EXPANDED_REPORT" \
  --output-dir "$RESPLIT_OUTPUT_ROOT" \
  --validation-fraction 0.2 \
  --minimum-validation-rows-per-family 5 \
  --seed 20260720
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-corpus-audit \
  --train-report "$RESPLIT_OUTPUT_ROOT/gravityspy_network_numeric_train_report.json" \
  --validation-report "$RESPLIT_OUTPUT_ROOT/gravityspy_network_numeric_val_report.json" \
  --output "$CORPUS_AUDIT_OUTPUT"
