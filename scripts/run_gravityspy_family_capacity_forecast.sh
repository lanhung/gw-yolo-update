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
  OUTPUT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

for input in \
  "$TASK_PYTHON" \
  "$EXISTING_TRAIN_REPORT" \
  "$EXISTING_VALIDATION_REPORT" \
  "$TRAIN_PLAN" \
  "$VALIDATION_PLAN" \
  "$PROMOTION_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required family-capacity input is absent: $input" >&2
    exit 2
  fi
done
for count in "$TRAIN_SHARD_COUNT" "$VALIDATION_SHARD_COUNT"; do
  if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "family-capacity shard counts must be positive integers" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD)" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "family-capacity checkout differs from GWYOLO_CODE_COMMIT" >&2
  exit 2
fi

arguments=(
  --materialized-report "$EXISTING_TRAIN_REPORT"
  --materialized-report "$EXISTING_VALIDATION_REPORT"
  --planned-manifest "$TRAIN_PLAN"
  --planned-manifest "$VALIDATION_PLAN"
  --promotion-config "$PROMOTION_CONFIG"
  --validation-fraction 0.2
  --minimum-train-rows-per-family 1
  --seed 20260720
  --output "$OUTPUT"
)
if [[ "${REQUIRE_READY:-0}" == "1" ]]; then
  arguments+=(--require-ready)
elif [[ "${REQUIRE_READY:-0}" != "0" ]]; then
  echo "REQUIRE_READY must be 0 or 1" >&2
  exit 2
fi
for split in train val; do
  if [[ "$split" == train ]]; then
    shard_count=$TRAIN_SHARD_COUNT
  else
    shard_count=$VALIDATION_SHARD_COUNT
  fi
  for ((shard = 0; shard < shard_count; shard++)); do
    report="$INDEPENDENT_OUTPUT_ROOT/$split-shard-$shard/gravityspy_network_numeric_report.json"
    if [[ -s "$report" ]]; then
      arguments+=(--materialized-report "$report")
    fi
  done
done

mkdir -p "$(dirname "$OUTPUT")"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src
exec "$TASK_PYTHON" -m gwyolo.cli \
  gravityspy-network-family-capacity-forecast "${arguments[@]}"
