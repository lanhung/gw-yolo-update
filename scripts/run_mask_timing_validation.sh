#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  MASK_VALIDATION_RECEIPT
  PIPELINE_REPORT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

config=${MASK_TIMING_CONFIG:-$TASK_CODE_DIR/configs/mask_timing_validation.yaml}
for path in "$TASK_PYTHON" "$MASK_VALIDATION_RECEIPT" "$PIPELINE_REPORT" "$config"; do
  if [[ ! -s "$path" ]]; then
    echo "required mask timing artifact is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli mask-timing-validation \
  --mask-validation-receipt "$MASK_VALIDATION_RECEIPT" \
  --pipeline-report "$PIPELINE_REPORT" \
  --config "$config" \
  --output "$OUTPUT_ROOT/mask_timing_validation_receipt.json"
