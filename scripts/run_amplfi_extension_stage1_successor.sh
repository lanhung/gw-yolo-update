#!/usr/bin/env bash
set -euo pipefail

# Convert a passing streamed-capacity extension into a hash-bound no-copy bank
# view, then launch publication-stage-1 AMPLFI training. The caller must wait
# for the extension producer before invoking this script.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  AMPLFI_PYTHON
  AMPLFI_CLI
  BACKGROUND_RECEIPT
  BACKGROUND_DATA_DIR
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required AMPLFI extension successor variable is unset: $variable" >&2
    exit 2
  fi
done
for path in "$TASK_PYTHON" "$AMPLFI_PYTHON" "$AMPLFI_CLI" "$BACKGROUND_RECEIPT"; do
  if [[ ! -s "$path" ]]; then
    echo "required AMPLFI extension successor input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "AMPLFI extension successor requires its exact checkout" >&2
  exit 3
fi
bank_report="$BACKGROUND_DATA_DIR/amplfi_training_bank_report.json"
if [[ ! -s "$bank_report" ]]; then
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src
  export GWYOLO_CODE_COMMIT
  "$TASK_PYTHON" -m gwyolo.cli amplfi-training-bank-freeze \
    --background-receipt "$BACKGROUND_RECEIPT" \
    --output-dir "$BACKGROUND_DATA_DIR"
fi
if [[ ! -s "$bank_report" ]]; then
  echo "AMPLFI extension successor omitted its frozen bank report" >&2
  exit 4
fi
export BACKGROUND_BANK_REPORT="$bank_report"
bash "$TASK_CODE_DIR/scripts/run_amplfi_publication_stage1.sh"
