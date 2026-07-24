#!/usr/bin/env bash
set -euo pipefail

# Bind an already completed source-safe detector-set OOD validation run to the
# publication ledger without repeating GPU training.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_OOD_RECEIPT
  GRAVITYSPY_CORPUS_AUDIT
  OUTPUT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in "$TASK_PYTHON" "$SOURCE_OOD_RECEIPT" "$GRAVITYSPY_CORPUS_AUDIT"; do
  if [[ ! -s "$path" ]]; then
    echo "required OOD binding input is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
observed_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli detector-set-ood-validation-bind \
  --source-receipt "$SOURCE_OOD_RECEIPT" \
  --corpus-audit "$GRAVITYSPY_CORPUS_AUDIT" \
  --output "$OUTPUT"
