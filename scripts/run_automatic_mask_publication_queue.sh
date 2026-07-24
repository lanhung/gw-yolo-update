#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_CODE_DIR
  TASK_PYTHON
  GWYOLO_CODE_COMMIT
  OVERLAP_VALIDATION_MANIFEST
  OVERLAP_CONFIG
  RAW_MASK_ENDPOINT
  GATE_CONFIG
  OUTPUT_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required automatic-mask queue variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "automatic-mask queue requires its exact checkout" >&2
  exit 3
fi
while true; do
  missing=0
  for path in \
    "$TASK_PYTHON" \
    "$OVERLAP_VALIDATION_MANIFEST" \
    "$OVERLAP_CONFIG" \
    "$RAW_MASK_ENDPOINT" \
    "$GATE_CONFIG"; do
    [[ -s "$path" ]] || missing=$((missing + 1))
  done
  (( missing == 0 )) && break
  sleep "${AUTOMATIC_MASK_POLL_SECONDS:-30}"
done
exec bash "$TASK_CODE_DIR/scripts/run_automatic_mask_publication_evidence.sh"
