#!/usr/bin/env bash
set -euo pipefail

# LEGACY OPTIONAL DIAGNOSTIC ONLY. Do not place this queue on the publication
# critical path; use run_automatic_mask_publication_evidence.sh.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  AUDIT_PLAN_DIR
  COMPLETED_ANNOTATION_MANIFEST
  MODEL_SELECTION_REPORT
  MODEL_CONFIG
  RAW_MASK_ENDPOINT
  GATE_CONFIG
  OUTPUT_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required human-mask publication queue variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "human-mask publication queue requires its declared immutable checkout" >&2
  exit 2
fi

inputs=(
  "$AUDIT_PLAN_DIR/gravityspy_mask_audit_tasks.jsonl"
  "$COMPLETED_ANNOTATION_MANIFEST"
  "$MODEL_SELECTION_REPORT"
  "$MODEL_CONFIG"
  "$RAW_MASK_ENDPOINT"
  "$GATE_CONFIG"
)
while true; do
  missing=()
  for path in "${inputs[@]}"; do
    [[ -s "$path" ]] || missing+=("$path")
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    break
  fi
  echo "human-mask publication queue waiting for ${#missing[@]} inputs" >&2
  sleep "${WAIT_SECONDS:-60}"
done

while command -v nvidia-smi >/dev/null 2>&1 \
  && [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; do
  echo "human-mask publication queue waiting for GPU idle" >&2
  sleep "${WAIT_SECONDS:-60}"
done

exec bash "$TASK_CODE_DIR/scripts/run_human_mask_publication_evidence.sh"
