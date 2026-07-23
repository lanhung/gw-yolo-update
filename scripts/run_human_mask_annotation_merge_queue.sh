#!/usr/bin/env bash
set -euo pipefail

# LEGACY OPTIONAL DIAGNOSTIC ONLY. Human consensus is not a publication gate.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  ANNOTATION_TASK_MANIFEST
  ANNOTATION_MANIFEST_A
  ANNOTATION_MANIFEST_B
  ANNOTATION_MANIFEST_C
  COMPLETED_ANNOTATION_MANIFEST
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required human-mask annotation merge variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "human-mask annotation merge requires its declared immutable checkout" >&2
  exit 2
fi
if [[ ! -s "$TASK_PYTHON" || ! -s "$ANNOTATION_TASK_MANIFEST" ]]; then
  echo "human-mask annotation merge lacks its executable or task manifest" >&2
  exit 2
fi
while [[ ! -s "$ANNOTATION_MANIFEST_A" \
  || ! -s "$ANNOTATION_MANIFEST_B" \
  || ! -s "$ANNOTATION_MANIFEST_C" ]]; do
  echo "human-mask annotation merge waiting for three finalized reviewers" >&2
  sleep "${WAIT_SECONDS:-60}"
done
if [[ -e "$COMPLETED_ANNOTATION_MANIFEST" ]]; then
  echo "completed human annotation manifest is immutable" >&2
  exit 2
fi

(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-mask-annotation-merge \
    --tasks "$ANNOTATION_TASK_MANIFEST" \
    --annotation-manifest "$ANNOTATION_MANIFEST_A" \
    --annotation-manifest "$ANNOTATION_MANIFEST_B" \
    --annotation-manifest "$ANNOTATION_MANIFEST_C" \
    --minimum-annotators 3 \
    --output "$COMPLETED_ANNOTATION_MANIFEST"
)
