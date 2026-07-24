#!/usr/bin/env bash
set -euo pipefail

# LEGACY OPTIONAL DIAGNOSTIC ONLY. This script is no longer part of the
# publication validation ledger or the O4b/GWTC-5 unlock path.

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
    echo "required human-mask publication variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "human-mask publication chain requires its declared immutable checkout" >&2
  exit 2
fi
TASK_MANIFEST="$AUDIT_PLAN_DIR/gravityspy_mask_audit_tasks.jsonl"
for path in \
  "$TASK_PYTHON" \
  "$TASK_MANIFEST" \
  "$COMPLETED_ANNOTATION_MANIFEST" \
  "$MODEL_SELECTION_REPORT" \
  "$MODEL_CONFIG" \
  "$RAW_MASK_ENDPOINT" \
  "$GATE_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "human-mask publication input is absent: $path" >&2
    exit 2
  fi
done

AUDIT_REPORT="$OUTPUT_DIR/gravityspy_human_mask_audit.json"
GOLD_DIR="$OUTPUT_DIR/human-consensus-gold"
GOLD_REPORT="$GOLD_DIR/gravityspy_human_consensus_mask_report.json"
PREDICTION_DIR="$OUTPUT_DIR/model-predictions"
PREDICTION_MANIFEST="$PREDICTION_DIR/gravityspy_mask_segmentation_predictions.jsonl"
SEGMENTATION_REPORT="$OUTPUT_DIR/human_consensus_mask_segmentation.json"
BOUND_ENDPOINT="$OUTPUT_DIR/raw_mask_human_consensus_endpoint.json"
mkdir -p "$OUTPUT_DIR"

run_cli() {
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli "$@"
  )
}
if [[ ! -e "$AUDIT_REPORT" ]]; then
  run_cli gravityspy-mask-audit-evaluate \
    --tasks "$TASK_MANIFEST" \
    --annotations "$COMPLETED_ANNOTATION_MANIFEST" \
    --output "$AUDIT_REPORT"
fi
if [[ ! -e "$GOLD_REPORT" ]]; then
  run_cli gravityspy-mask-consensus-materialize \
    --tasks "$TASK_MANIFEST" \
    --annotations "$COMPLETED_ANNOTATION_MANIFEST" \
    --audit-report "$AUDIT_REPORT" \
    --output-dir "$GOLD_DIR"
fi
if [[ ! -e "$PREDICTION_MANIFEST" ]]; then
  run_cli gravityspy-mask-segmentation-predict \
    --gold-report "$GOLD_REPORT" \
    --selection-report "$MODEL_SELECTION_REPORT" \
    --config "$MODEL_CONFIG" \
    --output-dir "$PREDICTION_DIR"
fi
if [[ ! -e "$SEGMENTATION_REPORT" ]]; then
  run_cli gravityspy-mask-segmentation-evaluate \
    --gold-report "$GOLD_REPORT" \
    --predictions "$PREDICTION_MANIFEST" \
    --bootstrap-replicates "${MASK_BOOTSTRAP_REPLICATES:-10000}" \
    --bootstrap-seed "${MASK_BOOTSTRAP_SEED:-20260720}" \
    --output "$SEGMENTATION_REPORT"
fi
if [[ ! -e "$BOUND_ENDPOINT" ]]; then
  run_cli candidate-search-raw-mask-human-endpoint-bind \
    --raw-mask-endpoint "$RAW_MASK_ENDPOINT" \
    --human-mask-segmentation-report "$SEGMENTATION_REPORT" \
    --gate-config "$GATE_CONFIG" \
    --output "$BOUND_ENDPOINT"
fi

"$TASK_PYTHON" - "$BOUND_ENDPOINT" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (
    report.get("status") != "bound_validation_raw_mask_human_consensus_evidence"
    or report.get("code_commit") != sys.argv[2]
    or report.get("scientific_claim_allowed") is not False
    or report.get("test_rows_read") != 0
):
    raise SystemExit("human-mask publication endpoint failed replay")
if report.get("passed") is not True:
    raise SystemExit("human-mask publication endpoint retained a negative gate result")
PY
