#!/usr/bin/env bash
set -euo pipefail

# Validation-only replacement for the legacy three-annotator mask gate. It
# replays masks from isolated physical components and binds them to the
# functional raw/mask continuous-background endpoint. It never opens O4b.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  OVERLAP_VALIDATION_MANIFEST
  OVERLAP_CONFIG
  RAW_MASK_ENDPOINT
  GATE_CONFIG
  OUTPUT_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required automatic-mask publication variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "automatic-mask publication requires its exact checkout" >&2
  exit 3
fi
for path in \
  "$TASK_PYTHON" \
  "$OVERLAP_VALIDATION_MANIFEST" \
  "$OVERLAP_CONFIG" \
  "$RAW_MASK_ENDPOINT" \
  "$GATE_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "automatic-mask publication input is absent: $path" >&2
    exit 3
  fi
done

audit="$OUTPUT_DIR/automatic_mask_policy_audit.json"
endpoint="$OUTPUT_DIR/raw_mask_automatic_endpoint.json"
mkdir -p "$OUTPUT_DIR"
run_cli() {
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli "$@"
  )
}
if [[ ! -e "$audit" ]]; then
  run_cli automatic-mask-policy-audit \
    --overlap-manifest "$OVERLAP_VALIDATION_MANIFEST" \
    --overlap-config "$OVERLAP_CONFIG" \
    --output "$audit"
fi
if [[ ! -e "$endpoint" ]]; then
  run_cli candidate-search-raw-mask-automatic-endpoint-bind \
    --raw-mask-endpoint "$RAW_MASK_ENDPOINT" \
    --automatic-mask-audit "$audit" \
    --gate-config "$GATE_CONFIG" \
    --output "$endpoint"
fi

"$TASK_PYTHON" - "$endpoint" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "bound_validation_raw_mask_automatic_evidence"
    or report.get("passed") is not True
    or report.get("code_commit") != sys.argv[2]
    or report.get("human_annotation_required") is not False
    or report.get("human_annotation_used") is not False
    or report.get("test_rows_read") != 0
    or report.get("scientific_claim_allowed") is not False
):
    raise SystemExit("automatic-mask publication endpoint failed replay")
print(endpoint := pathlib.Path(sys.argv[1]).resolve())
PY
