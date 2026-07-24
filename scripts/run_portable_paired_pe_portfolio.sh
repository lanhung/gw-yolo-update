#!/usr/bin/env bash
set -euo pipefail

# Reproject transferred within-backend evidence, then evaluate only matched
# within-backend deltas. Absolute DINGO/AMPLFI ranking remains forbidden.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  DINGO_EVIDENCE_BUNDLE_RECEIPT
  AMPLFI_EVIDENCE_BUNDLE_RECEIPT
  DINGO_IMPORT_ROOT
  AMPLFI_IMPORT_ROOT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli pe-within-backend-bundle-import \
  --bundle-receipt "$DINGO_EVIDENCE_BUNDLE_RECEIPT" \
  --output-dir "$DINGO_IMPORT_ROOT"
"$TASK_PYTHON" -m gwyolo.cli pe-within-backend-bundle-import \
  --bundle-receipt "$AMPLFI_EVIDENCE_BUNDLE_RECEIPT" \
  --output-dir "$AMPLFI_IMPORT_ROOT"

export DINGO_WITHIN_SUMMARY="$DINGO_IMPORT_ROOT/within_backend_summary.projected.json"
export AMPLFI_WITHIN_SUMMARY="$AMPLFI_IMPORT_ROOT/within_backend_summary.projected.json"
bash "$TASK_CODE_DIR/scripts/run_paired_pe_portfolio_validation.sh"
