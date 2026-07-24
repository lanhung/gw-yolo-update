#!/usr/bin/env bash
set -euo pipefail

# Import a hash-verified shared PE input bundle, run validation-only AMPLFI
# robustness, then export portable within-backend evidence.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PE_INPUT_BUNDLE_RECEIPT
  PE_INPUT_ROOT
  OUTPUT_ROOT
  EVIDENCE_BUNDLE_ROOT
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
"$TASK_PYTHON" -m gwyolo.cli pe-input-bundle-import \
  --bundle-receipt "$PE_INPUT_BUNDLE_RECEIPT" \
  --output-dir "$PE_INPUT_ROOT"

bash "$TASK_CODE_DIR/scripts/run_amplfi_within_backend_paired_smoke.sh"

"$TASK_PYTHON" -m gwyolo.cli pe-within-backend-bundle-export \
  --summary "$OUTPUT_ROOT/amplfi_within_backend_paired_smoke_summary.json" \
  --output-dir "$EVIDENCE_BUNDLE_ROOT"
