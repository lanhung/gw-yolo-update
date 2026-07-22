#!/usr/bin/env bash
set -euo pipefail

# Run the predeclared validation-only paired PE portfolio serially so DINGO and
# AMPLFI never race for the GPU. Child steps are resumable and write only to
# explicitly supplied, new output roots. This script never reads a test split.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  MODEL_SELECTION_OVERLAP_MANIFEST
  MODEL_SELECTION_VALIDATION_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  INDEPENDENT_PE_OVERLAP_REPORT
  INDEPENDENT_OVERLAP_AUDIT
  OVERLAP_MANIFEST
  INJECTION_MANIFEST
  DINGO_PYTHON
  DINGO_SOURCE_CONFIG
  DINGO_ACQUISITION_REPORT
  DINGO_MODEL_LOAD_RECEIPT
  DINGO_NATIVE_RUNTIME_RECEIPT
  DINGO_NATIVE_EVENT_SMOKE_SUMMARY
  DINGO_NATIVE_CONDITIONING_CONFIG
  AMPLFI_PYTHON
  AMPLFI_MODEL_METADATA
  AMPLFI_NATIVE_PRIOR
  PE_INPUT_OUTPUT_ROOT
  DINGO_OUTPUT_ROOT
  AMPLFI_OUTPUT_ROOT
  PORTFOLIO_OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
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

validation_limit="${PE_VALIDATION_LIMIT:-100}"
bootstrap_replicates="${PE_BOOTSTRAP_REPLICATES:-10000}"
if (( validation_limit < 100 )); then
  echo "PE_VALIDATION_LIMIT must be at least 100" >&2
  exit 4
fi
if (( bootstrap_replicates < 10000 )); then
  echo "PE_BOOTSTRAP_REPLICATES must be at least 10000" >&2
  exit 4
fi
if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
fi

cd "$TASK_CODE_DIR"
PE_SMOKE_LIMIT="$validation_limit" \
PE_SELECTION_SEED="${PE_SELECTION_SEED:-20260726}" \
OUTPUT_ROOT="$PE_INPUT_OUTPUT_ROOT" \
  bash scripts/run_promoted_paired_pe_smoke.sh

PE_INPUT_ROOT="$PE_INPUT_OUTPUT_ROOT" \
OUTPUT_ROOT="$DINGO_OUTPUT_ROOT" \
PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
  bash scripts/run_dingo_official_native_paired_smoke.sh

PE_INPUT_ROOT="$PE_INPUT_OUTPUT_ROOT" \
OUTPUT_ROOT="$AMPLFI_OUTPUT_ROOT" \
PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
  bash scripts/run_amplfi_within_backend_paired_smoke.sh

DINGO_WITHIN_SUMMARY="$DINGO_OUTPUT_ROOT/dingo_official_native_paired_smoke_summary.json" \
AMPLFI_WITHIN_SUMMARY="$AMPLFI_OUTPUT_ROOT/amplfi_within_backend_paired_smoke_summary.json" \
OUTPUT_ROOT="$PORTFOLIO_OUTPUT_ROOT" \
PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
  bash scripts/run_paired_pe_portfolio_validation.sh

"$TASK_PYTHON" - "$PORTFOLIO_OUTPUT_ROOT/paired_pe_portfolio_summary.json" \
  "$validation_limit" "$bootstrap_replicates" <<'PY'
import json
import pathlib
import sys


summary_path = pathlib.Path(sys.argv[1])
minimum_injections = int(sys.argv[2])
minimum_bootstraps = int(sys.argv[3])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if (
    summary.get("status") != "validation_only_paired_pe_portfolio_complete"
    or summary.get("evaluation_tier") != "publication_validation"
    or int(summary.get("paired_injections", -1)) < minimum_injections
    or int(summary.get("bootstrap_replicates", -1)) < minimum_bootstraps
    or summary.get("test_rows_read") != 0
    or summary.get("absolute_cross_backend_comparison_allowed") is not False
):
    raise SystemExit("paired PE publication-validation portfolio failed closed")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
