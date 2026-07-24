#!/usr/bin/env bash
set -euo pipefail

# Build the validation-only paired-PE evidence in two serial stages:
# 1. a bounded runtime proof on fewer than 100 events;
# 2. the predeclared >=100-event, >=10000-bootstrap publication validation.
#
# This wrapper intentionally does not wait for model or backend artifacts. A
# remote scheduler may wait for their producer PIDs, but every dependency must
# exist before this script starts so a failed producer cannot leave an
# unbounded waiter.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  MODEL_SELECTION_TRAIN_OVERLAP_MANIFEST
  MODEL_SELECTION_VALIDATION_OVERLAP_MANIFEST
  MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST
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
  SMOKE_PE_INPUT_OUTPUT_ROOT
  SMOKE_DINGO_OUTPUT_ROOT
  SMOKE_AMPLFI_OUTPUT_ROOT
  SMOKE_PORTFOLIO_OUTPUT_ROOT
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

source_paths=(
  "$TASK_PYTHON"
  "$FIVE_SEED_SUMMARY"
  "$UNIFORM_CONFIG"
  "$FAMILY_BALANCED_CONFIG"
  "$MODEL_SELECTION_TRAIN_OVERLAP_MANIFEST"
  "$MODEL_SELECTION_VALIDATION_OVERLAP_MANIFEST"
  "$MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST"
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT"
  "$INDEPENDENT_PE_OVERLAP_REPORT"
  "$INDEPENDENT_OVERLAP_AUDIT"
  "$OVERLAP_MANIFEST"
  "$INJECTION_MANIFEST"
  "$DINGO_PYTHON"
  "$DINGO_SOURCE_CONFIG"
  "$DINGO_ACQUISITION_REPORT"
  "$DINGO_MODEL_LOAD_RECEIPT"
  "$DINGO_NATIVE_RUNTIME_RECEIPT"
  "$DINGO_NATIVE_EVENT_SMOKE_SUMMARY"
  "$DINGO_NATIVE_CONDITIONING_CONFIG"
  "$AMPLFI_PYTHON"
  "$AMPLFI_MODEL_METADATA"
  "$AMPLFI_NATIVE_PRIOR"
)
for path in "${source_paths[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "required paired PE ladder input is absent: $path" >&2
    exit 3
  fi
done

smoke_limit="${PE_SMOKE_LIMIT:-3}"
validation_limit="${PE_VALIDATION_LIMIT:-100}"
bootstrap_replicates="${PE_BOOTSTRAP_REPLICATES:-10000}"
if (( smoke_limit < 1 || smoke_limit >= 100 )); then
  echo "PE_SMOKE_LIMIT must be in [1, 99]" >&2
  exit 4
fi
if (( validation_limit < 100 )); then
  echo "PE_VALIDATION_LIMIT must be at least 100" >&2
  exit 4
fi
if (( bootstrap_replicates < 10000 )); then
  echo "PE_BOOTSTRAP_REPLICATES must be at least 10000" >&2
  exit 4
fi

prior_summary="$SMOKE_PORTFOLIO_OUTPUT_ROOT/paired_pe_portfolio_summary.json"
if [[ ! -s "$prior_summary" ]]; then
  PE_SMOKE_LIMIT="$smoke_limit" \
  OUTPUT_ROOT="$SMOKE_PE_INPUT_OUTPUT_ROOT" \
    bash "$TASK_CODE_DIR/scripts/run_promoted_paired_pe_smoke.sh"

  PE_INPUT_ROOT="$SMOKE_PE_INPUT_OUTPUT_ROOT" \
  OUTPUT_ROOT="$SMOKE_DINGO_OUTPUT_ROOT" \
  PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
    bash "$TASK_CODE_DIR/scripts/run_dingo_official_native_paired_smoke.sh"

  PE_INPUT_ROOT="$SMOKE_PE_INPUT_OUTPUT_ROOT" \
  OUTPUT_ROOT="$SMOKE_AMPLFI_OUTPUT_ROOT" \
  PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
    bash "$TASK_CODE_DIR/scripts/run_amplfi_within_backend_paired_smoke.sh"

  DINGO_WITHIN_SUMMARY="$SMOKE_DINGO_OUTPUT_ROOT/dingo_official_native_paired_smoke_summary.json" \
  AMPLFI_WITHIN_SUMMARY="$SMOKE_AMPLFI_OUTPUT_ROOT/amplfi_within_backend_paired_smoke_summary.json" \
  OUTPUT_ROOT="$SMOKE_PORTFOLIO_OUTPUT_ROOT" \
  PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
    bash "$TASK_CODE_DIR/scripts/run_paired_pe_portfolio_validation.sh"
fi

PRIOR_SMOKE_PORTFOLIO_SUMMARY="$prior_summary" \
PE_VALIDATION_LIMIT="$validation_limit" \
PE_BOOTSTRAP_REPLICATES="$bootstrap_replicates" \
  bash "$TASK_CODE_DIR/scripts/run_paired_pe_publication_validation.sh"

