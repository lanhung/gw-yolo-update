#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BACKGROUND_MANIFEST
  BASELINE_PIPELINE_REPORT
  PROMOTED_PIPELINE_REPORT
  BASELINE_CALIBRATED_CANDIDATES
  PROMOTED_CALIBRATED_CANDIDATES
  BASELINE_INJECTION_RANKING_REPORT
  PROMOTED_INJECTION_RANKING_REPORT
  PROMOTION_CONFIG
  OUTPUT_ROOT
  COMPARISON_OUTPUT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$BACKGROUND_MANIFEST" \
  "$BASELINE_PIPELINE_REPORT" \
  "$PROMOTED_PIPELINE_REPORT" \
  "$BASELINE_CALIBRATED_CANDIDATES" \
  "$PROMOTED_CALIBRATED_CANDIDATES" \
  "$BASELINE_INJECTION_RANKING_REPORT" \
  "$PROMOTED_INJECTION_RANKING_REPORT" \
  "$PROMOTION_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "task code directory is invalid: $TASK_CODE_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
for arm in baseline promoted; do
  if [[ "$arm" == "baseline" ]]; then
    pipeline_report=$BASELINE_PIPELINE_REPORT
    candidates=$BASELINE_CALIBRATED_CANDIDATES
    rankings=$BASELINE_INJECTION_RANKING_REPORT
  else
    pipeline_report=$PROMOTED_PIPELINE_REPORT
    candidates=$PROMOTED_CALIBRATED_CANDIDATES
    rankings=$PROMOTED_INJECTION_RANKING_REPORT
  fi
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-block-recalibrate \
      --pipeline-report "$pipeline_report" \
      --background-manifest "$BACKGROUND_MANIFEST" \
      --calibrated-candidate-manifest "$candidates" \
      --injection-ranking-report "$rankings" \
      --output-dir "$OUTPUT_ROOT/$arm"
  )
done

if [[ ! -s "$COMPARISON_OUTPUT" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-compare \
      --baseline-report \
      "$OUTPUT_ROOT/baseline/candidate_validation_block_pipeline_report.json" \
      --promoted-report \
      "$OUTPUT_ROOT/promoted/candidate_validation_block_pipeline_report.json" \
      --config "$PROMOTION_CONFIG" \
      --output "$COMPARISON_OUTPUT"
  )
fi
