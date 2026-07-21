#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  SCORING_CODE_DIR
  SCORING_CODE_COMMIT
  BACKGROUND_VAL_MANIFEST
  INJECTION_ARRIVAL_MANIFEST
  BASELINE_CHECKPOINT
  BASELINE_CONFIG
  COHERENCE_CONFIG
  PROMOTED_PIPELINE_REPORT
  BASELINE_OUTPUT_ROOT
  PROMOTION_CONFIG
  COMPARISON_OUTPUT
  GWYOLO_CODE_COMMIT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$BACKGROUND_VAL_MANIFEST" \
  "$INJECTION_ARRIVAL_MANIFEST" \
  "$BASELINE_CHECKPOINT" \
  "$BASELINE_CONFIG" \
  "$COHERENCE_CONFIG" \
  "$PROMOTED_PIPELINE_REPORT" \
  "$PROMOTION_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$SCORING_CODE_DIR/src/gwyolo" ]]; then
  echo "scoring code directory is invalid: $SCORING_CODE_DIR" >&2
  exit 2
fi

while :; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$gpu_pids" ]] && break
  sleep 30
done
(
  cd "$SCORING_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT="$SCORING_CODE_COMMIT"
  "$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-pipeline \
    --background-manifest "$BACKGROUND_VAL_MANIFEST" \
    --injection-manifest "$INJECTION_ARRIVAL_MANIFEST" \
    --checkpoint "$BASELINE_CHECKPOINT" \
    --config "$BASELINE_CONFIG" \
    --coherence-config "$COHERENCE_CONFIG" \
    --output-dir "$BASELINE_OUTPUT_ROOT" \
    --chirp-threshold 0.3 \
    --minimum-bins 1 \
    --timing-association-window-seconds 0.25 \
    --timing-uncertainty-quantile 0.99 \
    --minimum-timing-matches 30 \
    --maximum-timing-uncertainty-seconds 0.01 \
    --slide-count 512 \
    --slide-step-seconds 8 \
    --target-far-per-year 100 \
    --bootstrap-replicates 10000 \
    --seed 20260720
)

"$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-compare \
  --baseline-report "$BASELINE_OUTPUT_ROOT/candidate_validation_pipeline_report.json" \
  --promoted-report "$PROMOTED_PIPELINE_REPORT" \
  --config "$PROMOTION_CONFIG" \
  --output "$COMPARISON_OUTPUT"
