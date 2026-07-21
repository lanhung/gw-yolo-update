#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  PROMOTION_REPORT
  ORIGINAL_UNIFORM_REPORT
  ORIGINAL_FAMILY_BALANCED_REPORT
  OVERLAP_TRAIN_MANIFEST
  OVERLAP_VALIDATION_MANIFEST
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  OUTPUT_ROOT
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
  "$PROMOTION_REPORT" \
  "$ORIGINAL_UNIFORM_REPORT" \
  "$ORIGINAL_FAMILY_BALANCED_REPORT" \
  "$OVERLAP_TRAIN_MANIFEST" \
  "$OVERLAP_VALIDATION_MANIFEST" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done

arm=$("$TASK_PYTHON" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("passed") and report.get("scale_to_five_seeds"):
    print(report.get("promoted_arm", ""))
' "$PROMOTION_REPORT")
if [[ "$arm" == uniform ]]; then
  config=$UNIFORM_CONFIG
  original_report=$ORIGINAL_UNIFORM_REPORT
elif [[ "$arm" == family_balanced ]]; then
  config=$FAMILY_BALANCED_CONFIG
  original_report=$ORIGINAL_FAMILY_BALANCED_REPORT
else
  echo "validation promotion did not authorize five-seed expansion"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
reports=(--report "$original_report")
cache_args=()
if [[ -n "${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" ]]; then
  cache_args=(--clean-validation-feature-cache-dir "$CLEAN_VALIDATION_FEATURE_CACHE_DIR")
fi
for seed in 20260721 20260722 20260723 20260724; do
  while :; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  run="$OUTPUT_ROOT/${arm}-seed${seed}"
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-finetune \
    --config "$config" \
    --overlap-train-manifest "$OVERLAP_TRAIN_MANIFEST" \
    --overlap-validation-manifest "$OVERLAP_VALIDATION_MANIFEST" \
    --clean-train-manifest "$CLEAN_TRAIN_MANIFEST" \
    --clean-validation-manifest "$CLEAN_VALIDATION_MANIFEST" \
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
    "${cache_args[@]}" \
    --output-dir "$run" \
    --seed "$seed"
  reports+=(--report "$run/overlap_finetune_report.json")
done

"$TASK_PYTHON" -m gwyolo.cli physical-overlap-five-seed-summarize \
  --promotion-report "$PROMOTION_REPORT" \
  "${reports[@]}" \
  --output "$OUTPUT_ROOT/five_seed_overlap_summary.json"
