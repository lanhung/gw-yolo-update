#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TRAIN_GLITCH_MANIFEST
  VAL_GLITCH_MANIFEST
  TRAIN_INJECTION_MANIFEST
  VAL_INJECTION_MANIFEST
  PRETRAINED_CHECKPOINT
  MATERIALIZATION_CONFIG
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

SEED=${SEED:-20260720}
CLEAN_VALIDATION_FEATURE_CACHE_DIR=${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}
for input in \
  "$TASK_PYTHON" \
  "$TRAIN_GLITCH_MANIFEST" \
  "$VAL_GLITCH_MANIFEST" \
  "$TRAIN_INJECTION_MANIFEST" \
  "$VAL_INJECTION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$MATERIALIZATION_CONFIG" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT"

train_overlap="$OUTPUT_ROOT/train-overlaps"
val_overlap="$OUTPUT_ROOT/val-overlaps"
if [[ ! -s "$train_overlap/physical_overlap_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-materialize \
    --gravityspy-manifest "$TRAIN_GLITCH_MANIFEST" \
    --injection-manifest "$TRAIN_INJECTION_MANIFEST" \
    --config "$MATERIALIZATION_CONFIG" \
    --output-dir "$train_overlap" \
    --split train \
    --seed "$SEED"
fi
if [[ ! -s "$val_overlap/physical_overlap_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-materialize \
    --gravityspy-manifest "$VAL_GLITCH_MANIFEST" \
    --injection-manifest "$VAL_INJECTION_MANIFEST" \
    --config "$MATERIALIZATION_CONFIG" \
    --output-dir "$val_overlap" \
    --split val \
    --seed "$SEED"
fi
audit="$OUTPUT_ROOT/physical_overlap_audit.json"
if [[ ! -s "$audit" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-audit \
    --manifest "$train_overlap/physical_overlap_train_manifest.jsonl" \
    --manifest "$val_overlap/physical_overlap_val_manifest.jsonl" \
    --output "$audit"
fi

cache_args=()
if [[ -n "$CLEAN_VALIDATION_FEATURE_CACHE_DIR" ]]; then
  cache_args=(--clean-validation-feature-cache-dir "$CLEAN_VALIDATION_FEATURE_CACHE_DIR")
fi
for arm in uniform family-balanced; do
  if [[ "$arm" == uniform ]]; then
    config=$UNIFORM_CONFIG
  else
    config=$FAMILY_BALANCED_CONFIG
  fi
  while :; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-finetune \
    --config "$config" \
    --overlap-train-manifest "$train_overlap/physical_overlap_train_manifest.jsonl" \
    --overlap-validation-manifest "$val_overlap/physical_overlap_val_manifest.jsonl" \
    --clean-train-manifest "$TRAIN_INJECTION_MANIFEST" \
    --clean-validation-manifest "$VAL_INJECTION_MANIFEST" \
    --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
    "${cache_args[@]}" \
    --output-dir "$OUTPUT_ROOT/$arm-seed$SEED" \
    --seed "$SEED"
done
