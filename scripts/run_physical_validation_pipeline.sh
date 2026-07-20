#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 CONFIG TRAIN_SNR_MANIFEST VALIDATION_SNR_MANIFEST PRETRAINED_CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

if [ -z "${GWYOLO_CODE_COMMIT:-}" ]; then
  echo "GWYOLO_CODE_COMMIT must identify the exact deployed code" >&2
  exit 2
fi

config_path=$1
train_manifest=$2
validation_manifest=$3
pretrained_checkpoint=$4
output_root=$5
task_python=${GWYOLO_TRAIN_PYTHON:-python}

mkdir -p "$output_root"
curriculum_dir="$output_root/curriculum"
finetune_dir="$output_root/finetune"

PYTHONPATH=src "$task_python" -m gwyolo.cli physical-snr-curriculum \
  --manifest "$train_manifest" \
  --output-dir "$curriculum_dir" \
  --minimum-snr 4 \
  --rescale-upper-snr 8 \
  --seed 20260720

PYTHONPATH=src "$task_python" -m gwyolo.cli physical-finetune \
  --config "$config_path" \
  --train-manifest "$curriculum_dir/physical_train_snr_curriculum.jsonl" \
  --validation-manifest "$validation_manifest" \
  --pretrained-checkpoint "$pretrained_checkpoint" \
  --output-dir "$finetune_dir" \
  --seed 20260720

selected_threshold=$(
  "$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_chirp_threshold"])' \
    "$finetune_dir/physical_finetune_report.json"
)

PYTHONPATH=src "$task_python" -m gwyolo.cli physical-checkpoint-audit \
  --config "$config_path" \
  --validation-manifest "$validation_manifest" \
  --checkpoint "$finetune_dir/best_physical_finetune.pt" \
  --chirp-threshold "$selected_threshold" \
  --output "$finetune_dir/physical_checkpoint_stratified_audit.json"
