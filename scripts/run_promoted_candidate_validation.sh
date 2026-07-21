#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  WAVEFORM_PYTHON
  FIVE_SEED_SUMMARY
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
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
  "$WAVEFORM_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$COHERENCE_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done

readarray -t selection < <("$TASK_PYTHON" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "completed_five_seed_source_safe_overlap_validation":
    raise SystemExit("five-seed summary has the wrong status")
print(report["promoted_arm"])
print(report["selected_checkpoint_path"])
' "$FIVE_SEED_SUMMARY")
arm=${selection[0]}
checkpoint=${selection[1]}
if [[ "$arm" == uniform ]]; then
  config=$UNIFORM_CONFIG
elif [[ "$arm" == family_balanced ]]; then
  config=$FAMILY_BALANCED_CONFIG
else
  echo "five-seed summary selected an unknown arm: $arm" >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "selected checkpoint is absent: $checkpoint" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
"$TASK_PYTHON" -m gwyolo.cli manifest-select-split \
  --manifest "$BACKGROUND_MANIFEST" \
  --split val \
  --output-dir "$OUTPUT_ROOT/background-val"
"$WAVEFORM_PYTHON" -m gwyolo.cli injection-arrival-annotate \
  --manifest "$INJECTION_MANIFEST" \
  --output-dir "$OUTPUT_ROOT/injection-arrivals"

while :; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$gpu_pids" ]] && break
  sleep 30
done
"$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-pipeline \
  --background-manifest "$OUTPUT_ROOT/background-val/val_manifest.jsonl" \
  --injection-manifest "$OUTPUT_ROOT/injection-arrivals/materialized_injections_arrivals.jsonl" \
  --checkpoint "$checkpoint" \
  --config "$config" \
  --coherence-config "$COHERENCE_CONFIG" \
  --output-dir "$OUTPUT_ROOT/pipeline" \
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
  --seed 20260720 \
  --model-selection-report "$FIVE_SEED_SUMMARY"
