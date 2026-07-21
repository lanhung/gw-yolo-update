#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  PARENT_PLAN
  EVENT_EXCLUSIONS
  CHECKPOINT
  CONFIG
  COHERENCE_CONFIG
  CACHE_ROOT
  OUTPUT_ROOT
  SHARD_STOP_EXCLUSIVE
  GWYOLO_CODE_COMMIT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

SHARD_START=${SHARD_START:-0}
PAIRS_PER_SHARD=${PAIRS_PER_SHARD:-4}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.2}
SEED=${SEED:-20260719}
MODEL_IFOS=${MODEL_IFOS:-"H1 L1 V1"}
Q_VALUES=${Q_VALUES:-"4 8 16"}
TARGET_SAMPLE_RATE=${TARGET_SAMPLE_RATE:-1024}
CONTEXT_DURATION=${CONTEXT_DURATION:-64}
CHIRP_THRESHOLD=${CHIRP_THRESHOLD:-0.5}
MINIMUM_BINS=${MINIMUM_BINS:-1}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-2}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
TARGET_RATE_PER_DETECTOR_YEAR=${TARGET_RATE_PER_DETECTOR_YEAR:-8766}

if ! [[ "$SHARD_START" =~ ^[0-9]+$ ]] \
  || ! [[ "$SHARD_STOP_EXCLUSIVE" =~ ^[0-9]+$ ]] \
  || ! [[ "$PAIRS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] \
  || (( SHARD_STOP_EXCLUSIVE <= SHARD_START )); then
  echo "invalid positive half-open shard range or pairs-per-shard" >&2
  exit 2
fi

for input in \
  "$TASK_PYTHON" \
  "$PARENT_PLAN" \
  "$EVENT_EXCLUSIONS" \
  "$CHECKPOINT" \
  "$CONFIG" \
  "$COHERENCE_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done

read -r -a model_ifos <<<"$MODEL_IFOS"
read -r -a q_values <<<"$Q_VALUES"
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
reports=()

for ((shard = SHARD_START; shard < SHARD_STOP_EXCLUSIVE; shard++)); do
  available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < MINIMUM_FREE_KB )); then
    echo "insufficient cache filesystem space before shard $shard" >&2
    exit 1
  fi
  while :; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  shard_output="$OUTPUT_ROOT/shard-$shard"
  "$TASK_PYTHON" -m gwyolo.cli background-morphology-stream-shard \
    --parent-plan "$PARENT_PLAN" \
    --event-exclusions "$EVENT_EXCLUSIONS" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --coherence-config "$COHERENCE_CONFIG" \
    --cache-root "$CACHE_ROOT" \
    --output-dir "$shard_output" \
    --shard-index "$shard" \
    --pairs-per-shard "$PAIRS_PER_SHARD" \
    --validation-fraction "$VALIDATION_FRACTION" \
    --seed "$SEED" \
    --model-ifos "${model_ifos[@]}" \
    --q-values "${q_values[@]}" \
    --target-sample-rate "$TARGET_SAMPLE_RATE" \
    --context-duration "$CONTEXT_DURATION" \
    --chirp-threshold "$CHIRP_THRESHOLD" \
    --minimum-bins "$MINIMUM_BINS" \
    --download-workers "$DOWNLOAD_WORKERS"
  report="$shard_output/streamed_background_shard_report.json"
  if [[ ! -s "$report" ]]; then
    echo "streaming shard completed without its immutable report: $shard" >&2
    exit 1
  fi
  reports+=(--shard-report "$report")
done

merge_dir="$OUTPUT_ROOT/merged"
merge_report="$merge_dir/streamed_background_merge_report.json"
if [[ ! -s "$merge_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli background-stream-merge \
    "${reports[@]}" \
    --output-dir "$merge_dir"
fi
"$TASK_PYTHON" -m gwyolo.cli background-morphology-calibrate \
  --merge-report "$merge_report" \
  --target-rate-per-detector-year "$TARGET_RATE_PER_DETECTOR_YEAR" \
  --output "$OUTPUT_ROOT/morphology_candidate_rate.json"
