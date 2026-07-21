#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TRAIN_SHARD_MANIFEST
  VAL_SHARD_MANIFEST
  TRAIN_SHARD_COUNT
  VAL_SHARD_COUNT
  TRAIN_EXISTING_MERGE_REPORT
  VAL_EXISTING_MERGE_REPORT
  CONFIG
  CACHE_ROOT
  OUTPUT_ROOT
  GWYOLO_CODE_COMMIT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

OUTPUT_DURATION=${OUTPUT_DURATION:-8}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
CHUNK_SAMPLES=${CHUNK_SAMPLES:-1048576}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}

for count in "$TRAIN_SHARD_COUNT" "$VAL_SHARD_COUNT"; do
  if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
    echo "shard counts must be positive integers" >&2
    exit 2
  fi
done
if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi
for input in \
  "$TASK_PYTHON" \
  "$TRAIN_SHARD_MANIFEST" \
  "$VAL_SHARD_MANIFEST" \
  "$TRAIN_EXISTING_MERGE_REPORT" \
  "$VAL_EXISTING_MERGE_REPORT" \
  "$CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"

materialize_with_retry() {
  local label=$1
  shift
  local attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    printf '%s %s attempt=%s\n' "$(date -u +%FT%TZ)" "$label" "$attempt"
    if "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-strain-materialize "$@"; then
      return 0
    fi
    if (( attempt < MAX_ATTEMPTS )); then
      sleep "$RETRY_DELAY_SECONDS"
    fi
  done
  echo "$label exhausted bounded materialization retries" >&2
  return 1
}

for split in train val; do
  if [[ "$split" == train ]]; then
    shard_manifest=$TRAIN_SHARD_MANIFEST
    shard_count=$TRAIN_SHARD_COUNT
    existing_report=$TRAIN_EXISTING_MERGE_REPORT
  else
    shard_manifest=$VAL_SHARD_MANIFEST
    shard_count=$VAL_SHARD_COUNT
    existing_report=$VAL_EXISTING_MERGE_REPORT
  fi

  reports=()
  for ((shard = 0; shard < shard_count; shard++)); do
    available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
    if (( available_kb < MINIMUM_FREE_KB )); then
      echo "insufficient cache space before $split shard $shard" >&2
      exit 1
    fi
    shard_output="$OUTPUT_ROOT/$split-shard-$shard"
    materialize_with_retry "$split-shard-$shard" \
      --manifest "$shard_manifest" \
      --shard "$shard" \
      --config "$CONFIG" \
      --cache-dir "$CACHE_ROOT/$split" \
      --output-dir "$shard_output" \
      --output-duration "$OUTPUT_DURATION" \
      --download-workers "$DOWNLOAD_WORKERS" \
      --chunk-samples "$CHUNK_SAMPLES"
    report="$shard_output/gravityspy_network_numeric_report.json"
    "$TASK_PYTHON" -m gwyolo.cli gravityspy-strain-evict \
      --materialization-report "$report" \
      --cache-dir "$CACHE_ROOT/$split" \
      --output "$shard_output/source_eviction_report.json"
    reports+=(--report "$report")
  done

  independent_dir="$OUTPUT_ROOT/$split-independent-merged"
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-numeric-merge \
    "${reports[@]}" \
    --output-dir "$independent_dir" \
    --split "$split"

  "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-numeric-merge \
    --report "$existing_report" \
    --report "$independent_dir/gravityspy_network_numeric_merge_report.json" \
    --output-dir "$OUTPUT_ROOT/$split-expanded-merged" \
    --split "$split"
done

"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-corpus-resplit \
  --report "$OUTPUT_ROOT/train-expanded-merged/gravityspy_network_numeric_merge_report.json" \
  --report "$OUTPUT_ROOT/val-expanded-merged/gravityspy_network_numeric_merge_report.json" \
  --output-dir "$OUTPUT_ROOT/source-component-safe-resplit" \
  --validation-fraction 0.2 \
  --seed 20260720

"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-corpus-audit \
  --train-report "$OUTPUT_ROOT/source-component-safe-resplit/gravityspy_network_numeric_train_report.json" \
  --validation-report "$OUTPUT_ROOT/source-component-safe-resplit/gravityspy_network_numeric_val_report.json" \
  --output "$OUTPUT_ROOT/gravityspy_network_expanded_corpus_audit.json"
