#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_MANIFEST
  CONFIG
  CACHE_ROOT
  LEGACY_OUTPUT_PREFIX
  OUTPUT_ROOT
  SOURCE_SHARD_COUNT
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
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-30}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
LEGACY_OUTPUT_SUFFIX=${LEGACY_OUTPUT_SUFFIX:-}
WAIT_PID=${WAIT_PID:-}

for value in "$SOURCE_SHARD_COUNT" "$MAX_ATTEMPTS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "SOURCE_SHARD_COUNT and MAX_ATTEMPTS must be positive integers" >&2
    exit 2
  fi
done
for value in "$DOWNLOAD_WORKERS" "$CHUNK_SAMPLES" "$MINIMUM_FREE_KB"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "download, chunk and free-space settings must be positive integers" >&2
    exit 2
  fi
done
if ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "RETRY_DELAY_SECONDS must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "$WAIT_PID" ]] && ! [[ "$WAIT_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAIT_PID must be a positive integer" >&2
  exit 2
fi
for input in "$TASK_PYTHON" "$SOURCE_MANIFEST" "$CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/.git" ]]; then
  echo "TASK_CODE_DIR is not an exact git checkout" >&2
  exit 2
fi
actual_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD)
if [[ "$actual_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" == "$LEGACY_OUTPUT_PREFIX"* ]]; then
  echo "fallback OUTPUT_ROOT must be separate from legacy shard outputs" >&2
  exit 2
fi

if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT

reports=()
for ((shard = 0; shard < SOURCE_SHARD_COUNT; shard++)); do
  legacy_dir="${LEGACY_OUTPUT_PREFIX}${shard}${LEGACY_OUTPUT_SUFFIX}"
  legacy_report="$legacy_dir/gravityspy_network_numeric_report.json"
  legacy_partial="$legacy_dir/materialization_partial.json"
  fallback_dir="$OUTPUT_ROOT/shard-$shard"
  fallback_report="$fallback_dir/gravityspy_network_numeric_report.json"
  selected_report=

  if [[ -s "$legacy_report" ]]; then
    selected_report=$legacy_report
  else
    available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
    if (( available_kb < MINIMUM_FREE_KB )); then
      echo "insufficient cache space before validation shard $shard" >&2
      exit 1
    fi
    completed=0
    inventory_args=()
    if [[ -s "$legacy_partial" ]]; then
      inventory_args=(--verified-source-inventory "$legacy_partial")
    fi
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
      printf '%s shard=%s attempt=%s\n' \
        "$(date -u +%FT%TZ)" "$shard" "$attempt"
      if "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-strain-materialize \
        --manifest "$SOURCE_MANIFEST" \
        --config "$CONFIG" \
        --cache-dir "$CACHE_ROOT" \
        --output-dir "$fallback_dir" \
        --output-duration "$OUTPUT_DURATION" \
        --download-workers "$DOWNLOAD_WORKERS" \
        --chunk-samples "$CHUNK_SAMPLES" \
        "${inventory_args[@]}" \
        --shard "$shard"; then
        completed=1
        break
      fi
      if (( attempt < MAX_ATTEMPTS )); then
        sleep "$RETRY_DELAY_SECONDS"
      fi
    done
    if (( completed != 1 )) || [[ ! -s "$fallback_report" ]]; then
      echo "validation shard $shard exhausted bounded fallback retries" >&2
      exit 1
    fi
    eviction_report="$fallback_dir/source_eviction_report.json"
    if [[ ! -s "$eviction_report" ]]; then
      "$TASK_PYTHON" -m gwyolo.cli gravityspy-strain-evict \
        --materialization-report "$fallback_report" \
        --cache-dir "$CACHE_ROOT" \
        --output "$eviction_report"
    fi
    selected_report=$fallback_report
  fi
  reports+=(--report "$selected_report")
done

merged_dir="$OUTPUT_ROOT/merged"
merged_report="$merged_dir/gravityspy_network_numeric_merge_report.json"
if [[ ! -s "$merged_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-numeric-merge \
    "${reports[@]}" \
    --output-dir "$merged_dir" \
    --split val
fi
printf 'validation fallback complete: %s\n' "$merged_report"
