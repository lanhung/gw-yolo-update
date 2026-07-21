#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TRAIN_SOURCE_MANIFEST
  VAL_SOURCE_MANIFEST
  TRAIN_REPORT_PREFIX
  VAL_REPORT_PREFIX
  SOURCE_REPORT_SUFFIX
  SOURCE_SHARD_COUNT
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

if ! [[ "$SOURCE_SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SOURCE_SHARD_COUNT must be positive" >&2
  exit 2
fi
for input in "$TASK_PYTHON" "$TRAIN_SOURCE_MANIFEST" "$VAL_SOURCE_MANIFEST" "$CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"

for split in train val; do
  if [[ "$split" == train ]]; then
    source_manifest=$TRAIN_SOURCE_MANIFEST
    report_prefix=$TRAIN_REPORT_PREFIX
  else
    source_manifest=$VAL_SOURCE_MANIFEST
    report_prefix=$VAL_REPORT_PREFIX
  fi
  reports=()
  plan_args=()
  for ((shard = 0; shard < SOURCE_SHARD_COUNT; shard++)); do
    report="${report_prefix}${shard}${SOURCE_REPORT_SUFFIX}"
    if [[ ! -s "$report" ]]; then
      echo "completed source shard report is absent: $report" >&2
      exit 1
    fi
    reports+=(--report "$report")
    plan_args+=(--materialization-report "$report")
  done
  plan_dir="$OUTPUT_ROOT/$split-plan"
  plan_report="$plan_dir/gravityspy_network_recovery_plan_report.json"
  if [[ ! -s "$plan_report" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-recovery-plan \
      --source-manifest "$source_manifest" \
      "${plan_args[@]}" \
      --output-dir "$plan_dir"
  fi
  recovery_manifest="$plan_dir/gravityspy_network_recovery_plan.jsonl"
  recovery_dir="$OUTPUT_ROOT/$split-materialized"
  available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < MINIMUM_FREE_KB )); then
    echo "insufficient cache space before $split recovery" >&2
    exit 1
  fi
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-strain-materialize \
    --manifest "$recovery_manifest" \
    --config "$CONFIG" \
    --cache-dir "$CACHE_ROOT/$split" \
    --output-dir "$recovery_dir" \
    --output-duration "$OUTPUT_DURATION" \
    --download-workers "$DOWNLOAD_WORKERS" \
    --chunk-samples "$CHUNK_SAMPLES"
  recovery_report="$recovery_dir/gravityspy_network_numeric_report.json"
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-strain-evict \
    --materialization-report "$recovery_report" \
    --cache-dir "$CACHE_ROOT/$split" \
    --output "$recovery_dir/source_eviction_report.json"
  merged_dir="$OUTPUT_ROOT/$split-merged"
  if [[ ! -s "$merged_dir/gravityspy_network_numeric_merge_report.json" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-numeric-merge \
      "${reports[@]}" \
      --report "$recovery_report" \
      --output-dir "$merged_dir" \
      --split "$split"
  fi
done
