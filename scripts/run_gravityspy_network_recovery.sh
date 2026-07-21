#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TRAIN_SOURCE_MANIFEST
  VAL_SOURCE_MANIFEST
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
REPORT_MODE=${REPORT_MODE:-prefix}

if [[ "$REPORT_MODE" == prefix ]]; then
  for variable in TRAIN_REPORT_PREFIX VAL_REPORT_PREFIX SOURCE_REPORT_SUFFIX SOURCE_SHARD_COUNT; do
    if [[ -z "${!variable:-}" ]]; then
      echo "required prefix-mode variable is unset: $variable" >&2
      exit 2
    fi
  done
  if ! [[ "$SOURCE_SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SOURCE_SHARD_COUNT must be positive" >&2
    exit 2
  fi
elif [[ "$REPORT_MODE" == merged ]]; then
  for variable in TRAIN_MERGE_REPORT VAL_MERGE_REPORT; do
    if [[ -z "${!variable:-}" ]]; then
      echo "required merged-mode variable is unset: $variable" >&2
      exit 2
    fi
  done
else
  echo "REPORT_MODE must be prefix or merged" >&2
  exit 2
fi
inputs=("$TASK_PYTHON" "$TRAIN_SOURCE_MANIFEST" "$VAL_SOURCE_MANIFEST" "$CONFIG")
if [[ "$REPORT_MODE" == merged ]]; then
  inputs+=("$TRAIN_MERGE_REPORT" "$VAL_MERGE_REPORT")
fi
for input in "${inputs[@]}"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"

for split in train val; do
  if [[ "$split" == train ]]; then
    source_manifest=$TRAIN_SOURCE_MANIFEST
    if [[ "$REPORT_MODE" == prefix ]]; then
      report_prefix=$TRAIN_REPORT_PREFIX
    else
      merge_report=$TRAIN_MERGE_REPORT
    fi
  else
    source_manifest=$VAL_SOURCE_MANIFEST
    if [[ "$REPORT_MODE" == prefix ]]; then
      report_prefix=$VAL_REPORT_PREFIX
    else
      merge_report=$VAL_MERGE_REPORT
    fi
  fi
  reports=()
  plan_args=()
  report_paths=()
  if [[ "$REPORT_MODE" == prefix ]]; then
    for ((shard = 0; shard < SOURCE_SHARD_COUNT; shard++)); do
      report_paths+=("${report_prefix}${shard}${SOURCE_REPORT_SUFFIX}")
    done
  else
    if ! report_output=$("$TASK_PYTHON" - "$merge_report" "$split" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


path = pathlib.Path(sys.argv[1])
split = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
sources = report.get("source_reports", [])
if (
    report.get("status") != "verified_merged_gravityspy_aligned_network_numeric_split"
    or report.get("split") != split
    or not sources
):
    raise SystemExit("merged network report is incomplete or has another split")
for item in sources:
    source = pathlib.Path(item["path"])
    if not source.is_file() or digest(source) != item.get("sha256"):
        raise SystemExit(f"merged source report hash mismatch: {source}")
    print(source)
PY
    ); then
      echo "failed to resolve completed shard reports from $split merge" >&2
      exit 1
    fi
    readarray -t report_paths <<<"$report_output"
  fi
  if (( ${#report_paths[@]} == 0 )); then
    echo "no completed source shard reports were resolved for $split" >&2
    exit 1
  fi
  for report in "${report_paths[@]}"; do
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
