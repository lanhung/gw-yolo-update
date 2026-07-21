#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PARENT_PLAN
  EVENT_EXCLUSIONS
  CACHE_ROOT
  OUTPUT_ROOT
  SHARD_STOP_EXCLUSIVE
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
TEST_FRACTION=${TEST_FRACTION:-0}
BACKGROUND_SEED=${BACKGROUND_SEED:-20260719}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-2}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-120}
VERIFIED_SOURCE_INVENTORY=${VERIFIED_SOURCE_INVENTORY:-}

if [[ "$SHARD_START" != "0" ]]; then
  echo "a merged acquisition range must begin at shard zero" >&2
  exit 2
fi
if ! [[ "$SHARD_STOP_EXCLUSIVE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$PAIRS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "shard bounds, retry count, and pairs per shard must be valid integers" >&2
  exit 2
fi
if [[ "$TEST_FRACTION" != "0" && "$TEST_FRACTION" != "0.0" ]]; then
  echo "development calibration acquisition must keep test_fraction=0" >&2
  exit 2
fi
for input in "$TASK_PYTHON" "$PARENT_PLAN" "$EVENT_EXCLUSIONS"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "code directory is invalid: $TASK_CODE_DIR" >&2
  exit 2
fi

inventory_args=()
if [[ -n "$VERIFIED_SOURCE_INVENTORY" ]]; then
  if [[ ! -f "$VERIFIED_SOURCE_INVENTORY" ]]; then
    echo "verified source inventory is absent: $VERIFIED_SOURCE_INVENTORY" >&2
    exit 2
  fi
  inventory_args+=(--verified-source-inventory "$VERIFIED_SOURCE_INVENTORY")
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
batch_reports=()
for ((shard = SHARD_START; shard < SHARD_STOP_EXCLUSIVE; shard++)); do
  available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < MINIMUM_FREE_KB )); then
    echo "insufficient cache filesystem space before acquisition shard $shard" >&2
    exit 1
  fi
  shard_output="$OUTPUT_ROOT/shard-$shard"
  shard_plan="$shard_output/acquisition_plan_shard.json"
  batch_report="$shard_output/download/batch_download_report.json"
  mkdir -p "$shard_output"
  if [[ ! -s "$shard_plan" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli gwosc-plan-shard \
        --plan "$PARENT_PLAN" \
        --shard-index "$shard" \
        --pairs-per-shard "$PAIRS_PER_SHARD" \
        --output "$shard_plan"
    )
  fi
  "$TASK_PYTHON" - "$shard_plan" "$PARENT_PLAN" "$shard" "$PAIRS_PER_SHARD" <<'PY'
import hashlib
import json
import pathlib
import sys

shard_path, parent_path, shard_index, pairs_per_shard = sys.argv[1:]
shard = json.loads(pathlib.Path(shard_path).read_text(encoding="utf-8"))
parent_hash = hashlib.sha256(pathlib.Path(parent_path).read_bytes()).hexdigest()
if (
    shard.get("status") != "development_acquisition_plan"
    or shard.get("locked_evaluation_data") is not False
    or shard.get("parent_plan_sha256") != parent_hash
    or int(shard.get("shard_index", -1)) != int(shard_index)
    or int(shard.get("pairs_per_shard", -1)) != int(pairs_per_shard)
    or int(shard.get("selected_pairs", 0)) <= 0
):
    raise SystemExit("acquisition shard plan has another identity or is incomplete")
PY

  completed=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    printf '%s acquisition-shard=%s attempt=%s\n' \
      "$(date -u +%FT%TZ)" "$shard" "$attempt"
    if [[ -s "$batch_report" ]] || (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli gwosc-batch-download \
        --plan "$shard_plan" \
        --cache-dir "$CACHE_ROOT" \
        --output-dir "$shard_output/download" \
        --download-workers "$DOWNLOAD_WORKERS" \
        "${inventory_args[@]}"
    ); then
      completed=1
      break
    fi
    if (( attempt < MAX_ATTEMPTS )); then
      sleep "$RETRY_DELAY_SECONDS"
    fi
  done
  if (( completed != 1 )) || [[ ! -s "$batch_report" ]]; then
    echo "acquisition shard $shard exhausted retries without a batch report" >&2
    exit 1
  fi
  "$TASK_PYTHON" - "$batch_report" "$shard_plan" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, plan_path = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
plan_hash = hashlib.sha256(pathlib.Path(plan_path).read_bytes()).hexdigest()
expected_files = int(plan["selected_pairs"]) * len(plan["detectors"])
keys = {(str(row.get("pair_id")), str(row.get("detector"))) for row in report.get("files", [])}
if (
    report.get("status") != "verified_development_strain_batch"
    or not report.get("passed")
    or report.get("plan_sha256") != plan_hash
    or int(report.get("selected_pairs", -1)) != int(plan["selected_pairs"])
    or int(report.get("verified_files", -1)) != expected_files
    or len(keys) != expected_files
    or any(not row.get("verification", {}).get("passed") for row in report.get("files", []))
):
    raise SystemExit("batch download report is incomplete or belongs to another shard")
PY
  batch_reports+=(--batch-report "$batch_report")
done

background_dir="$OUTPUT_ROOT/merged-background"
background_report="$background_dir/background_plan_report.json"
if [[ ! -s "$background_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli background-batch-plan \
      "${batch_reports[@]}" \
      --event-exclusions "$EVENT_EXCLUSIONS" \
      --output-dir "$background_dir" \
      --validation-fraction "$VALIDATION_FRACTION" \
      --test-fraction "$TEST_FRACTION" \
      --seed "$BACKGROUND_SEED" \
      --split-strategy hash_threshold_v1
  )
fi

report_paths=()
for ((shard = SHARD_START; shard < SHARD_STOP_EXCLUSIVE; shard++)); do
  report_paths+=("$OUTPUT_ROOT/shard-$shard/download/batch_download_report.json")
done
"$TASK_PYTHON" - "$background_report" "$PARENT_PLAN" "$SHARD_STOP_EXCLUSIVE" \
  "$PAIRS_PER_SHARD" "$TEST_FRACTION" "${report_paths[@]}" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, parent_path, shard_stop, pairs_per_shard, test_fraction, *batch_paths = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
parent = json.loads(pathlib.Path(parent_path).read_text(encoding="utf-8"))
expected_pairs = min(int(shard_stop) * int(pairs_per_shard), int(parent["selected_pairs"]))
expected_hashes = [hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in batch_paths]
manifest = pathlib.Path(report.get("manifest_path", ""))
overlaps = report.get("cross_split_block_overlaps", {})
if (
    report.get("status") != "verified_multi_segment_development_background"
    or not report.get("passed")
    or report.get("split_strategy") != "hash_threshold_v1"
    or int(report.get("source_pairs", -1)) != expected_pairs
    or report.get("source_batch_report_sha256s") != expected_hashes
    or float(test_fraction) != 0
    or int(report.get("splits", {}).get("test", {}).get("windows", -1)) != 0
    or any(overlaps.values())
    or not manifest.is_file()
    or hashlib.sha256(manifest.read_bytes()).hexdigest() != report.get("manifest_sha256")
):
    raise SystemExit("merged background report is incomplete, contaminated, or hash-inconsistent")
PY

printf '%s merged-background-report=%s\n' "$(date -u +%FT%TZ)" "$background_report"
