#!/usr/bin/env bash
set -euo pipefail

# Fill only the undersized detector-set validation strata. Every incremental
# shard uses the frozen hash-threshold assignment, contains no test split,
# is reduced to one numeric context per GPS block, and evicts its recoverable
# public HDF sources before the next shard is downloaded.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BASE_BACKGROUND_MANIFEST
  BASE_BACKGROUND_REPORT
  H1V1_ACQUISITION_PLAN
  L1V1_ACQUISITION_PLAN
  EVENT_EXCLUSIONS
  CACHE_ROOT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector-validation streaming variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$BASE_BACKGROUND_MANIFEST" \
  "$BASE_BACKGROUND_REPORT" \
  "$H1V1_ACQUISITION_PLAN" \
  "$L1V1_ACQUISITION_PLAN" \
  "$EVENT_EXCLUSIONS"; do
  if [[ ! -s "$path" ]]; then
    echo "required detector-validation streaming input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector-validation streaming requires its exact checkout" >&2
  exit 3
fi

pairs_per_shard=${PAIRS_PER_SHARD:-2}
minimum_per_subset=${MINIMUM_PER_DETECTOR_SUBSET:-25}
validation_fraction=${VALIDATION_FRACTION:-0.8}
background_seed=${DETECTOR_VALIDATION_BACKGROUND_SEED:-20260723}
download_workers=${DOWNLOAD_WORKERS:-8}
maximum_attempts=${MAXIMUM_DOWNLOAD_ATTEMPTS:-20}
retry_delay_seconds=${RETRY_DELAY_SECONDS:-60}
minimum_free_bytes=${MINIMUM_FREE_BYTES:-2147483648}
target_sample_rate=${TARGET_SAMPLE_RATE:-1024}
context_duration=${CONTEXT_DURATION_SECONDS:-64}
injections_per_subset=${INJECTIONS_PER_DETECTOR_SUBSET:-100}
injection_seed=${DETECTOR_VALIDATION_INJECTION_SEED:-20260723}
if (( pairs_per_shard < 1 || minimum_per_subset < 1 || maximum_attempts < 1 )); then
  echo "detector-validation streaming integer policy is invalid" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT

base_count() {
  local subset=$1
  "$TASK_PYTHON" - "$BASE_BACKGROUND_REPORT" "$subset" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status")
    != "exported_source_safe_detector_validation_background_bank"
    or report.get("candidate_scores_inspected") is not False
    or int(report.get("test_rows_read", -1)) != 0
):
    raise SystemExit("base detector-validation report failed replay")
print(int(report.get("detector_subset_counts", {}).get(sys.argv[2], 0)))
PY
}

receipt_rows() {
  local subset=$1
  shift
  "$TASK_PYTHON" - "$subset" "$@" <<'PY'
import json
import pathlib
import sys

subset, *paths = sys.argv[1:]
total = 0
for value in paths:
    report = json.loads(pathlib.Path(value).read_text(encoding="utf-8"))
    if (
        report.get("status") != "verified_streamed_detector_validation_shard"
        or report.get("passed") is not True
        or report.get("detector_subset") != subset
        or report.get("candidate_scores_inspected") is not False
        or int(report.get("test_rows_read", -1)) != 0
    ):
        raise SystemExit(f"streamed receipt failed replay: {value}")
    total += int(report["unique_validation_gps_blocks"])
print(total)
PY
}

acquire_subset() {
  local subset=$1
  local parent_plan=$2
  local subset_root=$3
  local parent_pairs
  local shard_count
  local existing
  local added
  mkdir -p "$subset_root"
  read -r parent_pairs shard_count < <(
    "$TASK_PYTHON" - "$parent_plan" "$pairs_per_shard" "$subset" <<'PY'
import json
import math
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pairs_per_shard = int(sys.argv[2])
expected = sys.argv[3].split("+")
if (
    plan.get("status") != "development_acquisition_plan"
    or plan.get("selection_rule")
    != "source_file_and_gps_disjoint_stratified_v1"
    or plan.get("candidate_scores_inspected") is not False
    or plan.get("test_data_opened") is not False
    or plan.get("locked_evaluation_data") is not False
    or sorted(plan.get("detectors", [])) != expected
):
    raise SystemExit("detector-validation parent plan failed replay")
pairs = int(plan["selected_pairs"])
print(pairs, math.ceil(pairs / pairs_per_shard))
PY
  )
  if (( parent_pairs < 1 || shard_count < 1 )); then
    echo "detector-validation parent plan has no usable shards: $subset" >&2
    exit 3
  fi

  existing=$(base_count "$subset")
  for ((shard = 0; shard < shard_count; shard++)); do
    mapfile -t current_receipts < <(
      find "$subset_root" -mindepth 2 -maxdepth 2 \
        -type f -name detector_validation_shard_receipt.json | sort
    )
    added=$(receipt_rows "$subset" "${current_receipts[@]}")
    if (( existing + added >= minimum_per_subset )); then
      break
    fi
    available=$(df --output=avail -B1 "$CACHE_ROOT" | tail -1)
    if (( available < minimum_free_bytes )); then
      echo "detector-validation cache is below its free-space guard: $available" >&2
      exit 4
    fi

    shard_root="$subset_root/shard-$shard"
    shard_plan="$shard_root/acquisition_plan.json"
    batch_root="$shard_root/download"
    batch_report="$batch_root/batch_download_report.json"
    background_root="$shard_root/background"
    background_report="$background_root/background_plan_report.json"
    background_manifest="$background_root/background_windows.jsonl"
    bank_root="$shard_root/bank"
    bank_report="$bank_root/background_bank_report.json"
    eviction_report="$shard_root/source_eviction_report.json"
    receipt="$shard_root/detector_validation_shard_receipt.json"
    mkdir -p "$shard_root"
    if [[ ! -s "$shard_plan" ]]; then
      "$TASK_PYTHON" -m gwyolo.cli gwosc-plan-shard \
        --plan "$parent_plan" \
        --shard-index "$shard" \
        --pairs-per-shard "$pairs_per_shard" \
        --output "$shard_plan"
    fi
    if [[ ! -s "$batch_report" ]]; then
      completed=0
      for ((attempt = 1; attempt <= maximum_attempts; attempt++)); do
        if "$TASK_PYTHON" -m gwyolo.cli gwosc-batch-download \
          --plan "$shard_plan" \
          --cache-dir "$CACHE_ROOT" \
          --output-dir "$batch_root" \
          --download-workers "$download_workers"; then
          completed=1
          break
        fi
        if (( attempt < maximum_attempts )); then
          sleep "$retry_delay_seconds"
        fi
      done
      if (( completed != 1 )) || [[ ! -s "$batch_report" ]]; then
        echo "detector-validation shard exhausted download retries: $subset/$shard" >&2
        exit 1
      fi
    fi
    if [[ ! -s "$background_report" ]]; then
      "$TASK_PYTHON" -m gwyolo.cli background-batch-plan \
        --batch-report "$batch_report" \
        --event-exclusions "$EVENT_EXCLUSIONS" \
        --output-dir "$background_root" \
        --validation-fraction "$validation_fraction" \
        --test-fraction 0 \
        --seed "$background_seed" \
        --split-strategy hash_threshold_v1
    fi
    if [[ ! -s "$bank_report" ]]; then
      "$TASK_PYTHON" -m gwyolo.cli background-bank-materialize \
        --background-manifest "$background_manifest" \
        --output-dir "$bank_root" \
        --target-sample-rate "$target_sample_rate" \
        --context-duration "$context_duration" \
        --split val \
        --maximum-windows-per-gps-block 1
    fi
    if [[ ! -s "$eviction_report" ]]; then
      "$TASK_PYTHON" -m gwyolo.cli background-bank-evict-sources \
        --background-bank-report "$bank_report" \
        --cache-root "$CACHE_ROOT" \
        --output "$eviction_report"
    fi
    if [[ ! -s "$receipt" ]]; then
      "$TASK_PYTHON" -m gwyolo.cli detector-validation-shard-seal \
        --parent-plan "$parent_plan" \
        --shard-plan "$shard_plan" \
        --batch-report "$batch_report" \
        --background-report "$background_report" \
        --background-bank-report "$bank_report" \
        --eviction-report "$eviction_report" \
        --output "$receipt"
    fi
  done

  mapfile -t final_receipts < <(
    find "$subset_root" -mindepth 2 -maxdepth 2 \
      -type f -name detector_validation_shard_receipt.json | sort
  )
  added=$(receipt_rows "$subset" "${final_receipts[@]}")
  if (( existing + added < minimum_per_subset )); then
    echo "detector-validation plan exhausted below subset floor: $subset" >&2
    exit 1
  fi
}

acquire_subset "H1+V1" "$H1V1_ACQUISITION_PLAN" "$OUTPUT_ROOT/H1-V1"
acquire_subset "L1+V1" "$L1V1_ACQUISITION_PLAN" "$OUTPUT_ROOT/L1-V1"

mapfile -t receipts < <(
  find "$OUTPUT_ROOT/H1-V1" "$OUTPUT_ROOT/L1-V1" -mindepth 2 -maxdepth 2 \
    -type f -name detector_validation_shard_receipt.json | sort
)
if (( ${#receipts[@]} < 2 )); then
  echo "detector-validation streaming produced too few sealed shards" >&2
  exit 1
fi
receipt_args=()
for receipt in "${receipts[@]}"; do
  receipt_args+=(--shard-receipt "$receipt")
done

merged_root="$OUTPUT_ROOT/merged"
"$TASK_PYTHON" -m gwyolo.cli detector-validation-background-merge \
  --base-manifest "$BASE_BACKGROUND_MANIFEST" \
  --base-report "$BASE_BACKGROUND_REPORT" \
  "${receipt_args[@]}" \
  --output-dir "$merged_root" \
  --minimum-per-detector-subset "$minimum_per_subset" \
  --require-ready

injection_root="$OUTPUT_ROOT/injections"
"$TASK_PYTHON" -m gwyolo.cli detector-validation-injection-plan \
  --background-manifest "$merged_root/background_windows.jsonl" \
  --background-report "$merged_root/detector_validation_background_report.json" \
  --output-dir "$injection_root" \
  --injections-per-detector-subset "$injections_per_subset" \
  --seed "$injection_seed"

"$TASK_PYTHON" - "$merged_root/detector_validation_background_report.json" \
  "$injection_root/detector_stratified_injection_plan.json" \
  "$GWYOLO_CODE_COMMIT" "$OUTPUT_ROOT/detector_validation_streaming_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


background_path, injection_path, commit, target_value = sys.argv[1:]
background = json.loads(pathlib.Path(background_path).read_text(encoding="utf-8"))
injections = json.loads(pathlib.Path(injection_path).read_text(encoding="utf-8"))
if (
    background.get("status")
    != "exported_source_safe_detector_validation_background_bank"
    or background.get("passed") is not True
    or background.get("publication_calibration_eligible") is not True
    or background.get("candidate_scores_inspected") is not False
    or int(background.get("test_rows_read", -1)) != 0
    or injections.get("status")
    != "frozen_detector_stratified_validation_injection_plan"
    or injections.get("passed") is not True
    or injections.get("background_report_sha256") != digest(background_path)
    or injections.get("candidate_scores_inspected") is not False
    or int(injections.get("test_rows_read", -1)) != 0
):
    raise SystemExit("detector-validation streaming chain failed final replay")
result = {
    "status": "verified_detector_stratified_validation_data_chain",
    "passed": True,
    "scientific_claim_allowed": False,
    "candidate_scores_inspected": False,
    "test_rows_read": 0,
    "code_commit": commit,
    "background_report_path": str(pathlib.Path(background_path).resolve()),
    "background_report_sha256": digest(background_path),
    "injection_plan_path": str(pathlib.Path(injection_path).resolve()),
    "injection_plan_sha256": digest(injection_path),
    "detector_subset_counts": background["detector_subset_counts"],
    "injection_counts": injections["detector_subset_counts"],
}
target = pathlib.Path(target_value)
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
