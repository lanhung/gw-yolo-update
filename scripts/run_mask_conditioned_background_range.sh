#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  MASK_VALIDATION_RECEIPT
  MASK_TIMING_RECEIPT
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  PARENT_PLAN
  VALIDATION_PURPOSE_AUDIT
  CAPACITY_FORECAST
  EVENT_EXCLUSIONS
  COHERENCE_CONFIG
  CACHE_ROOT
  OUTPUT_ROOT
  SHARD_STOP_EXCLUSIVE
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

PAIRS_PER_SHARD=${PAIRS_PER_SHARD:-4}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.2}
BACKGROUND_SEED=${BACKGROUND_SEED:-20260719}
MODEL_IFOS=${MODEL_IFOS:-"H1 L1 V1"}
Q_VALUES=${Q_VALUES:-"4 8 16"}
TARGET_SAMPLE_RATE=${TARGET_SAMPLE_RATE:-1024}
CONTEXT_DURATION=${CONTEXT_DURATION:-64}
CHIRP_THRESHOLD=${CHIRP_THRESHOLD:-0.3}
MINIMUM_BINS=${MINIMUM_BINS:-1}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-2}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
TARGET_FAR_PER_YEAR=${TARGET_FAR_PER_YEAR:-0.1}
ZERO_COUNT_CONFIDENCE=${ZERO_COUNT_CONFIDENCE:-0.9}
MINIMUM_MASK_EFFICIENCY_GAIN=${MINIMUM_MASK_EFFICIENCY_GAIN:-0.05}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-120}
CHECKPOINT=${CHECKPOINT:-}
CONFIG=${CONFIG:-}
CALIBRATION_CODE_DIR=${CALIBRATION_CODE_DIR:-}
VERIFIED_SOURCE_INVENTORY=${VERIFIED_SOURCE_INVENTORY:-}

if ! [[ "$SHARD_STOP_EXCLUSIVE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$PAIRS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "mask background range settings must be bounded non-negative integers" >&2
  exit 2
fi
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR is not the declared immutable checkout" >&2
  exit 2
fi

if [[ -z "$CHECKPOINT" || -z "$CONFIG" ]]; then
  for variable in FIVE_SEED_SUMMARY UNIFORM_CONFIG FAMILY_BALANCED_CONFIG; do
    if [[ -z "${!variable:-}" ]]; then
      echo "checkpoint/config selection variable is unset: $variable" >&2
      exit 2
    fi
  done
  selection=$(
    "$TASK_PYTHON" - "$FIVE_SEED_SUMMARY" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_five_seed_source_safe_overlap_validation"
    or report.get("passed") is not True
    or report.get("test_data_opened") is not False
):
    raise SystemExit("five-seed summary is not a promoted validation model")
print(report["promoted_arm"])
print(report["selected_checkpoint_path"])
PY
  )
  readarray -t selected <<<"$selection"
  if (( ${#selected[@]} != 2 )); then
    echo "five-seed selector did not return one arm and checkpoint" >&2
    exit 2
  fi
  CHECKPOINT=${selected[1]}
  if [[ "${selected[0]}" == uniform ]]; then
    CONFIG=$UNIFORM_CONFIG
  elif [[ "${selected[0]}" == family_balanced ]]; then
    CONFIG=$FAMILY_BALANCED_CONFIG
  else
    echo "five-seed selector returned an unknown arm" >&2
    exit 2
  fi
fi

for path in \
  "$TASK_PYTHON" \
  "$MASK_VALIDATION_RECEIPT" \
  "$MASK_TIMING_RECEIPT" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$PARENT_PLAN" \
  "$VALIDATION_PURPOSE_AUDIT" \
  "$CAPACITY_FORECAST" \
  "$EVENT_EXCLUSIONS" \
  "$COHERENCE_CONFIG" \
  "$CHECKPOINT" \
  "$CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "required mask background input is absent: $path" >&2
    exit 2
  fi
done

authorization="$OUTPUT_ROOT/publication_background_plan_authorization.json"
(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  "$TASK_PYTHON" -m gwyolo.cli candidate-background-plan-authorize \
    --independent-validation-endpoint "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    --parent-plan "$PARENT_PLAN" \
    --validation-purpose-audit "$VALIDATION_PURPOSE_AUDIT" \
    --capacity-forecast "$CAPACITY_FORECAST" \
    --shard-stop-exclusive "$SHARD_STOP_EXCLUSIVE" \
    --pairs-per-shard "$PAIRS_PER_SHARD" \
    --target-far-per-year "$TARGET_FAR_PER_YEAR" \
    --zero-count-confidence "$ZERO_COUNT_CONFIDENCE" \
    --minimum-safety-factor 1.5 \
    --output "$authorization"
)

required_shards=$(
  "$TASK_PYTHON" - "$PARENT_PLAN" "$PAIRS_PER_SHARD" <<'PY'
import json
import math
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
pairs = list(plan.get("pairs", []))
if (
    plan.get("status") != "development_acquisition_plan"
    or plan.get("locked_evaluation_data") is not False
    or not pairs
    or int(plan.get("selected_pairs", -1)) != len(pairs)
    or len({str(row.get("pair_id", "")) for row in pairs}) != len(pairs)
    or any(not str(row.get("pair_id", "")) for row in pairs)
):
    raise SystemExit("mask background parent is not a complete unlocked plan")
print(math.ceil(len(pairs) / int(sys.argv[2])))
PY
)
if [[ "$required_shards" != "$SHARD_STOP_EXCLUSIVE" ]]; then
  echo "SHARD_STOP_EXCLUSIVE must cover the complete parent plan: $required_shards" >&2
  exit 2
fi

inventory_args=()
if [[ -n "$VERIFIED_SOURCE_INVENTORY" ]]; then
  if [[ ! -s "$VERIFIED_SOURCE_INVENTORY" ]]; then
    echo "verified source inventory is absent" >&2
    exit 2
  fi
  inventory_args+=(--verified-source-inventory "$VERIFIED_SOURCE_INVENTORY")
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
calibration_commit=$(
  "$TASK_PYTHON" - "$MASK_TIMING_RECEIPT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_validation_only_mask_timing_gate"
    or report.get("coherent_background_scale_allowed") is not True
    or report.get("test_rows_read") != 0
):
    raise SystemExit("mask timing gate does not authorize coherent validation scaling")
print(report["pipeline_code_commit"])
PY
)
compatibility_args=()
if [[ "$calibration_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  if [[ -z "$CALIBRATION_CODE_DIR" ]] \
    || [[ "$(git -C "$CALIBRATION_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
      != "$calibration_commit" ]]; then
    echo "cross-commit mask streaming lacks its calibration checkout" >&2
    exit 2
  fi
  compatibility="$OUTPUT_ROOT/candidate_scoring_compatibility.json"
  if [[ ! -s "$compatibility" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT
      "$TASK_PYTHON" -m gwyolo.cli candidate-scoring-compatibility-audit \
        --reference-code-dir "$CALIBRATION_CODE_DIR" \
        --candidate-code-dir "$TASK_CODE_DIR" \
        --reference-commit "$calibration_commit" \
        --candidate-commit "$GWYOLO_CODE_COMMIT" \
        --output "$compatibility"
    )
  fi
  compatibility_args+=(--scoring-compatibility-report "$compatibility")
fi

read -r -a model_ifos <<<"$MODEL_IFOS"
read -r -a q_values <<<"$Q_VALUES"
reports=()
for ((shard = 0; shard < SHARD_STOP_EXCLUSIVE; shard++)); do
  available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < MINIMUM_FREE_KB )); then
    echo "insufficient cache filesystem space before mask shard $shard" >&2
    exit 1
  fi
  while :; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  shard_output="$OUTPUT_ROOT/shard-$shard"
  completed=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    printf '%s raw-mask-background-shard=%s attempt=%s\n' \
      "$(date -u +%FT%TZ)" "$shard" "$attempt"
    if (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT
      "$TASK_PYTHON" -m gwyolo.cli background-raw-mask-stream-shard \
        --parent-plan "$PARENT_PLAN" \
        --event-exclusions "$EVENT_EXCLUSIONS" \
        --mask-validation-receipt "$MASK_VALIDATION_RECEIPT" \
        --mask-timing-receipt "$MASK_TIMING_RECEIPT" \
        "${compatibility_args[@]}" \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --coherence-config "$COHERENCE_CONFIG" \
        --cache-root "$CACHE_ROOT" \
        --output-dir "$shard_output" \
        --shard-index "$shard" \
        --pairs-per-shard "$PAIRS_PER_SHARD" \
        --validation-fraction "$VALIDATION_FRACTION" \
        --seed "$BACKGROUND_SEED" \
        --model-ifos "${model_ifos[@]}" \
        --q-values "${q_values[@]}" \
        --target-sample-rate "$TARGET_SAMPLE_RATE" \
        --context-duration "$CONTEXT_DURATION" \
        --chirp-threshold "$CHIRP_THRESHOLD" \
        --minimum-bins "$MINIMUM_BINS" \
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
  if (( completed != 1 )); then
    echo "raw/mask background shard $shard exhausted bounded retries" >&2
    exit 1
  fi
  report="$shard_output/streamed_background_shard_report.json"
  if [[ ! -s "$report" ]]; then
    echo "raw/mask shard completed without its immutable report" >&2
    exit 1
  fi
  reports+=(--shard-report "$report")
done

merge_dir="$OUTPUT_ROOT/merged"
merge_report="$merge_dir/raw_mask_streamed_background_merge_report.json"
if [[ ! -s "$merge_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli background-raw-mask-stream-merge \
      "${reports[@]}" \
      --parent-plan "$PARENT_PLAN" \
      --output-dir "$merge_dir"
  )
fi

background_manifest="$merge_dir/raw/background_windows.jsonl"
schedule="$OUTPUT_ROOT/val_candidate_block_permutation_schedule.json"
if [[ ! -s "$schedule" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-block-permutation-schedule-freeze \
      --background-manifest "$background_manifest" \
      --output "$schedule" \
      --split val \
      --reference-ifo H1 \
      --shifted-ifo L1 \
      --target-far-per-year "$TARGET_FAR_PER_YEAR" \
      --zero-count-confidence "$ZERO_COUNT_CONFIDENCE"
  )
fi

for arm in raw mask; do
  readarray -t arm_values < <(
    "$TASK_PYTHON" - "$MASK_TIMING_RECEIPT" "$arm" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
arm = sys.argv[2]
method = report["required_method"]
timing = report["timing_reports"][arm]["report"]["methods"][method]
print(report["injection_ranking_reports"][arm]["path"])
print(report["physical_delay_limit_seconds"])
print(timing["empirical_timing_uncertainty_seconds"])
PY
  )
  if (( ${#arm_values[@]} != 3 )); then
    echo "$arm timing receipt did not return ranking/delay/uncertainty" >&2
    exit 2
  fi
  ranking=${arm_values[0]}
  physical=${arm_values[1]}
  uncertainty=${arm_values[2]}
  coincidence=$(
    "$TASK_PYTHON" - "$physical" "$uncertainty" <<'PY'
import sys
print(float(sys.argv[1]) + 2.0 * float(sys.argv[2]))
PY
  )
  candidates="$merge_dir/$arm/val_calibrated_candidates.jsonl"
  block_dir="$OUTPUT_ROOT/$arm/val_candidate_block_background"
  block_report="$block_dir/val_candidate_time_slide_report.json"
  if [[ ! -s "$block_report" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT
      "$TASK_PYTHON" -m gwyolo.cli candidate-block-permutations \
        --candidates "$candidates" \
        --background-manifest "$background_manifest" \
        --schedule "$schedule" \
        --output-dir "$block_dir" \
        --split val \
        --reference-ifo H1 \
        --shifted-ifo L1 \
        --coincidence-window-seconds "$coincidence" \
        --cluster-window-seconds 0.1 \
        --physical-delay-limit-seconds "$physical" \
        --empirical-timing-uncertainty-seconds "$uncertainty"
    )
  fi
  calibration="$OUTPUT_ROOT/$arm/frozen_validation_candidate_search_calibration.json"
  if [[ ! -s "$calibration" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT
      "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibrate \
        --validation-time-slide-report \
          "$block_dir/val_candidate_time_slide_report.json" \
        --validation-injection-ranking-report "$ranking" \
        --target-far-per-year "$TARGET_FAR_PER_YEAR" \
        --output "$calibration" \
        --bootstrap-replicates 10000 \
        --seed 20260720
    )
  fi
done

paired_comparison="$OUTPUT_ROOT/paired_validation_candidate_comparison.json"
if [[ ! -s "$paired_comparison" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-compare \
      --raw-calibration-report \
        "$OUTPUT_ROOT/raw/frozen_validation_candidate_search_calibration.json" \
      --mask-calibration-report \
        "$OUTPUT_ROOT/mask/frozen_validation_candidate_search_calibration.json" \
      --mask-validation-receipt "$MASK_VALIDATION_RECEIPT" \
      --mask-timing-receipt "$MASK_TIMING_RECEIPT" \
      --minimum-absolute-weighted-efficiency-gain \
        "$MINIMUM_MASK_EFFICIENCY_GAIN" \
      --bootstrap-replicates 10000 \
      --seed 20260720 \
      --output "$paired_comparison"
  )
fi

"$TASK_PYTHON" - \
  "$merge_report" \
  "$MASK_VALIDATION_RECEIPT" \
  "$MASK_TIMING_RECEIPT" \
  "$PARENT_PLAN" \
  "$EVENT_EXCLUSIONS" \
  "$CHECKPOINT" \
  "$CONFIG" \
  "$COHERENCE_CONFIG" \
  "$schedule" \
  "$OUTPUT_ROOT/raw/frozen_validation_candidate_search_calibration.json" \
  "$OUTPUT_ROOT/mask/frozen_validation_candidate_search_calibration.json" \
  "$paired_comparison" \
  "$authorization" \
  "$GWYOLO_CODE_COMMIT" \
  "$OUTPUT_ROOT/raw_mask_background_validation_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    merge_path,
    mask_validation_path,
    timing_path,
    parent_path,
    exclusions_path,
    checkpoint_path,
    config_path,
    coherence_path,
    schedule_path,
    raw_path,
    mask_path,
    comparison_path,
    authorization_path,
) = map(pathlib.Path, sys.argv[1:14])
code_commit = sys.argv[14]
output = pathlib.Path(sys.argv[15])
if output.exists():
    raise SystemExit("raw/mask background validation receipts are immutable")
merge = json.loads(merge_path.read_text(encoding="utf-8"))
timing = json.loads(timing_path.read_text(encoding="utf-8"))
calibrations = {
    "raw": json.loads(raw_path.read_text(encoding="utf-8")),
    "mask": json.loads(mask_path.read_text(encoding="utf-8")),
}
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
if (
    merge.get("status")
    != "verified_merged_streamed_raw_mask_candidate_background"
    or merge.get("complete_parent_plan") is not True
    or merge.get("test_rows_read") != 0
    or int(merge.get("split_counts", {}).get("test", -1)) != 0
    or merge.get("code_commit") != code_commit
    or merge.get("common_gate_identity", {}).get("checkpoint_sha256")
    != digest(checkpoint_path)
    or merge.get("common_gate_identity", {}).get("config_sha256")
    != digest(config_path)
    or merge.get("common_gate_identity", {}).get("coherence_config_sha256")
    != digest(coherence_path)
    or timing.get("coherent_background_scale_allowed") is not True
    or comparison.get("status")
    != "validation_only_paired_raw_mask_candidate_calibration_comparison"
    or comparison.get("scientific_claim_allowed") is not False
    or comparison.get("locked_test_allowed") is not False
    or comparison.get("test_rows_read") != 0
    or comparison.get("raw_calibration_report", {}).get("sha256")
    != digest(raw_path)
    or comparison.get("mask_calibration_report", {}).get("sha256")
    != digest(mask_path)
    or authorization.get("status")
    != "authorized_validation_candidate_continuous_background_plan"
    or authorization.get("passed") is not True
    or authorization.get("scientific_claim_allowed") is not False
    or authorization.get("test_rows_read") != 0
    or authorization.get("parent_plan", {}).get("sha256") != digest(parent_path)
    or any(
        report.get("status") != "frozen_validation_candidate_search_calibration"
        or report.get("publication_calibration_eligible") is not True
        or report.get("slide_schedule_audit", {}).get("passed") is not True
        for report in calibrations.values()
    )
):
    raise SystemExit("paired raw/mask background did not reach both validation FAR gates")
receipt = {
    "status": "completed_validation_only_raw_mask_continuous_background",
    "scientific_claim_allowed": False,
    "locked_test_allowed": False,
    "test_rows_read": 0,
    "validation_calibration_frozen": True,
    "continuous_background_search_claim_allowed": False,
    "locked_test_open_allowed": False,
    "locked_test_prerequisites_satisfied": False,
    "mask_locked_test_arm_eligible": comparison.get(
        "mask_locked_test_arm_eligible"
    ) is True,
    "code_commit": code_commit,
    "merge_report": {"path": str(merge_path), "sha256": digest(merge_path)},
    "mask_validation_receipt": {
        "path": str(mask_validation_path),
        "sha256": digest(mask_validation_path),
    },
    "mask_timing_receipt": {
        "path": str(timing_path),
        "sha256": digest(timing_path),
    },
    "inputs": {
        "background_plan_authorization": {
            "path": str(authorization_path),
            "sha256": digest(authorization_path),
        },
        "parent_plan": {"path": str(parent_path), "sha256": digest(parent_path)},
        "event_exclusions": {
            "path": str(exclusions_path),
            "sha256": digest(exclusions_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": digest(checkpoint_path),
        },
        "config": {"path": str(config_path), "sha256": digest(config_path)},
        "coherence_config": {
            "path": str(coherence_path),
            "sha256": digest(coherence_path),
        },
    },
    "block_schedule": {
        "path": str(schedule_path),
        "sha256": digest(schedule_path),
    },
    "calibrations": {
        "raw": {"path": str(raw_path), "sha256": digest(raw_path)},
        "mask": {"path": str(mask_path), "sha256": digest(mask_path)},
    },
    "paired_validation_comparison": {
        "path": str(comparison_path),
        "sha256": digest(comparison_path),
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY
