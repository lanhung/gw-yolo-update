#!/usr/bin/env bash
set -euo pipefail

# Extend, rather than refit, the frozen 80-pair AMPLFI background bank. The
# capacity policy and hash split seed remain unchanged. Acquisition stops at
# the first source-disjoint shard whose merged metadata clears every frozen
# train/validation duration and GPS-block requirement.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BASE_ACQUISITION_PLAN
  BASE_STREAM_MERGE_REPORT
  EVENT_EXCLUSIONS
  CAPACITY_POLICY
  CACHE_ROOT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required AMPLFI capacity-extension variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$BASE_ACQUISITION_PLAN" \
  "$BASE_STREAM_MERGE_REPORT" \
  "$EVENT_EXCLUSIONS" \
  "$CAPACITY_POLICY"; do
  if [[ ! -s "$path" ]]; then
    echo "required AMPLFI capacity-extension input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "AMPLFI capacity extension requires its exact checkout" >&2
  exit 3
fi

target_pairs=${EXTENSION_TARGET_PAIRS:-8}
pairs_per_shard=${PAIRS_PER_SHARD:-2}
extension_seed=${EXTENSION_PLAN_SEED:-20260731}
background_seed=${BACKGROUND_SEED:-20260727}
validation_fraction=${VALIDATION_FRACTION:-0.2}
download_workers=${DOWNLOAD_WORKERS:-8}
maximum_attempts=${MAXIMUM_DOWNLOAD_ATTEMPTS:-20}
retry_delay_seconds=${RETRY_DELAY_SECONDS:-60}
minimum_free_bytes=${MINIMUM_FREE_BYTES:-2147483648}
if (( target_pairs < 1 || pairs_per_shard < 1 || maximum_attempts < 1 )); then
  echo "AMPLFI capacity-extension integer policy is invalid" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT" "$OUTPUT_ROOT/bank"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT

mapfile -t inherited_exclusions < <(
  "$TASK_PYTHON" - "$BASE_ACQUISITION_PLAN" "$BASE_STREAM_MERGE_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


plan_path, merge_path = sys.argv[1:]
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
merge = json.loads(pathlib.Path(merge_path).read_text(encoding="utf-8"))
if (
    plan.get("status") != "development_acquisition_plan"
    or plan.get("selection_rule") != "stratified_exclusion_complement_v1"
    or plan.get("candidate_scores_inspected") is not False
    or plan.get("test_data_opened") is not False
    or plan.get("locked_evaluation_data") is not False
    or plan.get("run") != "O4a"
    or plan.get("detectors") != ["H1", "L1"]
    or merge.get("status") != "verified_streamed_amplfi_background_bank"
    or merge.get("passed") is not True
    or merge.get("parent_plan_sha256") != digest(plan_path)
    or int(merge.get("test_strain_rows_read", -1)) != 0
    or int(merge.get("test_rows_exported", -1)) != 0
):
    raise SystemExit("AMPLFI base stream failed extension replay")
print(str(pathlib.Path(plan_path).resolve()))
for record in plan.get("exclusion_plans", []):
    path = pathlib.Path(record["path"]).resolve()
    if digest(path) != record["sha256"]:
        raise SystemExit(f"AMPLFI inherited exclusion changed: {path}")
    print(path)
PY
)
if (( ${#inherited_exclusions[@]} < 2 )); then
  echo "AMPLFI extension did not recover its complete exclusion lineage" >&2
  exit 3
fi
exclude_args=()
for plan in "${inherited_exclusions[@]}"; do
  exclude_args+=(--exclude-plan "$plan")
done

extension_plan="$OUTPUT_ROOT/source_disjoint_extension_plan.json"
if [[ ! -s "$extension_plan" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli gwosc-plan-disjoint \
    --run O4a \
    --detectors H1 L1 \
    --sample-rate-khz 4 \
    "${exclude_args[@]}" \
    --target-pairs "$target_pairs" \
    --seed "$extension_seed" \
    --output "$extension_plan"
fi

shard_count=$(( (target_pairs + pairs_per_shard - 1) / pairs_per_shard ))
ready=0
final_merge=
final_capacity=
for ((shard = 0; shard < shard_count; shard++)); do
  available=$(df --output=avail -B1 "$CACHE_ROOT" | tail -1)
  if (( available < minimum_free_bytes )); then
    echo "AMPLFI extension cache is below its free-space guard: $available" >&2
    exit 4
  fi
  shard_root="$OUTPUT_ROOT/shard-$shard"
  shard_plan="$shard_root/acquisition_plan.json"
  batch_root="$shard_root/download"
  batch_report="$batch_root/batch_download_report.json"
  background_root="$shard_root/background"
  background_report="$background_root/background_plan_report.json"
  export_report="$shard_root/amplfi_export_report.json"
  eviction_report="$shard_root/source_eviction_report.json"
  mkdir -p "$shard_root"
  if [[ ! -s "$shard_plan" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli gwosc-plan-shard \
      --plan "$extension_plan" \
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
      echo "AMPLFI extension shard exhausted download retries: $shard" >&2
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
  if [[ ! -s "$export_report" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli amplfi-background-export \
      --manifest "$background_root/background_windows.jsonl" \
      --output-dir "$OUTPUT_ROOT/bank" \
      --report "$export_report" \
      --target-sample-rate 2048 \
      --minimum-segment-seconds 16
  fi
  if [[ ! -s "$eviction_report" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli amplfi-background-source-evict \
      --batch-report "$batch_report" \
      --background-report "$background_report" \
      --export-report "$export_report" \
      --cache-root "$CACHE_ROOT" \
      --output "$eviction_report"
  fi

  mapfile -t completed_roots < <(
    find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'shard-*' \
      | while read -r candidate; do
          if [[ -s "$candidate/source_eviction_report.json" ]]; then
            printf '%s\n' "$candidate"
          fi
        done \
      | sort -V
  )
  shard_args=()
  for candidate in "${completed_roots[@]}"; do
    shard_args+=(--shard-dir "$candidate")
  done
  merge_root="$OUTPUT_ROOT/merge-after-$shard"
  "$TASK_PYTHON" -m gwyolo.cli amplfi-background-extension-merge \
    --base-merge-report "$BASE_STREAM_MERGE_REPORT" \
    --extension-plan "$extension_plan" \
    "${shard_args[@]}" \
    --output-dir "$merge_root"
  capacity="$merge_root/amplfi_background_capacity.json"
  if "$TASK_PYTHON" -m gwyolo.cli amplfi-background-capacity-audit \
    --manifest "$merge_root/amplfi_background_train_val.jsonl" \
    --policy "$CAPACITY_POLICY" \
    --output "$capacity"; then
    ready=1
    final_merge="$merge_root/amplfi_background_stream_extension_merge.json"
    final_capacity="$capacity"
    break
  fi
done
if (( ready != 1 )); then
  echo "AMPLFI extension exhausted its predeclared reserve below capacity" >&2
  exit 1
fi

"$TASK_PYTHON" - "$final_merge" "$final_capacity" "$extension_plan" \
  "$GWYOLO_CODE_COMMIT" "$OUTPUT_ROOT/amplfi_background_capacity_extension_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


merge_path, capacity_path, plan_path, commit, target_value = sys.argv[1:]
merge = json.loads(pathlib.Path(merge_path).read_text(encoding="utf-8"))
capacity = json.loads(pathlib.Path(capacity_path).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
if (
    merge.get("status")
    != "verified_extended_streamed_amplfi_background_bank"
    or merge.get("passed") is not True
    or merge.get("candidate_scores_inspected") is not False
    or int(merge.get("test_strain_rows_read", -1)) != 0
    or capacity.get("status") != "amplfi_background_capacity_ready"
    or capacity.get("passed") is not True
    or int(capacity.get("test_strain_rows_read", -1)) != 0
    or capacity.get("manifest_sha256") != merge.get("background_manifest_sha256")
    or plan.get("candidate_scores_inspected") is not False
    or plan.get("test_data_opened") is not False
):
    raise SystemExit("AMPLFI extension failed its final frozen-capacity replay")
result = {
    "status": "verified_capacity_ready_amplfi_background_extension",
    "passed": True,
    "scientific_claim_allowed": False,
    "candidate_scores_inspected": False,
    "test_rows_read": 0,
    "code_commit": commit,
    "extension_plan_path": str(pathlib.Path(plan_path).resolve()),
    "extension_plan_sha256": digest(plan_path),
    "extension_source_pairs_used": merge["extension_source_pairs"],
    "stream_merge_report_path": str(pathlib.Path(merge_path).resolve()),
    "stream_merge_report_sha256": digest(merge_path),
    "capacity_report_path": str(pathlib.Path(capacity_path).resolve()),
    "capacity_report_sha256": digest(capacity_path),
    "background_manifest_path": merge["background_manifest_path"],
    "background_manifest_sha256": merge["background_manifest_sha256"],
    "checks": capacity["checks"],
}
target = pathlib.Path(target_value)
if target.is_file():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing AMPLFI extension receipt has another identity")
else:
    part = target.with_suffix(target.suffix + ".part")
    part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
