#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON TASK_CODE_DIR GWYOLO_CODE_COMMIT PARENT_PLAN EVENT_EXCLUSIONS
  CACHE_ROOT OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

PAIRS_PER_SHARD=${PAIRS_PER_SHARD:-4}
SHARD_COUNT=${SHARD_COUNT:-20}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.2}
BACKGROUND_SEED=${BACKGROUND_SEED:-20260727}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
CAPACITY_POLICY=${CAPACITY_POLICY:-$TASK_CODE_DIR/configs/amplfi_background_capacity_policy.yaml}
for path in "$TASK_PYTHON" "$PARENT_PLAN" "$EVENT_EXCLUSIONS" "$CAPACITY_POLICY"; do
  if [[ ! -s "$path" ]]; then
    echo "required AMPLFI acquisition input is absent: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] || (( PAIRS_PER_SHARD * SHARD_COUNT != 80 )); then
  echo "AMPLFI acquisition requires an exact 80-pair code/plan layout" >&2
  exit 2
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT" "$OUTPUT_ROOT/bank"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src

for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  shard_root="$OUTPUT_ROOT/shard-$shard"
  shard_plan="$shard_root/acquisition_plan.json"
  batch_root="$shard_root/download"
  batch_report="$batch_root/batch_download_report.json"
  background_root="$shard_root/background"
  background_report="$background_root/background_plan_report.json"
  export_report="$shard_root/amplfi_export_report.json"
  eviction_report="$shard_root/source_eviction_report.json"
  mkdir -p "$shard_root"
  if [[ -s "$eviction_report" ]]; then
    continue
  fi
  if [[ ! -s "$shard_plan" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli gwosc-plan-shard \
      --plan "$PARENT_PLAN" \
      --shard-index "$shard" \
      --pairs-per-shard "$PAIRS_PER_SHARD" \
      --output "$shard_plan"
  fi
  if [[ ! -s "$batch_report" ]]; then
    completed=0
    for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
      if "$TASK_PYTHON" -m gwyolo.cli gwosc-batch-download \
        --plan "$shard_plan" \
        --cache-dir "$CACHE_ROOT" \
        --output-dir "$batch_root" \
        --download-workers "$DOWNLOAD_WORKERS"; then
        completed=1
        break
      fi
      if (( attempt < MAX_ATTEMPTS )); then
        sleep "$RETRY_DELAY_SECONDS"
      fi
    done
    if (( completed != 1 )) || [[ ! -s "$batch_report" ]]; then
      echo "AMPLFI acquisition shard $shard exhausted download retries" >&2
      exit 1
    fi
  fi
  if [[ ! -s "$background_report" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli background-batch-plan \
      --batch-report "$batch_report" \
      --event-exclusions "$EVENT_EXCLUSIONS" \
      --output-dir "$background_root" \
      --validation-fraction "$VALIDATION_FRACTION" \
      --test-fraction 0 \
      --seed "$BACKGROUND_SEED" \
      --split-strategy hash_threshold_v1
  fi
  manifest=$(
    "$TASK_PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["manifest_path"])' \
      "$background_report"
  )
  if [[ ! -s "$export_report" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli amplfi-background-export \
      --manifest "$manifest" \
      --output-dir "$OUTPUT_ROOT/bank" \
      --report "$export_report" \
      --target-sample-rate 2048 \
      --minimum-segment-seconds 16
  fi
  "$TASK_PYTHON" -m gwyolo.cli amplfi-background-source-evict \
    --batch-report "$batch_report" \
    --background-report "$background_report" \
    --export-report "$export_report" \
    --cache-root "$CACHE_ROOT" \
    --output "$eviction_report"
done

merged_manifest="$OUTPUT_ROOT/amplfi_background_train_val.jsonl"
merge_report="$OUTPUT_ROOT/amplfi_background_stream_merge.json"
"$TASK_PYTHON" - "$PARENT_PLAN" "$OUTPUT_ROOT" "$SHARD_COUNT" \
  "$PAIRS_PER_SHARD" "$merged_manifest" "$merge_report" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


parent_path, root_value, shard_count, pairs_per_shard, manifest_value, report_value = sys.argv[1:]
root = pathlib.Path(root_value).resolve()
parent = json.loads(pathlib.Path(parent_path).read_text(encoding="utf-8"))
parent_hash = digest(parent_path)
if (
    parent.get("status") != "development_acquisition_plan"
    or parent.get("locked_evaluation_data") is not False
    or parent.get("test_data_opened") is not False
    or parent.get("candidate_scores_inspected") is not False
    or int(parent.get("selected_pairs", -1)) != 80
    or int(parent.get("excluded_unique_pair_ids", -1)) < 880
):
    raise SystemExit("AMPLFI parent is not the frozen disjoint 80-pair plan")
rows = []
exports = []
shards = []
blocks = {}
evicted_source_files = 0
evicted_source_bytes = 0
for shard in range(int(shard_count)):
    shard_root = root / f"shard-{shard}"
    paths = {
        "plan": shard_root / "acquisition_plan.json",
        "batch": shard_root / "download/batch_download_report.json",
        "background": shard_root / "background/background_plan_report.json",
        "export": shard_root / "amplfi_export_report.json",
        "eviction": shard_root / "source_eviction_report.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise SystemExit(f"AMPLFI shard {shard} is incomplete")
    values = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    plan = values["plan"]
    background = values["background"]
    export = values["export"]
    eviction = values["eviction"]
    if (
        plan.get("parent_plan_sha256") != parent_hash
        or int(plan.get("shard_index", -1)) != shard
        or int(plan.get("pair_index_start_inclusive", -1)) != shard * int(pairs_per_shard)
        or int(plan.get("pair_index_stop_exclusive", -1))
        != (shard + 1) * int(pairs_per_shard)
        or background.get("status") != "verified_multi_segment_development_background"
        or background.get("passed") is not True
        or background.get("split_strategy") != "hash_threshold_v1"
        or int(background.get("splits", {}).get("test", {}).get("windows", -1)) != 0
        or export.get("status") != "group_safe_amplfi_background"
        or int(export.get("split_file_counts", {}).get("test", -1)) != 0
        or eviction.get("status") != "verified_exported_amplfi_source_eviction"
        or eviction.get("recoverable") is not True
        or eviction.get("amplfi_export_report_sha256") != digest(paths["export"])
        or pathlib.Path(export.get("manifest_path", "")).resolve()
        != pathlib.Path(background.get("manifest_path", "")).resolve()
        or export.get("manifest_sha256") != background.get("manifest_sha256")
    ):
        raise SystemExit(f"AMPLFI shard {shard} failed identity replay")
    for item in export.get("files", []):
        path = pathlib.Path(item["path"])
        if not path.is_file() or digest(path) != item.get("sha256"):
            raise SystemExit(f"AMPLFI shard {shard} export changed")
        exports.append(item)
    manifest = pathlib.Path(background["manifest_path"])
    if digest(manifest) != background.get("manifest_sha256"):
        raise SystemExit(f"AMPLFI shard {shard} manifest changed")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("split") not in {"train", "val"}:
            raise SystemExit("AMPLFI merge encountered a non-training split")
        block = str(row["gps_block"])
        if block in blocks and blocks[block] != row["split"]:
            raise SystemExit("AMPLFI GPS block crossed train/validation")
        blocks[block] = row["split"]
        rows.append(row)
    shards.append({key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()})
    evicted_source_files += int(eviction.get("removed_files", 0))
    evicted_source_bytes += int(eviction.get("removed_bytes", 0))
window_ids = [str(row["window_id"]) for row in rows]
export_paths = [str(row["path"]) for row in exports]
if len(window_ids) != len(set(window_ids)) or len(export_paths) != len(set(export_paths)):
    raise SystemExit("AMPLFI merge repeats windows or exported files")
rows.sort(key=lambda row: (str(row["split"]), int(row["gps_start"]), str(row["window_id"])))
manifest_target = pathlib.Path(manifest_value)
manifest_part = manifest_target.with_suffix(manifest_target.suffix + ".part")
manifest_part.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
os.replace(manifest_part, manifest_target)
result = {
    "status": "verified_streamed_amplfi_background_bank",
    "passed": True,
    "scientific_claim_allowed": False,
    "test_strain_rows_read": 0,
    "test_rows_exported": 0,
    "parent_plan_path": str(pathlib.Path(parent_path).resolve()),
    "parent_plan_sha256": parent_hash,
    "shards": shards,
    "shard_count": len(shards),
    "source_pairs": 80,
    "background_manifest_path": str(manifest_target.resolve()),
    "background_manifest_sha256": digest(manifest_target),
    "background_windows": len(rows),
    "unique_gps_blocks": len(blocks),
    "exported_files": len(exports),
    "exported_file_bytes": sum(pathlib.Path(row["path"]).stat().st_size for row in exports),
    "cross_split_gps_block_overlap": 0,
    "source_files_evicted": evicted_source_files,
    "source_bytes_evicted": evicted_source_bytes,
    "recoverable": True,
}
report_target = pathlib.Path(report_value)
report_part = report_target.with_suffix(report_target.suffix + ".part")
report_part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(report_part, report_target)
PY

capacity_report="$OUTPUT_ROOT/amplfi_background_capacity.json"
if [[ ! -s "$capacity_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli amplfi-background-capacity-audit \
    --manifest "$merged_manifest" \
    --policy "$CAPACITY_POLICY" \
    --output "$capacity_report"
fi

"$TASK_PYTHON" - "$merge_report" "$capacity_report" "$GWYOLO_CODE_COMMIT" \
  "$OUTPUT_ROOT/amplfi_background_acquisition_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


merge_path, capacity_path, commit, target_value = sys.argv[1:]
merge = json.loads(pathlib.Path(merge_path).read_text(encoding="utf-8"))
capacity = json.loads(pathlib.Path(capacity_path).read_text(encoding="utf-8"))
if (
    merge.get("status") != "verified_streamed_amplfi_background_bank"
    or merge.get("passed") is not True
    or capacity.get("status") != "amplfi_background_capacity_ready"
    or capacity.get("passed") is not True
    or capacity.get("test_strain_rows_read") != 0
    or capacity.get("manifest_sha256") != merge.get("background_manifest_sha256")
):
    raise SystemExit("AMPLFI streamed background did not pass the frozen capacity gate")
result = {
    "status": "verified_capacity_ready_amplfi_training_background",
    "passed": True,
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "stream_merge_report_path": str(pathlib.Path(merge_path).resolve()),
    "stream_merge_report_sha256": digest(merge_path),
    "capacity_report_path": str(pathlib.Path(capacity_path).resolve()),
    "capacity_report_sha256": digest(capacity_path),
    "background_manifest_path": merge["background_manifest_path"],
    "background_manifest_sha256": merge["background_manifest_sha256"],
    "checks": capacity["checks"],
    "code_commit": commit,
}
target = pathlib.Path(target_value)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing AMPLFI acquisition receipt has another identity")
else:
    part = target.with_suffix(target.suffix + ".part")
    part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
