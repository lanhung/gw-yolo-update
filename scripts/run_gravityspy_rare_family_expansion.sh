#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  CAPACITY_REPORT
  CANDIDATE_PLAN
  CURRENT_TRAIN_PLAN
  CURRENT_VALIDATION_PLAN
  EXISTING_TRAIN_MANIFEST
  EXISTING_VALIDATION_MANIFEST
  BASE_TRAIN_REPORT
  BASE_VALIDATION_REPORT
  CONFIG
  CACHE_ROOT
  OUTPUT_ROOT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$CAPACITY_REPORT" \
  "$CANDIDATE_PLAN" \
  "$CURRENT_TRAIN_PLAN" \
  "$CURRENT_VALIDATION_PLAN" \
  "$EXISTING_TRAIN_MANIFEST" \
  "$EXISTING_VALIDATION_MANIFEST" \
  "$BASE_TRAIN_REPORT" \
  "$BASE_VALIDATION_REPORT" \
  "$CONFIG"; do
  if [[ ! -s "$input" ]]; then
    echo "required rare-family input is absent: $input" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD)" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "rare-family checkout differs from GWYOLO_CODE_COMMIT" >&2
  exit 2
fi

TARGET_LABEL=${TARGET_LABEL:-Helix}
TARGET_ADDITIONAL_ROWS=${TARGET_ADDITIONAL_ROWS:-95}
MAXIMUM_SOURCE_FILES=${MAXIMUM_SOURCE_FILES:-103}
FILES_PER_SHARD=${FILES_PER_SHARD:-16}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
for value in \
  "$TARGET_ADDITIONAL_ROWS" \
  "$MAXIMUM_SOURCE_FILES" \
  "$FILES_PER_SHARD" \
  "$MINIMUM_FREE_KB" \
  "$MAX_ATTEMPTS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "rare-family integer settings must be positive" >&2
    exit 2
  fi
done
if ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "RETRY_DELAY_SECONDS must be non-negative" >&2
  exit 2
fi

decision=$("$TASK_PYTHON" - "$CAPACITY_REPORT" "$TARGET_LABEL" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
label = sys.argv[2]
if report.get("status") != "score_blind_gravityspy_family_capacity_forecast":
    raise SystemExit("rare-family expansion received another capacity report")
if report.get("acquisition_plan_complete") is not True:
    raise SystemExit("rare-family expansion cannot run before the frozen plan completes")
if report.get("passed") is True:
    print("already-ready")
elif label in report.get("families_with_current_shortfall", []):
    print("expand")
else:
    raise SystemExit(f"unsupported rare-family shortfalls: {report.get('families_with_current_shortfall')}")
PY
)
if [[ "$decision" == already-ready ]]; then
  exit 0
fi

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src
selection_root="$OUTPUT_ROOT/selection"
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-strain-select \
  --manifest "$CANDIDATE_PLAN" \
  --output-dir "$selection_root" \
  --per-label "$TARGET_ADDITIONAL_ROWS" \
  --maximum-source-files "$MAXIMUM_SOURCE_FILES" \
  --seed 20260720 \
  --target-label "$TARGET_LABEL" \
  --exclusion-manifest "$CURRENT_TRAIN_PLAN" \
  --exclusion-manifest "$CURRENT_VALIDATION_PLAN" \
  --exclusion-manifest "$EXISTING_TRAIN_MANIFEST" \
  --exclusion-manifest "$EXISTING_VALIDATION_MANIFEST"
selection_report="$selection_root/gravityspy_network_source_selection_report.json"
selection_manifest="$selection_root/gravityspy_network_train_selected_sources.jsonl"
"$TASK_PYTHON" - "$selection_report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("target_met") is not True or int(report.get("selected_rows", 0)) <= 0:
    raise SystemExit("rare-family source selection did not meet its frozen target")
PY

shard_root="$OUTPUT_ROOT/shards"
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-strain-shard \
  --manifest "$selection_manifest" \
  --output-dir "$shard_root" \
  --files-per-shard "$FILES_PER_SHARD" \
  --seed 20260720
shard_manifest="$shard_root/gravityspy_network_strain_shards.jsonl"
shard_report="$shard_root/gravityspy_network_strain_shard_report.json"
shard_count=$("$TASK_PYTHON" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["shards"]))' \
  "$shard_report")

reports=()
for ((shard = 0; shard < shard_count; shard++)); do
  available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < MINIMUM_FREE_KB )); then
    echo "insufficient cache space before rare-family shard $shard" >&2
    exit 1
  fi
  shard_output="$OUTPUT_ROOT/materialized-shard-$shard"
  completed=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if "$TASK_PYTHON" -m gwyolo.cli gravityspy-network-strain-materialize \
      --manifest "$shard_manifest" \
      --shard "$shard" \
      --config "$CONFIG" \
      --cache-dir "$CACHE_ROOT" \
      --output-dir "$shard_output" \
      --output-duration 8 \
      --download-workers 8 \
      --chunk-samples 1048576; then
      completed=1
      break
    fi
    if (( attempt < MAX_ATTEMPTS )); then
      sleep "$RETRY_DELAY_SECONDS"
    fi
  done
  if (( completed == 0 )); then
    echo "rare-family shard $shard exhausted bounded retries" >&2
    exit 1
  fi
  report="$shard_output/gravityspy_network_numeric_report.json"
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-strain-evict \
    --materialization-report "$report" \
    --cache-dir "$CACHE_ROOT" \
    --output "$shard_output/source_eviction_report.json"
  reports+=(--report "$report")
done

extra_root="$OUTPUT_ROOT/extra-merged"
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-numeric-merge \
  "${reports[@]}" \
  --output-dir "$extra_root" \
  --split train
augmented_train_root="$OUTPUT_ROOT/augmented-train"
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-numeric-merge \
  --report "$BASE_TRAIN_REPORT" \
  --report "$extra_root/gravityspy_network_numeric_merge_report.json" \
  --output-dir "$augmented_train_root" \
  --split train

resplit_root="$OUTPUT_ROOT/source-component-safe-resplit"
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-corpus-resplit \
  --report "$augmented_train_root/gravityspy_network_numeric_merge_report.json" \
  --report "$BASE_VALIDATION_REPORT" \
  --output-dir "$resplit_root" \
  --validation-fraction 0.2 \
  --minimum-validation-rows-per-family 5 \
  --seed 20260720
"$TASK_PYTHON" -m gwyolo.cli gravityspy-network-corpus-audit \
  --train-report "$resplit_root/gravityspy_network_numeric_train_report.json" \
  --validation-report "$resplit_root/gravityspy_network_numeric_val_report.json" \
  --output "$OUTPUT_ROOT/gravityspy_network_augmented_corpus_audit.json"
