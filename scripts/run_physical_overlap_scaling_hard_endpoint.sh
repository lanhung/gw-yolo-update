#!/usr/bin/env bash
set -euo pipefail

# Bind the two-control mask-IoU scaling diagnostic to one score-blind validation
# hard subset. The subset is frozen before this script reads any scaling metric.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SCALING_OUTPUT_ROOT
  OVERLAP_VALIDATION_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  OUTPUT_ROOT
  NEXT_PHYSICAL_SCALE
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

hard_config=${HARD_SUBSET_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scaling_hard_subset.yaml}
fixed_epoch_config=${FIXED_EPOCH_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scale_fixed_epochs.yaml}
fixed_update_config=${FIXED_UPDATE_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scale_fixed_updates.yaml}
scaling_summary=${SCALING_SUMMARY:-$SCALING_OUTPUT_ROOT/physical_overlap_data_scaling_summary.json}
subset_report=${SCALING_SUBSET_REPORT:-$SCALING_OUTPUT_ROOT/subsets/physical_overlap_scaling_subsets.json}
for path in \
  "$TASK_PYTHON" \
  "$OVERLAP_VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$hard_config" \
  "$fixed_epoch_config" \
  "$fixed_update_config"; do
  if [[ ! -s "$path" ]]; then
    echo "required hard-endpoint artifact is absent: $path" >&2
    exit 3
  fi
done
if [[ ! "$NEXT_PHYSICAL_SCALE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NEXT_PHYSICAL_SCALE must be a positive integer" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi
assigned_gpu=${HARD_ENDPOINT_CUDA_VISIBLE_DEVICES:-0}
if ! [[ "$assigned_gpu" =~ ^[0-9]+$ ]]; then
  echo "hard endpoint requires one non-negative physical GPU index" >&2
  exit 3
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "hard-endpoint output root is immutable: $OUTPUT_ROOT" >&2
  exit 4
fi

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
export CUDA_VISIBLE_DEVICES="$assigned_gpu"
mkdir -p "$OUTPUT_ROOT/logs"
hard_root="$OUTPUT_ROOT/hard-subset"

# This must happen before the script reads scaling_summary or any finetune report.
"$TASK_PYTHON" -m gwyolo.cli physical-overlap-scale-hard-subset-freeze \
  --validation-manifest "$OVERLAP_VALIDATION_MANIFEST" \
  --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT" \
  --config "$hard_config" \
  --output-dir "$hard_root" \
  >"$OUTPUT_ROOT/logs/hard-subset-freeze.log" 2>&1
hard_report="$hard_root/physical_overlap_scaling_hard_subset_report.json"

if [[ ! -s "$scaling_summary" || ! -s "$subset_report" ]]; then
  echo "scaling diagnostic or subset report is absent after hard-subset freeze" >&2
  exit 5
fi

if ! resolved=$(
  "$TASK_PYTHON" - "$scaling_summary" "$subset_report" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary_path = pathlib.Path(sys.argv[1]).resolve()
subset_path = pathlib.Path(sys.argv[2]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
subset = json.loads(subset_path.read_text(encoding="utf-8"))
if (
    summary.get("status") != "completed_group_safe_physical_overlap_data_scaling_curve"
    or summary.get("passed") is not True
    or summary.get("test_rows_read") != 0
    or pathlib.Path(summary.get("subset_report_path", "")).resolve() != subset_path
    or summary.get("subset_report_sha256") != digest(subset_path)
):
    raise SystemExit("scaling diagnostic failed hard-endpoint handoff")
scale_by_hash = {
    row["manifest_sha256"]: int(row["scale"])
    for row in subset.get("subsets", [])
}
for identity in summary.get("finetune_reports", []):
    path = pathlib.Path(identity["path"]).resolve()
    if not path.is_file() or digest(path) != identity["sha256"]:
        raise SystemExit("finetune report changed before hard-endpoint evaluation")
    report = json.loads(path.read_text(encoding="utf-8"))
    control = report.get("training_control", {}).get("control")
    scale = scale_by_hash.get(report.get("overlap_train_manifest_sha256"))
    seed = report.get("seed")
    if control not in {"fixed_epochs", "fixed_optimizer_updates"} or scale is None:
        raise SystemExit("finetune report lacks its scaling cell identity")
    print(f"{control}\t{scale}\t{seed}\t{path}")
PY
); then
  echo "hard-endpoint scaling-cell resolution failed" >&2
  exit 5
fi
mapfile -t cell_lines <<<"$resolved"
if (( ${#cell_lines[@]} < 30 )); then
  echo "hard-endpoint evaluation requires the complete two-control five-seed matrix" >&2
  exit 5
fi

wait_for_idle_gpu() {
  while true; do
    gpu_pids=$(
      nvidia-smi -i "$assigned_gpu" \
        --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d' || true
    )
    [[ -z "$gpu_pids" ]] && return
    sleep 30
  done
}

cell_args=()
for line in "${cell_lines[@]}"; do
  IFS=$'\t' read -r control scale seed finetune_report <<<"$line"
  if [[ "$control" == "fixed_epochs" ]]; then
    config="$fixed_epoch_config"
  else
    config="$fixed_update_config"
  fi
  cell_dir="$OUTPUT_ROOT/cells/$control/scale-$scale/seed-$seed"
  cell_report="$cell_dir/hard_endpoint_cell.json"
  mkdir -p "$cell_dir"
  wait_for_idle_gpu
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-scale-hard-endpoint-cell \
    --config "$config" \
    --subset-report "$subset_report" \
    --hard-subset-report "$hard_report" \
    --finetune-report "$finetune_report" \
    --scale "$scale" \
    --output "$cell_report" \
    >"$OUTPUT_ROOT/logs/cell-$control-$scale-$seed.log" 2>&1
  cell_args+=(--hard-endpoint-report "$cell_report")
done

"$TASK_PYTHON" -m gwyolo.cli physical-overlap-scale-hard-endpoint-bind \
  --scaling-summary "$scaling_summary" \
  --hard-subset-report "$hard_report" \
  "${cell_args[@]}" \
  --next-scale "$NEXT_PHYSICAL_SCALE" \
  --output "$OUTPUT_ROOT/physical_overlap_data_scaling_hard_endpoint_bound.json" \
  >"$OUTPUT_ROOT/logs/hard-endpoint-bind.log" 2>&1
