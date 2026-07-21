#!/usr/bin/env bash
set -euo pipefail

# Run the group-safe overlap data-scaling curve under both equal-epoch and
# equal-optimizer-update controls. All model selection remains validation-only.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  OVERLAP_TRAIN_MANIFEST
  OVERLAP_VALIDATION_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

fixed_epoch_config=${FIXED_EPOCH_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scale_fixed_epochs.yaml}
fixed_update_config=${FIXED_UPDATE_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scale_fixed_updates.yaml}
for path in \
  "$TASK_PYTHON" \
  "$OVERLAP_TRAIN_MANIFEST" \
  "$OVERLAP_VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$fixed_epoch_config" \
  "$fixed_update_config"; do
  if [[ ! -s "$path" ]]; then
    echo "required overlap-scaling artifact is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi

read -r -a scales <<<"${OVERLAP_SCALES:-250 500 1000}"
read -r -a seeds <<<"${OVERLAP_SCALE_SEEDS:-20260728 20260729 20260730 20260731 20260732}"
if (( ${#scales[@]} < 3 || ${#seeds[@]} != 5 )); then
  echo "overlap scaling requires at least three scales and exactly five seeds" >&2
  exit 4
fi
scale_args=()
for scale in "${scales[@]}"; do
  if [[ ! "$scale" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid overlap scale: $scale" >&2
    exit 4
  fi
  scale_args+=(--scale "$scale")
done
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid overlap scale seed: $seed" >&2
    exit 4
  fi
done

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
export CUDA_VISIBLE_DEVICES="${OVERLAP_CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_ROOT/logs"
subset_root="$OUTPUT_ROOT/subsets"
subset_report="$subset_root/physical_overlap_scaling_subsets.json"

"$TASK_PYTHON" -m gwyolo.cli physical-overlap-scale-subsets \
  --train-manifest "$OVERLAP_TRAIN_MANIFEST" \
  --validation-manifest "$OVERLAP_VALIDATION_MANIFEST" \
  --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT" \
  "${scale_args[@]}" \
  --include-full \
  --seed "${OVERLAP_SCALE_RANK_SEED:-20260728}" \
  --output-dir "$subset_root" \
  >"$OUTPUT_ROOT/logs/subsets.log" 2>&1

if ! resolved=$(
  "$TASK_PYTHON" - "$subset_report" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "frozen_group_safe_physical_overlap_scaling_subsets"
    or report.get("passed") is not True
    or report.get("test_rows_read") != 0
):
    raise SystemExit("overlap scaling subset freeze failed")
for row in report["subsets"]:
    path = pathlib.Path(row["manifest_path"])
    if not path.is_file() or digest(path) != row["manifest_sha256"]:
        raise SystemExit("overlap scaling subset replay failed")
    print(f'{row["scale"]}\t{path}')
PY
); then
  echo "overlap scaling subset resolution failed" >&2
  exit 5
fi
mapfile -t subset_lines <<<"$resolved"
if (( ${#subset_lines[@]} < 3 )); then
  echo "overlap scaling produced fewer than three nested subsets" >&2
  exit 5
fi

wait_for_idle_gpu() {
  while true; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && return
    sleep 30
  done
}

reports=()
cache_args=()
if [[ -n "${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" ]]; then
  cache_args=(
    --clean-validation-feature-cache-dir "$CLEAN_VALIDATION_FEATURE_CACHE_DIR"
  )
fi
for control in fixed_epochs fixed_optimizer_updates; do
  if [[ "$control" == "fixed_epochs" ]]; then
    config="$fixed_epoch_config"
  else
    config="$fixed_update_config"
  fi
  for line in "${subset_lines[@]}"; do
    IFS=$'\t' read -r scale manifest <<<"$line"
    for seed in "${seeds[@]}"; do
      run_root="$OUTPUT_ROOT/runs/$control/scale-$scale/seed-$seed"
      mkdir -p "$run_root"
      wait_for_idle_gpu
      "$TASK_PYTHON" -m gwyolo.cli physical-overlap-finetune \
        --config "$config" \
        --overlap-train-manifest "$manifest" \
        --overlap-validation-manifest "$OVERLAP_VALIDATION_MANIFEST" \
        --clean-train-manifest "$CLEAN_TRAIN_MANIFEST" \
        --clean-validation-manifest "$CLEAN_VALIDATION_MANIFEST" \
        --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
        --output-dir "$run_root" \
        --seed "$seed" \
        "${cache_args[@]}" \
        >"$OUTPUT_ROOT/logs/$control-scale-$scale-seed-$seed.log" 2>&1
      report="$run_root/overlap_finetune_report.json"
      if [[ ! -s "$report" ]]; then
        echo "overlap scaling run omitted its report: $report" >&2
        exit 6
      fi
      reports+=(--report "$report")
    done
  done
done

"$TASK_PYTHON" -m gwyolo.cli physical-overlap-scale-summarize \
  --subset-report "$subset_report" \
  "${reports[@]}" \
  --output "$OUTPUT_ROOT/physical_overlap_data_scaling_summary.json" \
  --minimum-seeds 5 \
  --minimum-material-glitch-iou-gain "${MINIMUM_MATERIAL_GLITCH_IOU_GAIN:-0.01}" \
  --minimum-clean-chirp-iou-retention "${MINIMUM_CLEAN_CHIRP_IOU_RETENTION:-0.95}" \
  --bootstrap-replicates "${OVERLAP_SCALE_BOOTSTRAP_REPLICATES:-2000}" \
  --bootstrap-seed "${OVERLAP_SCALE_BOOTSTRAP_SEED:-20260728}" \
  >"$OUTPUT_ROOT/logs/summary.log" 2>&1
