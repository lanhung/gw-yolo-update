#!/usr/bin/env bash
set -euo pipefail

# Expand the frozen one-seed glitch adapter only after its absolute validation
# gate passes. The expensive data-scaling curve runs only after five-seed
# stability passes. No test data are opened by this successor.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  ORIGINAL_ADAPTER_REPORT
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
    echo "required glitch-adapter successor variable is unset: $variable" >&2
    exit 2
  fi
done

adapter_config=${ADAPTER_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_finetune_glitch_adapter.yaml}
promotion_config=${PROMOTION_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_sampling_promotion.yaml}
stability_config=${STABILITY_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_five_seed_stability.yaml}
fixed_epoch_config=${FIXED_EPOCH_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scale_fixed_epochs_glitch_adapter.yaml}
fixed_update_config=${FIXED_UPDATE_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_scale_fixed_updates_glitch_adapter.yaml}
assigned_gpu=${GWYOLO_ASSIGNED_GPU_INDEX:-0}
for path in \
  "$TASK_PYTHON" \
  "$ORIGINAL_ADAPTER_REPORT" \
  "$OVERLAP_TRAIN_MANIFEST" \
  "$OVERLAP_VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$adapter_config" \
  "$promotion_config" \
  "$stability_config" \
  "$fixed_epoch_config" \
  "$fixed_update_config"; do
  if [[ ! -s "$path" ]]; then
    echo "required glitch-adapter successor input is absent: $path" >&2
    exit 3
  fi
done
if ! [[ "$assigned_gpu" =~ ^[0-9]+$ ]]; then
  echo "assigned GPU index must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "glitch-adapter successor checkout differs from its declared commit" >&2
  exit 3
fi
"$TASK_PYTHON" - \
  "$adapter_config" "$fixed_epoch_config" "$fixed_update_config" <<'PY'
import pathlib
import sys

import yaml


def settings(path):
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))[
        "overlap_training"
    ]


base, fixed_epochs, fixed_updates = (settings(path) for path in sys.argv[1:])
if base.get("training_scope") != "glitch_adapter_only":
    raise SystemExit("base config is not the frozen glitch adapter")
ignored = {"seed", "training_control", "epochs", "max_optimizer_updates"}
base_policy = {key: value for key, value in base.items() if key not in ignored}
for name, candidate in (
    ("fixed_epochs", fixed_epochs),
    ("fixed_optimizer_updates", fixed_updates),
):
    candidate_policy = {
        key: value for key, value in candidate.items() if key not in ignored
    }
    if candidate_policy != base_policy or candidate.get("training_control") != name:
        raise SystemExit(f"{name} changes the frozen glitch-adapter policy")
if fixed_epochs.get("epochs") != 20 or "max_optimizer_updates" in fixed_epochs:
    raise SystemExit("fixed-epoch glitch-adapter control changed")
if (
    fixed_updates.get("max_optimizer_updates") != 4000
    or int(fixed_updates.get("epochs", 0)) < 20
):
    raise SystemExit("fixed-update glitch-adapter control changed")
PY

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
mkdir -p "$OUTPUT_ROOT"
promotion="$OUTPUT_ROOT/glitch_adapter_one_seed_promotion.json"
summary="$OUTPUT_ROOT/five-seed/five_seed_overlap_summary.json"
five_seed_gate_receipt="$OUTPUT_ROOT/glitch_adapter_five_seed_gate_receipt.json"
scaling_root="$OUTPUT_ROOT/scaling"
scaling_summary="$scaling_root/physical_overlap_data_scaling_summary.json"
hard_root="$OUTPUT_ROOT/hard-endpoint"
hard_report="$hard_root/physical_overlap_data_scaling_hard_endpoint_bound.json"
receipt="$OUTPUT_ROOT/glitch_adapter_validation_scaling_successor_receipt.json"

write_receipt() {
  local status=$1
  local five_seed_promoted=$2
  local scaling_status=$3
  local hard_status=$4
  "$TASK_PYTHON" - \
    "$receipt" "$status" "$five_seed_promoted" "$scaling_status" "$hard_status" \
    "$ORIGINAL_ADAPTER_REPORT" "$promotion" "$summary" "$scaling_summary" \
    "$hard_report" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    raw_output,
    status,
    raw_five_seed_promoted,
    scaling_status,
    hard_status,
    raw_original,
    raw_promotion,
    raw_summary,
    raw_scaling,
    raw_hard,
    commit,
) = sys.argv[1:]
artifacts = {}
for label, raw_path, required in (
    ("original_adapter_report", raw_original, True),
    ("one_seed_promotion", raw_promotion, True),
    ("five_seed_summary", raw_summary, raw_five_seed_promoted == "true"),
    ("scaling_summary", raw_scaling, scaling_status == "completed"),
    ("hard_endpoint", raw_hard, hard_status == "completed"),
):
    path = pathlib.Path(raw_path).resolve()
    if required and not path.is_file():
        raise SystemExit(f"required successor artifact is absent: {label}")
    if path.is_file():
        artifacts[label] = {"path": str(path), "sha256": digest(path)}
result = {
    "status": status,
    "execution_passed": True,
    "five_seed_promoted": raw_five_seed_promoted == "true",
    "scaling_status": scaling_status,
    "hard_endpoint_status": hard_status,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "scientific_blocker": (
        "validation-only mask evidence cannot replace continuous-background "
        "FAR/IFAR/<VT> or the one-time locked evaluation"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "code_commit": commit,
    "artifacts": artifacts,
}
output = pathlib.Path(raw_output)
output.parent.mkdir(parents=True, exist_ok=True)
part = output.with_suffix(output.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, output)
PY
}

if [[ ! -s "$promotion" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-single-arm-promote \
    --report "$ORIGINAL_ADAPTER_REPORT" \
    --config "$promotion_config" \
    --arm glitch_adapter \
    --output "$promotion"
fi
one_seed_passed=$(
  "$TASK_PYTHON" - "$promotion" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "validation_only_overlap_single_arm_promotion"
    or report.get("test_data_opened") is not False
    or report.get("scientific_claim_allowed") is not False
):
    raise SystemExit("single-arm promotion violates the validation boundary")
print("true" if report.get("passed") is True else "false")
PY
)
if [[ "$one_seed_passed" != true ]]; then
  write_receipt \
    completed_glitch_adapter_negative_one_seed false \
    not_authorized_by_one_seed_gate not_run
  exit 0
fi

wait_for_idle_assigned_gpu() {
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

reports=(--report "$ORIGINAL_ADAPTER_REPORT")
cache_args=()
if [[ -n "${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" ]]; then
  cache_args=(
    --clean-validation-feature-cache-dir "$CLEAN_VALIDATION_FEATURE_CACHE_DIR"
  )
fi
for seed in 20260721 20260722 20260723 20260724; do
  run="$OUTPUT_ROOT/five-seed/glitch-adapter-seed$seed"
  report="$run/overlap_finetune_report.json"
  if [[ ! -s "$report" ]]; then
    wait_for_idle_assigned_gpu
    env CUDA_VISIBLE_DEVICES="$assigned_gpu" \
      "$TASK_PYTHON" -m gwyolo.cli physical-overlap-finetune \
      --config "$adapter_config" \
      --overlap-train-manifest "$OVERLAP_TRAIN_MANIFEST" \
      --overlap-validation-manifest "$OVERLAP_VALIDATION_MANIFEST" \
      --clean-train-manifest "$CLEAN_TRAIN_MANIFEST" \
      --clean-validation-manifest "$CLEAN_VALIDATION_MANIFEST" \
      --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
      "${cache_args[@]}" \
      --output-dir "$run" \
      --seed "$seed"
  fi
  if [[ ! -s "$report" ]]; then
    echo "glitch-adapter seed omitted its report: $report" >&2
    exit 5
  fi
  reports+=(--report "$report")
done

mkdir -p "$(dirname "$summary")"
if [[ ! -s "$summary" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-five-seed-summarize \
    --promotion-report "$promotion" \
    "${reports[@]}" \
    --stability-config "$stability_config" \
    --output "$summary"
fi
five_seed_passed=$(
  "$TASK_PYTHON" - "$summary" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_five_seed_source_safe_overlap_validation"
    or report.get("promoted_arm") != "glitch_adapter"
    or report.get("test_data_opened") is not False
):
    raise SystemExit("glitch-adapter five-seed summary failed boundary replay")
print("true" if report.get("passed") is True else "false")
PY
)
if [[ "$five_seed_passed" != true ]]; then
  write_receipt \
    completed_glitch_adapter_negative_five_seed false \
    not_authorized_by_five_seed_gate not_run
  exit 0
fi
if [[ ! -s "$five_seed_gate_receipt" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-adapter-five-seed-gate \
    --original-report "$ORIGINAL_ADAPTER_REPORT" \
    --promotion-report "$promotion" \
    --five-seed-summary "$summary" \
    --output "$five_seed_gate_receipt"
fi

if [[ ! -s "$scaling_summary" ]]; then
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    OVERLAP_TRAIN_MANIFEST="$OVERLAP_TRAIN_MANIFEST" \
    OVERLAP_VALIDATION_MANIFEST="$OVERLAP_VALIDATION_MANIFEST" \
    GRAVITYSPY_CORPUS_AUDIT="$GRAVITYSPY_CORPUS_AUDIT" \
    CLEAN_TRAIN_MANIFEST="$CLEAN_TRAIN_MANIFEST" \
    CLEAN_VALIDATION_MANIFEST="$CLEAN_VALIDATION_MANIFEST" \
    PRETRAINED_CHECKPOINT="$PRETRAINED_CHECKPOINT" \
    CLEAN_VALIDATION_FEATURE_CACHE_DIR="${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" \
    FIXED_EPOCH_CONFIG="$fixed_epoch_config" \
    FIXED_UPDATE_CONFIG="$fixed_update_config" \
    OUTPUT_ROOT="$scaling_root" \
    OVERLAP_CUDA_VISIBLE_DEVICES="$assigned_gpu" \
    OVERLAP_SCALES="${OVERLAP_SCALES:-250 500 1000}" \
    OVERLAP_SCALE_SEEDS="${OVERLAP_SCALE_SEEDS:-20260728 20260729 20260730 20260731 20260732}" \
    bash scripts/run_physical_overlap_data_scaling.sh
fi
if [[ ! -s "$scaling_summary" ]]; then
  echo "glitch-adapter scaling omitted its summary" >&2
  exit 6
fi

next_scale=$(
  "$TASK_PYTHON" - "$scaling_summary" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status")
    != "completed_group_safe_physical_overlap_data_scaling_curve"
    or report.get("passed") is not True
    or report.get("test_rows_read") != 0
):
    raise SystemExit("glitch-adapter scaling summary is incomplete")
if report.get("scale_promotion_authorized") is not True:
    print("-")
else:
    lower, upper = (int(value) for value in report["promotion_data_doubling"])
    maximum = max(int(value) for value in report["scales"])
    candidate = min(int(2.5 * upper), max(2 * upper, maximum + 1))
    print(candidate if lower > 0 and upper / lower >= 1.8 else "-")
PY
)
hard_status=not_authorized_by_scaling_gate
if [[ "$next_scale" != - ]]; then
  if [[ ! -s "$hard_report" ]]; then
    env \
      TASK_PYTHON="$TASK_PYTHON" \
      TASK_CODE_DIR="$TASK_CODE_DIR" \
      GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
      SCALING_OUTPUT_ROOT="$scaling_root" \
      OVERLAP_VALIDATION_MANIFEST="$OVERLAP_VALIDATION_MANIFEST" \
      GRAVITYSPY_CORPUS_AUDIT="$GRAVITYSPY_CORPUS_AUDIT" \
      FIXED_EPOCH_CONFIG="$fixed_epoch_config" \
      FIXED_UPDATE_CONFIG="$fixed_update_config" \
      OUTPUT_ROOT="$hard_root" \
      NEXT_PHYSICAL_SCALE="$next_scale" \
      HARD_ENDPOINT_CUDA_VISIBLE_DEVICES="$assigned_gpu" \
      bash scripts/run_physical_overlap_scaling_hard_endpoint.sh
  fi
  if [[ ! -s "$hard_report" ]]; then
    echo "glitch-adapter hard endpoint omitted its report" >&2
    exit 7
  fi
  hard_status=completed
fi

write_receipt \
  completed_glitch_adapter_validation_scaling_successor true \
  completed "$hard_status"
