#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FAILED_HEAD_CODE_COMMIT
  FAILED_HEAD_CHAIN_ROOT
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required glitch-adapter fallback variable is unset: $variable" >&2
    exit 2
  fi
done

SEED=${SEED:-20260720}
CLEAN_VALIDATION_FEATURE_CACHE_DIR=${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}
if ! [[ "$SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "glitch-adapter fallback seed must be a positive integer" >&2
  exit 2
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "glitch-adapter fallback checkout differs from its declared commit" >&2
  exit 3
fi
config="$TASK_CODE_DIR/configs/physical_overlap_finetune_glitch_adapter.yaml"
receipt="$FAILED_HEAD_CHAIN_ROOT/source_safe_overlap_chain_receipt.json"
for path in \
  "$TASK_PYTHON" \
  "$config" \
  "$receipt" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT"; do
  if [[ ! -s "$path" ]]; then
    echo "glitch-adapter fallback input is absent: $path" >&2
    exit 3
  fi
done

train_overlap="$FAILED_HEAD_CHAIN_ROOT/train-overlaps"
validation_overlap="$FAILED_HEAD_CHAIN_ROOT/val-overlaps"
for root in "$train_overlap" "$validation_overlap"; do
  if [[ ! -s "$root/physical_overlap_report.json" ]]; then
    echo "glitch-adapter fallback cannot replay overlap root: $root" >&2
    exit 3
  fi
done

mkdir -p "$OUTPUT_ROOT"
authorization="$OUTPUT_ROOT/glitch_adapter_fallback_authorization.json"
if [[ ! -s "$authorization" ]]; then
  "$TASK_PYTHON" - \
    "$receipt" \
    "$config" \
    "$authorization" \
    "$FAILED_HEAD_CODE_COMMIT" \
    "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt_path = pathlib.Path(sys.argv[1]).resolve()
config_path = pathlib.Path(sys.argv[2]).resolve()
output = pathlib.Path(sys.argv[3])
failed_commit = sys.argv[4]
fallback_commit = sys.argv[5]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("status")
    not in {
        "completed_source_safe_overlap_negative_promotion",
        "completed_source_safe_overlap_negative_five_seed",
    }
    or receipt.get("execution_passed") is not True
    or receipt.get("five_seed_promoted") is not False
    or receipt.get("scientific_claim_allowed") is not False
    or int(receipt.get("test_rows_read", -1)) != 0
    or receipt.get("code_commit") != failed_commit
):
    raise SystemExit("glitch-adapter fallback requires the corrected negative head chain")
head_reports = []
for arm in ("uniform_report", "family_balanced_report"):
    entry = receipt.get("inputs", {}).get(arm, {})
    path = pathlib.Path(str(entry.get("path", "")))
    if not path.is_file() or entry.get("sha256") != digest(path):
        raise SystemExit(f"negative head receipt does not bind {arm}")
    report = json.loads(path.read_text(encoding="utf-8"))
    scope = report.get("training_scope", {})
    if (
        report.get("status") != "validation_selected_real_glitch_overlap_finetune"
        or report.get("checkpoint_selection_metric") != "validation_loss"
        or scope.get("scope") != "glitch_head_only"
        or scope.get("non_glitch_state_preserved_bit_exact") is not True
        or report.get("scientific_claim_allowed") is not False
    ):
        raise SystemExit(f"{arm} is not a corrected bit-exact negative head run")
    head_reports.append(
        {
            "arm": arm,
            "path": str(path.resolve()),
            "sha256": digest(path),
            "calibrated_glitch_iou": float(
                report["calibrated_overlap_validation"]["glitch"]["iou"]
            ),
            "best_epoch": int(report["best_epoch"]),
        }
    )
settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
    "overlap_training"
]
expected = {
    "training_scope": "glitch_adapter_only",
    "glitch_adapter_channels": 16,
    "checkpoint_selection_metric": "validation_loss",
    "epochs": 20,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "minimum_clean_chirp_iou_retention": 0.95,
}
for key, value in expected.items():
    if settings.get(key) != value:
        raise SystemExit(f"glitch-adapter fallback policy changed: {key}")
if (
    float(settings.get("clean_chirp_distillation_weight", -1)) != 0.0
    or float(settings.get("clean_chirp_weight", -1)) != 0.25
    or float(settings.get("clean_glitch_distillation_weight", -1)) != 0.25
):
    raise SystemExit("glitch-adapter clean-data policy changed")
result = {
    "status": "authorized_validation_only_glitch_adapter_overlap_fallback",
    "passed": True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "test_rows_read": 0,
    "test_evaluation": None,
    "trigger": "corrected_glitch_head_chain_failed_promotion",
    "failed_head_chain_receipt": {
        "path": str(receipt_path),
        "sha256": digest(receipt_path),
        "code_commit": failed_commit,
    },
    "failed_head_reports": head_reports,
    "fallback_config": {
        "path": str(config_path),
        "sha256": digest(config_path),
    },
    "frozen_revision": {
        **expected,
        "clean_chirp_weight": 0.25,
        "clean_chirp_distillation_weight": 0.0,
        "clean_glitch_distillation_weight": 0.25,
        "expected_invariant": "base_detector_set_and_chirp_logits_bit_exact",
        "adapter_policy": "zero_initialized_residual_glitch_decoder_v1",
    },
    "code_commit": fallback_commit,
}
part = output.with_suffix(output.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, output)
PY
fi

adapter_root="$OUTPUT_ROOT/adapter-seed$SEED"
cache_args=()
if [[ -n "$CLEAN_VALIDATION_FEATURE_CACHE_DIR" ]]; then
  cache_args=(
    --clean-validation-feature-cache-dir
    "$CLEAN_VALIDATION_FEATURE_CACHE_DIR"
  )
fi
exec env PYTHONPATH="$TASK_CODE_DIR/src" GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-finetune \
  --config "$config" \
  --overlap-train-manifest \
  "$train_overlap/physical_overlap_train_manifest.jsonl" \
  --overlap-validation-manifest \
  "$validation_overlap/physical_overlap_val_manifest.jsonl" \
  --clean-train-manifest "$CLEAN_TRAIN_MANIFEST" \
  --clean-validation-manifest "$CLEAN_VALIDATION_MANIFEST" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  "${cache_args[@]}" \
  --output-dir "$adapter_root" \
  --seed "$SEED"
