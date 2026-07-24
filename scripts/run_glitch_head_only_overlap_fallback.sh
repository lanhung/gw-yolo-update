#!/usr/bin/env bash
set -euo pipefail

# Validation-only structural fallback after teacher-anchored optimization fails.
# The backbone and chirp rows remain bit-exact; only glitch head rows can update.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FAILED_TEACHER_CHAIN_ROOT
  TRAIN_GLITCH_MANIFEST
  VALIDATION_GLITCH_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required glitch-head fallback variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$TRAIN_GLITCH_MANIFEST" \
  "$VALIDATION_GLITCH_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT"; do
  if [[ ! -s "$path" ]]; then
    echo "glitch-head fallback input is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$FAILED_TEACHER_CHAIN_ROOT" ]]; then
  echo "failed teacher-anchor chain root is absent" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "glitch-head fallback checkout differs from its declared commit" >&2
  exit 3
fi

teacher_authorization="$FAILED_TEACHER_CHAIN_ROOT/../teacher_anchor_fallback_authorization.json"
uniform_config="$TASK_CODE_DIR/configs/physical_overlap_finetune_glitch_head_only.yaml"
family_config="$TASK_CODE_DIR/configs/physical_overlap_finetune_family_balanced_glitch_head_only.yaml"
for path in "$teacher_authorization" "$uniform_config" "$family_config"; do
  if [[ ! -s "$path" ]]; then
    echo "glitch-head fallback policy input is absent: $path" >&2
    exit 3
  fi
done

mkdir -p "$OUTPUT_ROOT"
authorization="$OUTPUT_ROOT/glitch_head_only_fallback_authorization.json"
if [[ ! -s "$authorization" ]]; then
  "$TASK_PYTHON" - \
    "$FAILED_TEACHER_CHAIN_ROOT" \
    "$teacher_authorization" \
    "$uniform_config" \
    "$family_config" \
    "$authorization" \
    "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

import yaml


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


failed_root = pathlib.Path(sys.argv[1]).resolve()
teacher_authorization_path = pathlib.Path(sys.argv[2]).resolve()
configs = [pathlib.Path(value).resolve() for value in sys.argv[3:5]]
output = pathlib.Path(sys.argv[5])
commit = sys.argv[6]
teacher_authorization = json.loads(
    teacher_authorization_path.read_text(encoding="utf-8")
)
if (
    teacher_authorization.get("status")
    != "authorized_validation_only_teacher_anchored_overlap_fallback"
    or teacher_authorization.get("passed") is not True
    or teacher_authorization.get("scientific_claim_allowed") is not False
    or int(teacher_authorization.get("test_rows_read", -1)) != 0
):
    raise SystemExit("teacher-anchor authorization replay failed")
source_receipt_path = failed_root / "source_safe_overlap_chain_receipt.json"
source_receipt = (
    json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if source_receipt_path.is_file()
    else None
)
negative_receipt = bool(
    source_receipt
    and source_receipt.get("status")
    in {
        "completed_source_safe_overlap_negative_promotion",
        "completed_source_safe_overlap_negative_five_seed",
    }
    and source_receipt.get("execution_passed") is True
    and source_receipt.get("five_seed_promoted") is False
    and source_receipt.get("scientific_claim_allowed") is False
    and int(source_receipt.get("test_rows_read", -1)) == 0
)
failed_histories = []
for history_path in sorted(failed_root.glob("*-seed*/history.json")):
    rows = json.loads(history_path.read_text(encoding="utf-8"))
    if (
        isinstance(rows, list)
        and len(rows) == 20
        and int(rows[-1].get("epoch", -1)) == 20
        and not any(row.get("checkpoint_eligible") is True for row in rows)
    ):
        failed_histories.append(
            {
                "path": str(history_path.resolve()),
                "sha256": digest(history_path),
                "epochs": len(rows),
                "maximum_clean_chirp_iou_retention": max(
                    float(row["clean_chirp_iou_retention"]) for row in rows
                ),
            }
        )
if not negative_receipt and not failed_histories:
    raise SystemExit(
        "glitch-head fallback requires a completed negative teacher-anchor "
        "chain or 20-epoch clean-retention failure"
    )
settings = [
    yaml.safe_load(path.read_text(encoding="utf-8"))["overlap_training"]
    for path in configs
]
for value in settings:
    if (
        value.get("training_scope") != "glitch_head_only"
        or value.get("checkpoint_selection_metric") != "validation_loss"
        or float(value.get("clean_chirp_distillation_weight", -1)) != 0.0
        or float(value.get("clean_chirp_weight", -1)) != 0.25
        or float(value.get("clean_glitch_distillation_weight", -1)) != 0.25
        or float(value.get("learning_rate", -1)) != 0.0001
        or float(value.get("weight_decay", -1)) != 0.0
        or float(value.get("minimum_clean_chirp_iou_retention", -1)) != 0.95
        or int(value.get("epochs", -1)) != 20
    ):
        raise SystemExit("glitch-head-only fallback policy changed")
result = {
    "status": "authorized_validation_only_glitch_head_only_overlap_fallback",
    "passed": True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "test_rows_read": 0,
    "test_evaluation": None,
    "trigger": (
        "negative_teacher_anchor_chain"
        if negative_receipt
        else "complete_teacher_anchor_clean_retention_failure"
    ),
    "teacher_anchor_authorization": {
        "path": str(teacher_authorization_path),
        "sha256": digest(teacher_authorization_path),
    },
    "teacher_chain_receipt": (
        {
            "path": str(source_receipt_path.resolve()),
            "sha256": digest(source_receipt_path),
        }
        if source_receipt_path.is_file()
        else None
    ),
    "failed_histories": failed_histories,
    "fallback_configs": [
        {"path": str(path), "sha256": digest(path)} for path in configs
    ],
    "frozen_revision": {
        "training_scope": "glitch_head_only",
        "checkpoint_selection_metric": "validation_loss",
        "learning_rate": 0.0001,
        "weight_decay": 0.0,
        "clean_chirp_weight": 0.25,
        "clean_chirp_distillation_weight": 0.0,
        "clean_glitch_distillation_weight": 0.25,
        "minimum_clean_chirp_iou_retention": 0.95,
        "epochs": 20,
        "expected_invariant": "backbone_and_chirp_head_bit_exact",
    },
    "code_commit": commit,
}
part = output.with_suffix(output.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, output)
PY
fi

chain_root="$OUTPUT_ROOT/chain"
mkdir -p "$chain_root"
for split_root in train-overlaps val-overlaps; do
  source_root="$FAILED_TEACHER_CHAIN_ROOT/$split_root"
  target_root="$chain_root/$split_root"
  if [[ ! -s "$source_root/physical_overlap_report.json" ]]; then
    echo "glitch-head fallback cannot replay $split_root" >&2
    exit 3
  fi
  if [[ ! -e "$target_root" ]]; then
    ln -s "$source_root" "$target_root"
  elif [[ "$(readlink -f "$target_root")" != "$(readlink -f "$source_root")" ]]; then
    echo "glitch-head fallback overlap replay points elsewhere" >&2
    exit 3
  fi
done

exec env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  TRAIN_GLITCH_MANIFEST="$TRAIN_GLITCH_MANIFEST" \
  VALIDATION_GLITCH_MANIFEST="$VALIDATION_GLITCH_MANIFEST" \
  GRAVITYSPY_CORPUS_AUDIT="$GRAVITYSPY_CORPUS_AUDIT" \
  CLEAN_TRAIN_MANIFEST="$CLEAN_TRAIN_MANIFEST" \
  CLEAN_VALIDATION_MANIFEST="$CLEAN_VALIDATION_MANIFEST" \
  PRETRAINED_CHECKPOINT="$PRETRAINED_CHECKPOINT" \
  UNIFORM_CONFIG="$uniform_config" \
  FAMILY_BALANCED_CONFIG="$family_config" \
  OUTPUT_ROOT="$chain_root" \
  bash "$TASK_CODE_DIR/scripts/run_source_safe_overlap_publication.sh"
