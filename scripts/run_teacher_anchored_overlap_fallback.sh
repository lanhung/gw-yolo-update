#!/usr/bin/env bash
set -euo pipefail

# A validation-only optimization fallback for catastrophic clean-mask
# forgetting. It is authorized only after the preceding source-safe chain has
# completed negatively or a full arm has no retention-eligible checkpoint.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FAILED_CHAIN_ROOT
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
    echo "required teacher-anchor fallback variable is unset: $variable" >&2
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
    echo "teacher-anchor fallback input is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$FAILED_CHAIN_ROOT" ]]; then
  echo "failed source-safe chain root is absent" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "teacher-anchor fallback checkout differs from its declared commit" >&2
  exit 3
fi

uniform_config="$TASK_CODE_DIR/configs/physical_overlap_finetune_teacher_anchor.yaml"
family_config="$TASK_CODE_DIR/configs/physical_overlap_finetune_family_balanced_teacher_anchor.yaml"
for path in "$uniform_config" "$family_config"; do
  if [[ ! -s "$path" ]]; then
    echo "teacher-anchor fallback config is absent: $path" >&2
    exit 3
  fi
done
mkdir -p "$OUTPUT_ROOT"
authorization="$OUTPUT_ROOT/teacher_anchor_fallback_authorization.json"
if [[ ! -s "$authorization" ]]; then
  "$TASK_PYTHON" - \
    "$FAILED_CHAIN_ROOT" \
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
configs = [pathlib.Path(value).resolve() for value in sys.argv[2:4]]
output = pathlib.Path(sys.argv[4])
commit = sys.argv[5]
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
        "teacher-anchor fallback requires a completed negative chain or "
        "20-epoch clean-retention failure"
    )
settings = [
    yaml.safe_load(path.read_text(encoding="utf-8"))["overlap_training"]
    for path in configs
]
for value in settings:
    if (
        float(value.get("clean_chirp_distillation_weight", 0)) != 4.0
        or float(value.get("clean_chirp_weight", -1)) != 0.25
        or float(value.get("learning_rate", -1)) != 0.00001
        or float(value.get("minimum_clean_chirp_iou_retention", -1))
        != 0.95
        or int(value.get("epochs", -1)) != 20
    ):
        raise SystemExit("teacher-anchor fallback policy changed")
result = {
    "status": "authorized_validation_only_teacher_anchored_overlap_fallback",
    "passed": True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "test_rows_read": 0,
    "test_evaluation": None,
    "trigger": (
        "negative_source_safe_chain"
        if negative_receipt
        else "complete_clean_retention_failure"
    ),
    "source_chain_receipt": (
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
        "learning_rate": 0.00001,
        "clean_chirp_weight": 0.25,
        "clean_chirp_distillation_weight": 4.0,
        "minimum_clean_chirp_iou_retention": 0.95,
        "epochs": 20,
    },
    "code_commit": commit,
}
part = output.with_suffix(output.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, output)
PY
fi

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
  OUTPUT_ROOT="$OUTPUT_ROOT/chain" \
  bash "$TASK_CODE_DIR/scripts/run_source_safe_overlap_publication.sh"
