#!/usr/bin/env bash
set -euo pipefail

# Wait for the audited detector-set overlap corpus, then package the exact
# overlap tensors, H1/L1 clean distillation inputs, unique background HDFs,
# checkpoint, configs and evidence graph for relocation to a CUDA worker.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  OVERLAP_RECEIPT
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  FINETUNE_CONFIG
  OVERLAP_CONFIG
  BUNDLE_ROOT
  QUEUE_RECEIPT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector-set bundle variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector-set training bundle requires its exact checkout" >&2
  exit 3
fi

write_incomplete() {
  mkdir -p "$(dirname "$QUEUE_RECEIPT")"
  "$TASK_PYTHON" - "$QUEUE_RECEIPT" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
result = {
    "status": "detector_set_training_bundle_queue_upstream_incomplete",
    "passed": False,
    "scientific_claim_allowed": False,
    "scientific_blocker": "detector-set overlap materialization ended without its final receipt",
    "test_rows_read": 0,
    "test_evaluation": None,
    "code_commit": sys.argv[2],
}
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, target)
PY
}

while [[ ! -s "$OVERLAP_RECEIPT" ]]; do
  if [[ -n "${UPSTREAM_PID:-}" ]] && ! kill -0 "$UPSTREAM_PID" 2>/dev/null; then
    write_incomplete
    exit 0
  fi
  sleep "${QUEUE_POLL_SECONDS:-30}"
done
for path in \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$FINETUNE_CONFIG" \
  "$OVERLAP_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "detector-set bundle input is absent: $path" >&2
    exit 3
  fi
done

cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli detector-set-training-bundle-export \
  --overlap-receipt "$OVERLAP_RECEIPT" \
  --clean-train-manifest "$CLEAN_TRAIN_MANIFEST" \
  --clean-validation-manifest "$CLEAN_VALIDATION_MANIFEST" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --config "finetune=$FINETUNE_CONFIG" \
  --config "overlap_factory=$OVERLAP_CONFIG" \
  --output-dir "$BUNDLE_ROOT"

bundle="$BUNDLE_ROOT/detector_set_training_input_bundle.json"
mkdir -p "$(dirname "$QUEUE_RECEIPT")"
"$TASK_PYTHON" - "$QUEUE_RECEIPT" "$bundle" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
bundle = pathlib.Path(sys.argv[2]).resolve()
loaded = json.loads(bundle.read_text(encoding="utf-8"))
if (
    loaded.get("status") != "portable_detector_set_training_input_bundle"
    or loaded.get("passed") is not True
    or loaded.get("test_rows_read") != 0
    or loaded.get("test_evaluation") is not None
    or loaded.get("detector_complete_clean_training_authorized") is not False
):
    raise SystemExit("detector-set training bundle crossed its scientific boundary")
result = {
    "status": "detector_set_training_bundle_queue_completed",
    "passed": True,
    "scientific_claim_allowed": False,
    "detector_complete_clean_training_authorized": False,
    "test_rows_read": 0,
    "test_evaluation": None,
    "bundle": {
        "path": str(bundle),
        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "object_count": loaded["object_count"],
        "object_bytes": loaded["object_bytes"],
    },
    "code_commit": sys.argv[3],
}
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, target)
PY
