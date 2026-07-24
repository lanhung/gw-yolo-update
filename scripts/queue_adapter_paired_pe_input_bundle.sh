#!/usr/bin/env bash
set -euo pipefail

# Wait for the validation-only adapter decision, materialize a publication-size
# paired PE input set only after a positive five-seed gate, and freeze a
# content-addressed cross-machine bundle. A negative decision is a valid,
# machine-readable terminal result and never opens test data.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  ADAPTER_CONFIG
  MODEL_SELECTION_TRAIN_OVERLAP_MANIFEST
  MODEL_SELECTION_VALIDATION_OVERLAP_MANIFEST
  MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  INDEPENDENT_PE_OVERLAP_REPORT
  INDEPENDENT_OVERLAP_AUDIT
  OVERLAP_MANIFEST
  INJECTION_MANIFEST
  OUTPUT_ROOT
  BUNDLE_ROOT
  QUEUE_RECEIPT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
observed_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi

while [[ ! -s "$FIVE_SEED_SUMMARY" ]]; do
  if [[ -n "${UPSTREAM_PID:-}" ]] && ! kill -0 "$UPSTREAM_PID" 2>/dev/null; then
    mkdir -p "$(dirname "$QUEUE_RECEIPT")"
    "$TASK_PYTHON" - "$QUEUE_RECEIPT" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
result = {
    "status": "adapter_paired_pe_input_queue_not_authorized",
    "passed": False,
    "scientific_claim_allowed": False,
    "scientific_blocker": "upstream ended without a five-seed validation summary",
    "test_rows_read": 0,
    "code_commit": sys.argv[2],
}
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
    exit 0
  fi
  sleep 30
done

decision=$(
  "$TASK_PYTHON" - "$FIVE_SEED_SUMMARY" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
positive = (
    value.get("status") == "completed_five_seed_source_safe_overlap_validation"
    and value.get("passed") is True
    and value.get("promoted_arm") == "glitch_adapter"
    and value.get("five_seed_stability", {}).get("passed") is True
    and value.get("test_data_opened") is False
)
print("positive" if positive else "negative")
PY
)
if [[ "$decision" != positive ]]; then
  mkdir -p "$(dirname "$QUEUE_RECEIPT")"
  "$TASK_PYTHON" - "$QUEUE_RECEIPT" "$FIVE_SEED_SUMMARY" \
    "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2]).resolve()
result = {
    "status": "adapter_paired_pe_input_queue_negative_validation",
    "passed": False,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "the adapter did not pass its five-seed validation gate; paired PE input "
        "materialization and cross-machine transfer were not authorized"
    ),
    "test_rows_read": 0,
    "five_seed_summary_path": str(summary),
    "five_seed_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
    "code_commit": sys.argv[3],
}
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
  exit 0
fi

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  FIVE_SEED_SUMMARY="$FIVE_SEED_SUMMARY" \
  UNIFORM_CONFIG="$UNIFORM_CONFIG" \
  FAMILY_BALANCED_CONFIG="$FAMILY_BALANCED_CONFIG" \
  ADAPTER_CONFIG="$ADAPTER_CONFIG" \
  MODEL_SELECTION_TRAIN_OVERLAP_MANIFEST="$MODEL_SELECTION_TRAIN_OVERLAP_MANIFEST" \
  MODEL_SELECTION_VALIDATION_OVERLAP_MANIFEST="$MODEL_SELECTION_VALIDATION_OVERLAP_MANIFEST" \
  MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST="$MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST" \
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  INDEPENDENT_PE_OVERLAP_REPORT="$INDEPENDENT_PE_OVERLAP_REPORT" \
  INDEPENDENT_OVERLAP_AUDIT="$INDEPENDENT_OVERLAP_AUDIT" \
  OVERLAP_MANIFEST="$OVERLAP_MANIFEST" \
  INJECTION_MANIFEST="$INJECTION_MANIFEST" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  PE_SMOKE_LIMIT="${PE_VALIDATION_LIMIT:-100}" \
  PE_SELECTION_SEED="${PE_SELECTION_SEED:-20260722}" \
  GWYOLO_PE_MINIMUM_GPS_BLOCKS="${PE_MINIMUM_GPS_BLOCKS:-25}" \
  bash "$TASK_CODE_DIR/scripts/run_promoted_paired_pe_smoke.sh"

export PYTHONPATH="$TASK_CODE_DIR/src"
export GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli pe-input-bundle-export \
  --summary "$OUTPUT_ROOT/paired_pe_smoke_summary.json" \
  --output-dir "$BUNDLE_ROOT"

"$TASK_PYTHON" - "$QUEUE_RECEIPT" "$FIVE_SEED_SUMMARY" \
  "$OUTPUT_ROOT/paired_pe_smoke_summary.json" \
  "$BUNDLE_ROOT/paired_pe_input_bundle.json" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
labels = ("five_seed_summary", "paired_pe_input_summary", "input_bundle")
paths = [pathlib.Path(value).resolve() for value in sys.argv[2:5]]
loaded = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
if (
    loaded[0].get("test_data_opened") is not False
    or loaded[1].get("test_rows_read") != 0
    or loaded[2].get("test_rows_read") != 0
    or loaded[2].get("status") != "portable_paired_pe_input_bundle"
    or loaded[2].get("passed") is not True
):
    raise SystemExit("adapter paired PE bundle crossed its validation-only boundary")
artifacts = {
    label: {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for label, path in zip(labels, paths)
}
result = {
    "status": "adapter_paired_pe_input_bundle_ready",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "real within-backend DINGO/AMPLFI posteriors and a matched-event "
        "validation portfolio remain required"
    ),
    "required_split": "val",
    "test_rows_read": 0,
    "paired_injections": loaded[1]["paired_injections"],
    "evaluation_tier": loaded[1]["evaluation_tier"],
    "artifacts": artifacts,
    "code_commit": sys.argv[5],
}
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
