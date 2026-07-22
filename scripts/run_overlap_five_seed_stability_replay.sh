#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_FIVE_SEED_SUMMARY
  FIVE_SEED_STABILITY_CONFIG
  OUTPUT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$SOURCE_FIVE_SEED_SUMMARY" \
  "$FIVE_SEED_STABILITY_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "required five-seed replay input is absent: $path" >&2
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
if [[ -e "$OUTPUT" ]]; then
  echo "five-seed stability replay output already exists: $OUTPUT" >&2
  exit 4
fi

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli physical-overlap-five-seed-stability-replay \
  --source-summary "$SOURCE_FIVE_SEED_SUMMARY" \
  --stability-config "$FIVE_SEED_STABILITY_CONFIG" \
  --output "$OUTPUT"

"$TASK_PYTHON" - "$OUTPUT" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = report.get("five_seed_stability", {})
if (
    report.get("status") != "completed_five_seed_source_safe_overlap_validation"
    or audit.get("status") != "five_seed_reproducibility_gate_v1"
    or audit.get("passed") is not report.get("passed")
    or report.get("test_data_opened") is not False
    or report.get("scientific_claim_allowed") is not False
):
    raise SystemExit("five-seed stability replay failed closed")
print(json.dumps(report, indent=2, sort_keys=True))
PY
