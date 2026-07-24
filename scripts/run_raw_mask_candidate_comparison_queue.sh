#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  UPSTREAM_RECEIPT
  UPSTREAM_PID
  UPSTREAM_IDENTITY
  RAW_CALIBRATION_REPORT
  MASK_CALIBRATION_REPORT
  MASK_VALIDATION_RECEIPT
  MASK_TIMING_RECEIPT
  OUTPUT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required raw/mask comparison queue variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "raw/mask comparison queue requires its declared immutable checkout" >&2
  exit 2
fi

while [[ ! -s "$UPSTREAM_RECEIPT" ]]; do
  if [[ ! -r "/proc/$UPSTREAM_PID/cmdline" ]] \
    || ! tr '\0' ' ' <"/proc/$UPSTREAM_PID/cmdline" | grep -Fq "$UPSTREAM_IDENTITY"; then
    echo "raw/mask background upstream ended without its receipt: $UPSTREAM_RECEIPT" >&2
    exit 1
  fi
  sleep 30
done
for path in \
  "$TASK_PYTHON" \
  "$UPSTREAM_RECEIPT" \
  "$RAW_CALIBRATION_REPORT" \
  "$MASK_CALIBRATION_REPORT" \
  "$MASK_VALIDATION_RECEIPT" \
  "$MASK_TIMING_RECEIPT"; do
  if [[ ! -s "$path" ]]; then
    echo "raw/mask comparison queue input is absent: $path" >&2
    exit 2
  fi
done

"$TASK_PYTHON" - \
  "$UPSTREAM_RECEIPT" \
  "$RAW_CALIBRATION_REPORT" \
  "$MASK_CALIBRATION_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


receipt_path, raw_path, mask_path = map(pathlib.Path, sys.argv[1:4])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("status") != "completed_validation_only_raw_mask_candidate_background"
    or receipt.get("scientific_claim_allowed") is not False
    or receipt.get("locked_test_open_allowed") is not False
    or receipt.get("locked_test_prerequisites_satisfied") is not False
    or receipt.get("test_rows_read") != 0
):
    raise SystemExit("raw/mask background receipt failed queue replay")
for arm, path in (("raw", raw_path), ("mask", mask_path)):
    identity = receipt.get("calibrations", {}).get(arm, {})
    if pathlib.Path(str(identity.get("path", ""))).resolve() != path.resolve():
        raise SystemExit(f"{arm} calibration path differs from upstream receipt")
    if identity.get("sha256") != digest(path):
        raise SystemExit(f"{arm} calibration hash differs from upstream receipt")
PY

if [[ ! -e "$OUTPUT" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-compare \
      --raw-calibration-report "$RAW_CALIBRATION_REPORT" \
      --mask-calibration-report "$MASK_CALIBRATION_REPORT" \
      --mask-validation-receipt "$MASK_VALIDATION_RECEIPT" \
      --mask-timing-receipt "$MASK_TIMING_RECEIPT" \
      --minimum-absolute-weighted-efficiency-gain \
        "${MINIMUM_MASK_EFFICIENCY_GAIN:-0.05}" \
      --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-10000}" \
      --seed "${BOOTSTRAP_SEED:-20260720}" \
      --output "$OUTPUT"
  )
fi

"$TASK_PYTHON" - "$OUTPUT" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import pathlib
import sys


output = pathlib.Path(sys.argv[1])
code_commit = sys.argv[2]
report = json.loads(output.read_text(encoding="utf-8"))
if (
    report.get("status")
    != "validation_only_paired_raw_mask_candidate_calibration_comparison"
    or report.get("scientific_claim_allowed") is not False
    or report.get("locked_test_allowed") is not False
    or report.get("locked_test_prerequisites_satisfied") is not False
    or report.get("test_rows_read") != 0
    or report.get("code_commit") != code_commit
):
    raise SystemExit("raw/mask paired comparison failed queue replay")
PY
