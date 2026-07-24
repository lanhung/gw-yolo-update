#!/usr/bin/env bash
set -euo pipefail

# Bind the seven no-refit calibration stresses to the newly materialized
# four-subset injection corpus. The perturbation plan is frozen only after the
# physical H1/L1/V1 tensor audit and the promoted validation calibration exist.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  REFERENCE_CODE_DIR
  REFERENCE_CODE_COMMIT
  FIVE_SEED_SUMMARY
  PROMOTED_BLOCK_PIPELINE_REPORT
  BACKGROUND_MANIFEST
  PHYSICAL_MATERIALIZATION_AUDIT
  INJECTION_MANIFEST
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
  NETWORK_CONFIG
  TIMING_CALIBRATION_REPORT
  BLOCK_SCHEDULE
  BASELINE_CALIBRATION_REPORT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector calibration-successor variable is unset: $variable" >&2
    exit 2
  fi
done
adapter_config=${ADAPTER_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_finetune_glitch_adapter.yaml}
for path in \
  "$TASK_PYTHON" \
  "$BACKGROUND_MANIFEST" \
  "$PHYSICAL_MATERIALIZATION_AUDIT" \
  "$INJECTION_MANIFEST" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$COHERENCE_CONFIG" \
  "$NETWORK_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "detector calibration-successor input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector calibration successor requires its exact checkout" >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT"
calibration_plan="$OUTPUT_ROOT/detector_stratified_calibration_perturbation_plan.json"
robustness_config="$TASK_CODE_DIR/configs/calibration_perturbation_o4a_validation.yaml"
"$TASK_PYTHON" - "$PHYSICAL_MATERIALIZATION_AUDIT" "$INJECTION_MANIFEST" \
  "$BACKGROUND_MANIFEST" "$robustness_config" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


audit_path, injection_path, background_path, config_path = sys.argv[1:]
audit = json.loads(pathlib.Path(audit_path).read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in pathlib.Path(injection_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
background = [
    json.loads(line)
    for line in pathlib.Path(background_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
required = {"H1+L1", "H1+V1", "L1+V1", "H1+L1+V1"}
counts = {}
for row in rows:
    subset = "+".join(row.get("ifos", []))
    counts[subset] = counts.get(subset, 0) + 1
if (
    audit.get("status")
    != "verified_detector_stratified_physical_injection_materialization"
    or audit.get("passed") is not True
    or audit.get("publication_calibration_eligible") is not True
    or audit.get("manifest_sha256") != digest(injection_path)
    or audit.get("candidate_scores_inspected") is not False
    or int(audit.get("test_rows_read", -1)) != 0
    or set(counts) != required
    or any(counts[subset] < 25 for subset in required)
    or {row["gps_block"] for row in rows}
    & {row["gps_block"] for row in background}
    or not pathlib.Path(config_path).is_file()
):
    raise SystemExit("detector-stratified calibration preflight failed")
PY

if [[ ! -s "$calibration_plan" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli calibration-perturbation-plan-freeze \
      --background-manifest "$BACKGROUND_MANIFEST" \
      --injection-manifest "$INJECTION_MANIFEST" \
      --config "$robustness_config" \
      --output "$calibration_plan"
  )
fi

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  REFERENCE_CODE_DIR="$REFERENCE_CODE_DIR" \
  REFERENCE_CODE_COMMIT="$REFERENCE_CODE_COMMIT" \
  FIVE_SEED_SUMMARY="$FIVE_SEED_SUMMARY" \
  PROMOTED_BLOCK_PIPELINE_REPORT="$PROMOTED_BLOCK_PIPELINE_REPORT" \
  CALIBRATION_PLAN="$calibration_plan" \
  BACKGROUND_MANIFEST="$BACKGROUND_MANIFEST" \
  INJECTION_MANIFEST="$INJECTION_MANIFEST" \
  UNIFORM_CONFIG="$UNIFORM_CONFIG" \
  FAMILY_BALANCED_CONFIG="$FAMILY_BALANCED_CONFIG" \
  ADAPTER_CONFIG="$adapter_config" \
  COHERENCE_CONFIG="$COHERENCE_CONFIG" \
  NETWORK_CONFIG="$NETWORK_CONFIG" \
  TIMING_CALIBRATION_REPORT="$TIMING_CALIBRATION_REPORT" \
  BLOCK_SCHEDULE="$BLOCK_SCHEDULE" \
  BASELINE_CALIBRATION_REPORT="$BASELINE_CALIBRATION_REPORT" \
  OUTPUT_ROOT="$OUTPUT_ROOT/scenarios" \
  bash "$TASK_CODE_DIR/scripts/queue_calibration_robustness_validation.sh"
