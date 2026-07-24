#!/usr/bin/env bash
set -euo pipefail

# Turn the frozen four-subset recipes into actual antenna-projected
# H1/L1/V1 signals. This stage cannot run until the score-blind background
# chain is complete and does not fit thresholds or inspect test data.

required=(
  TASK_PYTHON
  WAVEFORM_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  DETECTOR_VALIDATION_DATA_ROOT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector physical-materialization variable is unset: $variable" >&2
    exit 2
  fi
done
for path in "$TASK_PYTHON" "$WAVEFORM_PYTHON"; do
  if [[ ! -x "$path" ]]; then
    echo "required detector physical-materialization runtime is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector physical materialization requires its exact checkout" >&2
  exit 3
fi
if ! "$WAVEFORM_PYTHON" -c 'import lal; import lalsimulation; import pycbc'; then
  echo "WAVEFORM_PYTHON requires LALSuite and PyCBC" >&2
  exit 3
fi

sample_rate=${SAMPLE_RATE:-1024}
context_duration=${CONTEXT_DURATION_SECONDS:-}
waveform_cases_per_family=${WAVEFORM_CASES_PER_FAMILY:-10}
maximum_attempts=${MAXIMUM_MATERIALIZATION_ATTEMPTS:-3}
retry_delay_seconds=${RETRY_DELAY_SECONDS:-60}
data_receipt="$DETECTOR_VALIDATION_DATA_ROOT/detector_validation_streaming_receipt.json"
background_manifest="$DETECTOR_VALIDATION_DATA_ROOT/merged/background_windows.jsonl"
background_report="$DETECTOR_VALIDATION_DATA_ROOT/merged/detector_validation_background_report.json"
injection_plan="$DETECTOR_VALIDATION_DATA_ROOT/injections/detector_stratified_injection_plan.json"
recipes="$DETECTOR_VALIDATION_DATA_ROOT/injections/detector_stratified_injection_recipes.jsonl"
for path in \
  "$data_receipt" \
  "$background_manifest" \
  "$background_report" \
  "$injection_plan" \
  "$recipes"; do
  if [[ ! -s "$path" ]]; then
    echo "detector physical-materialization input is absent: $path" >&2
    exit 3
  fi
done

mkdir -p "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
manifest_context_duration=$(
  "$TASK_PYTHON" - "$data_receipt" "$background_report" "$injection_plan" \
    "$background_manifest" "$recipes" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


receipt_path, background_path, plan_path, manifest_path, recipes_path = sys.argv[1:]
receipt = json.loads(pathlib.Path(receipt_path).read_text(encoding="utf-8"))
background = json.loads(pathlib.Path(background_path).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
if (
    receipt.get("status")
    != "verified_detector_stratified_validation_data_chain"
    or receipt.get("passed") is not True
    or receipt.get("background_report_sha256") != digest(background_path)
    or receipt.get("injection_plan_sha256") != digest(plan_path)
    or receipt.get("candidate_scores_inspected") is not False
    or int(receipt.get("test_rows_read", -1)) != 0
    or background.get("publication_calibration_eligible") is not True
    or background.get("manifest_sha256") != digest(manifest_path)
    or plan.get("status")
    != "frozen_detector_stratified_validation_injection_plan"
    or plan.get("passed") is not True
    or plan.get("manifest_sha256") != digest(recipes_path)
    or plan.get("candidate_scores_inspected") is not False
    or int(plan.get("test_rows_read", -1)) != 0
):
    raise SystemExit("detector physical-materialization preflight failed replay")
rows = [
    json.loads(line)
    for line in pathlib.Path(manifest_path).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
durations = set()
rows_without_banks = 0
for row in rows:
    bank = row.get("background_bank")
    if not isinstance(bank, dict):
        rows_without_banks += 1
        continue
    with np.load(bank["path"], allow_pickle=False) as arrays:
        noise = arrays["noise"]
        sample_rate = int(arrays["sample_rate"])
        if noise.ndim != 2 or sample_rate <= 0:
            raise SystemExit("detector background bank has invalid context shape")
        durations.add(noise.shape[1] / sample_rate)
if (
    not rows
    or rows_without_banks
    or len(durations) != 1
    or not all(math.isfinite(value) and value > 0 for value in durations)
):
    raise SystemExit(
        "detector background banks require one positive context duration"
    )
print(next(iter(durations)))
PY
)
if [[ -z "$context_duration" ]]; then
  context_duration=$manifest_context_duration
else
  "$TASK_PYTHON" - "$context_duration" "$manifest_context_duration" <<'PY'
import math
import sys

configured, manifest = map(float, sys.argv[1:])
if (
    not math.isfinite(configured)
    or configured <= 0
    or abs(configured - manifest) > 1e-9
):
    raise SystemExit(
        "configured context duration differs from detector background manifest"
    )
PY
fi

waveform_report="$OUTPUT_ROOT/waveform_validation_report.json"
if [[ ! -s "$waveform_report" ]]; then
  "$WAVEFORM_PYTHON" -m gwyolo.cli waveform-validate \
    --recipes "$recipes" \
    --output "$waveform_report" \
    --sample-rate "$sample_rate" \
    --reference-duration 128 \
    --per-family "$waveform_cases_per_family"
fi
"$TASK_PYTHON" - "$waveform_report" "$recipes" "$sample_rate" \
  "$waveform_cases_per_family" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, recipe_path, sample_rate, per_family = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
if (
    report.get("passed") is not True
    or report.get("validation_scope")
    != "external_reference_waveform_equivalence"
    or report.get("recipe_manifest_sha256")
    != hashlib.sha256(pathlib.Path(recipe_path).read_bytes()).hexdigest()
    or int(report.get("sample_rate", -1)) != int(sample_rate)
    or int(report.get("per_family", -1)) != int(per_family)
    or any(row.get("passed") is not True for row in report.get("cases", []))
):
    raise SystemExit("detector waveform-equivalence gate failed")
PY

materialized_root="$OUTPUT_ROOT/materialized"
materialization_report="$materialized_root/materialization_report.json"
materialized_manifest="$materialized_root/materialized_injections.jsonl"
if [[ ! -s "$materialization_report" ]]; then
  completed=0
  for ((attempt = 1; attempt <= maximum_attempts; attempt++)); do
    if "$WAVEFORM_PYTHON" -m gwyolo.cli injection-materialize \
      --recipes "$recipes" \
      --background-manifest "$background_manifest" \
      --output-dir "$materialized_root" \
      --sample-rate "$sample_rate" \
      --context-duration "$context_duration" \
      --storage-mode signal_scaled_float16 \
      --split val \
      --backend-validation-report "$waveform_report"; then
      completed=1
      break
    fi
    if (( attempt < maximum_attempts )); then
      sleep "$retry_delay_seconds"
    fi
  done
  if (( completed != 1 )); then
    echo "detector physical materialization exhausted bounded retries" >&2
    exit 1
  fi
fi

snr_root="$OUTPUT_ROOT/snr"
snr_report="$snr_root/snr_annotation_report.json"
snr_manifest="$snr_root/materialized_injections_snr.jsonl"
if [[ ! -s "$snr_report" ]]; then
  "$WAVEFORM_PYTHON" -m gwyolo.cli injection-snr-annotate \
    --manifest "$materialized_manifest" \
    --output-dir "$snr_root" \
    --low-frequency 20 \
    --high-frequency 500 \
    --psd-segment-seconds 8 \
    --psd-stride-seconds 4
fi

arrival_root="$OUTPUT_ROOT/arrivals"
arrival_report="$arrival_root/detector_arrival_annotation_report.json"
if [[ ! -s "$arrival_report" ]]; then
  "$WAVEFORM_PYTHON" -m gwyolo.cli injection-arrival-annotate \
    --manifest "$snr_manifest" \
    --output-dir "$arrival_root"
fi

"$TASK_PYTHON" -m gwyolo.cli detector-validation-materialization-audit \
  --injection-plan "$injection_plan" \
  --materialization-report "$materialization_report" \
  --snr-report "$snr_report" \
  --arrival-report "$arrival_report" \
  --output "$OUTPUT_ROOT/detector_stratified_physical_materialization_audit.json"
