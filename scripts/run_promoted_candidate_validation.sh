#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  FIVE_SEED_SUMMARY
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
  OUTPUT_ROOT
  GWYOLO_CODE_COMMIT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$COHERENCE_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done

if ! selection_output=$("$TASK_PYTHON" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "completed_five_seed_source_safe_overlap_validation":
    raise SystemExit("five-seed summary has the wrong status")
print(report["promoted_arm"])
print(report["selected_checkpoint_path"])
' "$FIVE_SEED_SUMMARY"); then
  echo "failed to resolve checkpoint/config from five-seed summary" >&2
  exit 2
fi
readarray -t selection <<<"$selection_output"
if (( ${#selection[@]} != 2 )) || [[ -z "${selection[0]}" || -z "${selection[1]}" ]]; then
  echo "five-seed summary did not return exactly one arm and checkpoint" >&2
  exit 2
fi
arm=${selection[0]}
checkpoint=${selection[1]}
if [[ "$arm" == uniform ]]; then
  config=$UNIFORM_CONFIG
elif [[ "$arm" == family_balanced ]]; then
  config=$FAMILY_BALANCED_CONFIG
else
  echo "five-seed summary selected an unknown arm: $arm" >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "selected checkpoint is absent: $checkpoint" >&2
  exit 2
fi

"$TASK_PYTHON" - "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$BACKGROUND_MANIFEST" "$INJECTION_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys

endpoint_path, background_path, injection_path = sys.argv[1:]
endpoint = json.loads(pathlib.Path(endpoint_path).read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
components = endpoint.get("component_reports", {})
expected_components = {
    "purpose_partition",
    "injection_plan",
    "waveform_validation",
    "materialization",
    "snr_annotation",
    "arrival_annotation",
}
if (
    endpoint.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or not endpoint.get("passed")
    or endpoint.get("test_rows_read") != 0
    or endpoint.get("test_evaluation") is not None
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or pathlib.Path(endpoint["candidate_calibration_background_manifest_path"]).resolve()
    != pathlib.Path(background_path).resolve()
    or endpoint.get("candidate_calibration_background_manifest_sha256")
    != digest(background_path)
    or pathlib.Path(endpoint["injection_arrival_manifest_path"]).resolve()
    != pathlib.Path(injection_path).resolve()
    or endpoint.get("injection_arrival_manifest_sha256") != digest(injection_path)
    or set(components) != expected_components
    or any(
        digest(item["path"]) != item["sha256"]
        for item in components.values()
    )
):
    raise SystemExit("candidate validation inputs do not match the frozen independent endpoint")
PY

mkdir -p "$OUTPUT_ROOT"
"$TASK_PYTHON" -m gwyolo.cli manifest-select-split \
  --manifest "$BACKGROUND_MANIFEST" \
  --split val \
  --output-dir "$OUTPUT_ROOT/background-val"
while :; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$gpu_pids" ]] && break
  sleep 30
done
"$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-pipeline \
  --background-manifest "$OUTPUT_ROOT/background-val/val_manifest.jsonl" \
  --injection-manifest "$INJECTION_MANIFEST" \
  --checkpoint "$checkpoint" \
  --config "$config" \
  --coherence-config "$COHERENCE_CONFIG" \
  --output-dir "$OUTPUT_ROOT/pipeline" \
  --chirp-threshold 0.3 \
  --minimum-bins 1 \
  --timing-association-window-seconds 0.25 \
  --timing-uncertainty-quantile 0.99 \
  --minimum-timing-matches 30 \
  --maximum-timing-uncertainty-seconds 0.01 \
  --slide-count 512 \
  --slide-step-seconds 8 \
  --target-far-per-year 100 \
  --bootstrap-replicates 10000 \
  --seed 20260720 \
  --model-selection-report "$FIVE_SEED_SUMMARY"
