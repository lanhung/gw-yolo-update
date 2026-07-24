#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  ACQUISITION_ROOT
  BASELINE_TRAIN_MANIFEST
  BASELINE_VALIDATION_MANIFEST
  OUTPUT_ROOT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

VALIDATION_COUNT=${VALIDATION_COUNT:-3000}
INJECTION_SEED=${INJECTION_SEED:-20260725}
MINIMUM_UNIQUE_GPS_BLOCKS=${MINIMUM_UNIQUE_GPS_BLOCKS:-50}
MINIMUM_PURPOSE_GPS_BLOCKS=${MINIMUM_PURPOSE_GPS_BLOCKS:-25}
PURPOSE_PARTITION_SEED=${PURPOSE_PARTITION_SEED:-20260725}
WAVEFORM_CASES_PER_FAMILY=${WAVEFORM_CASES_PER_FAMILY:-10}
SAMPLE_RATE=${SAMPLE_RATE:-2048}
CONTEXT_DURATION=${CONTEXT_DURATION:-64}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS:-60}
WAVEFORM_PYTHON=${WAVEFORM_PYTHON:-$TASK_PYTHON}

if ! [[ "$VALIDATION_COUNT" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MINIMUM_UNIQUE_GPS_BLOCKS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MINIMUM_PURPOSE_GPS_BLOCKS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$WAVEFORM_CASES_PER_FAMILY" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "counts and retry settings must be positive bounded integers" >&2
  exit 2
fi
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "code directory is invalid: $TASK_CODE_DIR" >&2
  exit 2
fi

background_manifest="$ACQUISITION_ROOT/merged-background/background_windows.jsonl"
background_report="$ACQUISITION_ROOT/merged-background/background_plan_report.json"
for input in \
  "$TASK_PYTHON" \
  "$WAVEFORM_PYTHON" \
  "$background_manifest" \
  "$background_report" \
  "$BASELINE_TRAIN_MANIFEST" \
  "$BASELINE_VALIDATION_MANIFEST"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done

waveform_runtime_checked=0
require_waveform_runtime() {
  if (( waveform_runtime_checked == 1 )); then
    return
  fi
  if ! "$WAVEFORM_PYTHON" -c 'import lal; import lalsimulation; import pycbc'; then
    echo "WAVEFORM_PYTHON requires LALSuite and PyCBC" >&2
    exit 2
  fi
  waveform_runtime_checked=1
}

mkdir -p "$OUTPUT_ROOT"
disjoint_dir="$OUTPUT_ROOT/disjoint-background"
disjoint_manifest="$disjoint_dir/background_windows.jsonl"
disjoint_report="$disjoint_dir/background_plan_report.json"
if [[ ! -s "$disjoint_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli background-disjoint-subset \
      --background-manifest "$background_manifest" \
      --background-report "$background_report" \
      --exclude-manifest "$BASELINE_TRAIN_MANIFEST" \
      --exclude-manifest "$BASELINE_VALIDATION_MANIFEST" \
      --output-dir "$disjoint_dir" \
      --split val
  )
fi
"$TASK_PYTHON" - "$disjoint_report" "$disjoint_manifest" \
  "$MINIMUM_UNIQUE_GPS_BLOCKS" "$background_report" \
  "$BASELINE_TRAIN_MANIFEST" "$BASELINE_VALIDATION_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, manifest_path, minimum_blocks, source_report, *exclusions = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
expected_exclusions = [digest(path) for path in exclusions]
observed_exclusions = [row.get("sha256") for row in report.get("exclusion_manifests", [])]
if (
    report.get("status") != "verified_group_disjoint_development_background_subset"
    or not report.get("passed")
    or report.get("required_split") != "val"
    or report.get("split_strategy") != "hash_threshold_v1"
    or int(report.get("unique_gps_blocks", -1)) < int(minimum_blocks)
    or int(report.get("selected_exclusion_gps_block_overlap", -1)) != 0
    or int(report.get("splits", {}).get("test", {}).get("windows", -1)) != 0
    or report.get("source_background_report_sha256") != digest(source_report)
    or observed_exclusions != expected_exclusions
    or report.get("manifest_sha256") != digest(manifest_path)
):
    raise SystemExit("independent validation background failed its frozen GPS-disjoint gate")
PY

purpose_dir="$OUTPUT_ROOT/purpose-partition"
purpose_report="$purpose_dir/background_purpose_partition_report.json"
calibration_manifest="$purpose_dir/candidate_calibration/background_windows.jsonl"
injection_background_manifest="$purpose_dir/injection_validation/background_windows.jsonl"
injection_background_report="$purpose_dir/injection_validation/background_plan_report.json"
if [[ ! -s "$purpose_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli background-purpose-partition \
      --background-manifest "$disjoint_manifest" \
      --background-report "$disjoint_report" \
      --output-dir "$purpose_dir" \
      --injection-fraction 0.5 \
      --seed "$PURPOSE_PARTITION_SEED"
  )
fi
"$TASK_PYTHON" - "$purpose_report" "$disjoint_manifest" "$disjoint_report" \
  "$calibration_manifest" "$injection_background_manifest" \
  "$injection_background_report" "$MINIMUM_PURPOSE_GPS_BLOCKS" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    report_path,
    source_manifest,
    source_report,
    calibration_manifest,
    injection_manifest,
    injection_report,
    minimum_blocks,
) = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
purposes = report.get("purposes", {})
calibration = purposes.get("candidate_calibration", {})
injection = purposes.get("injection_validation", {})
if (
    report.get("status") != "verified_validation_gps_purpose_partition"
    or not report.get("passed")
    or int(report.get("purpose_gps_block_overlap", -1)) != 0
    or report.get("complete_source_gps_block_coverage") is not True
    or report.get("source_background_manifest_sha256") != digest(source_manifest)
    or report.get("source_background_report_sha256") != digest(source_report)
    or int(calibration.get("unique_gps_blocks", -1)) < int(minimum_blocks)
    or int(injection.get("unique_gps_blocks", -1)) < int(minimum_blocks)
    or calibration.get("manifest_sha256") != digest(calibration_manifest)
    or injection.get("manifest_sha256") != digest(injection_manifest)
    or injection.get("report_sha256") != digest(injection_report)
):
    raise SystemExit("candidate-calibration/injection-validation GPS partition failed")
PY

recipes_dir="$OUTPUT_ROOT/recipes"
recipes="$recipes_dir/injection_recipes.jsonl"
recipe_report="$recipes_dir/injection_plan_report.json"
if [[ ! -s "$recipe_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli injection-plan \
      --background-manifest "$injection_background_manifest" \
      --background-report "$injection_background_report" \
      --output-dir "$recipes_dir" \
      --train-count 0 \
      --validation-count "$VALIDATION_COUNT" \
      --test-count 0 \
      --seed "$INJECTION_SEED"
  )
fi
"$TASK_PYTHON" - "$recipe_report" "$recipes" "$injection_background_manifest" \
  "$injection_background_report" "$VALIDATION_COUNT" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, recipes_path, background_manifest, background_report, count = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
if (
    report.get("status") != "cosmological_injection_recipe_plan_requires_validated_waveform_backend"
    or int(report.get("recipes", -1)) != int(count)
    or int(report.get("unique_injection_ids", -1)) != int(count)
    or int(report.get("unique_waveform_ids", -1)) != int(count)
    or report.get("counts_by_split") != {"val": int(count)}
    or report.get("requested_counts_by_split") != {"train": 0, "val": int(count), "test": 0}
    or any(report.get("cross_split_injection_overlaps", {}).values())
    or not report.get("approximant_domain_audit", {}).get("passed")
    or report.get("background_manifest_sha256") != digest(background_manifest)
    or report.get("background_report_sha256") != digest(background_report)
    or report.get("manifest_sha256") != digest(recipes_path)
):
    raise SystemExit("independent validation injection recipes failed their identity gate")
PY

waveform_report="$recipes_dir/waveform_validation_report.json"
if [[ ! -s "$waveform_report" ]]; then
  require_waveform_runtime
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$WAVEFORM_PYTHON" -m gwyolo.cli waveform-validate \
      --recipes "$recipes" \
      --output "$waveform_report" \
      --sample-rate "$SAMPLE_RATE" \
      --reference-duration 128 \
      --per-family "$WAVEFORM_CASES_PER_FAMILY"
  )
fi
"$TASK_PYTHON" - "$waveform_report" "$recipes" "$WAVEFORM_CASES_PER_FAMILY" \
  "$SAMPLE_RATE" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, recipes_path, per_family, sample_rate = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
if (
    not report.get("passed")
    or int(report.get("per_family", -1)) != int(per_family)
    or int(report.get("sample_rate", -1)) != int(sample_rate)
    or int(report.get("selected_cases", -1)) != 3 * int(per_family)
    or any(not row.get("passed") for row in report.get("cases", []))
    or report.get("recipe_manifest_sha256")
    != hashlib.sha256(pathlib.Path(recipes_path).read_bytes()).hexdigest()
):
    raise SystemExit("external waveform-equivalence validation did not pass")
PY

materialized_dir="$OUTPUT_ROOT/materialized"
materialized_manifest="$materialized_dir/materialized_injections.jsonl"
materialization_report="$materialized_dir/materialization_report.json"
if [[ ! -s "$materialization_report" ]]; then
  require_waveform_runtime
  completed=0
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    printf '%s independent-validation-materialization attempt=%s\n' \
      "$(date -u +%FT%TZ)" "$attempt"
    if (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$WAVEFORM_PYTHON" -m gwyolo.cli injection-materialize \
        --recipes "$recipes" \
        --background-manifest "$injection_background_manifest" \
        --output-dir "$materialized_dir" \
        --sample-rate "$SAMPLE_RATE" \
        --context-duration "$CONTEXT_DURATION" \
        --storage-mode signal_scaled_float16 \
        --split val \
        --backend-validation-report "$waveform_report"
    ); then
      completed=1
      break
    fi
    if (( attempt < MAX_ATTEMPTS )); then
      sleep "$RETRY_DELAY_SECONDS"
    fi
  done
  if (( completed != 1 )); then
    echo "independent validation materialization exhausted bounded retries" >&2
    exit 1
  fi
fi
"$TASK_PYTHON" - "$materialization_report" "$materialized_manifest" "$recipes" \
  "$injection_background_manifest" "$waveform_report" "$VALIDATION_COUNT" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, manifest_path, recipes, background, waveform, count = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
identity = report.get("identity_audit", {})
if (
    report.get("status") != "materialized_externally_validated_backend"
    or not report.get("waveform_materialization_validated")
    or report.get("selected_split") != "val"
    or int(report.get("selected_recipes", -1)) != int(count)
    or int(identity.get("unique_injection_ids", -1)) != int(count)
    or int(identity.get("unique_waveform_ids", -1)) != int(count)
    or report.get("storage_mode") != "signal_scaled_float16"
    or report.get("recipe_manifest_sha256") != digest(recipes)
    or report.get("background_manifest_sha256") != digest(background)
    or report.get("backend_validation_report_sha256") != digest(waveform)
    or report.get("manifest_sha256") != digest(manifest_path)
):
    raise SystemExit("materialized independent validation corpus failed its provenance gate")
PY

snr_dir="$OUTPUT_ROOT/snr"
snr_manifest="$snr_dir/materialized_injections_snr.jsonl"
snr_report="$snr_dir/snr_annotation_report.json"
if [[ ! -s "$snr_report" ]]; then
  require_waveform_runtime
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$WAVEFORM_PYTHON" -m gwyolo.cli injection-snr-annotate \
      --manifest "$materialized_manifest" \
      --output-dir "$snr_dir" \
      --low-frequency 20 \
      --high-frequency 500 \
      --psd-segment-seconds 8 \
      --psd-stride-seconds 4
  )
fi

arrival_dir="$OUTPUT_ROOT/arrivals"
arrival_manifest="$arrival_dir/materialized_injections_arrivals.jsonl"
arrival_report="$arrival_dir/detector_arrival_annotation_report.json"
if [[ ! -s "$arrival_report" ]]; then
  require_waveform_runtime
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$WAVEFORM_PYTHON" -m gwyolo.cli injection-arrival-annotate \
      --manifest "$snr_manifest" \
      --output-dir "$arrival_dir"
  )
fi
"$TASK_PYTHON" - "$snr_report" "$snr_manifest" "$materialized_manifest" \
  "$arrival_report" "$arrival_manifest" "$VALIDATION_COUNT" <<'PY'
import hashlib
import json
import pathlib
import sys

snr_report_path, snr_manifest, materialized, arrival_report_path, arrival_manifest, count = sys.argv[1:]
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
snr = json.loads(pathlib.Path(snr_report_path).read_text(encoding="utf-8"))
arrival = json.loads(pathlib.Path(arrival_report_path).read_text(encoding="utf-8"))
if (
    snr.get("status") != "empirical_noise_optimal_snr_annotation"
    or int(snr.get("rows", -1)) != int(count)
    or snr.get("split_counts") != {"val": int(count)}
    or snr.get("input_manifest_sha256") != digest(materialized)
    or snr.get("output_manifest_sha256") != digest(snr_manifest)
    or arrival.get("status") != "verified_geometric_detector_arrival_annotation"
    or int(arrival.get("rows", -1)) != int(count)
    or int(arrival.get("unique_injection_ids", -1)) != int(count)
    or arrival.get("splits") != {"val": int(count)}
    or arrival.get("input_manifest_sha256") != digest(snr_manifest)
    or arrival.get("manifest_sha256") != digest(arrival_manifest)
):
    raise SystemExit("SNR or detector-arrival annotation failed its final identity gate")
PY

endpoint_report="$OUTPUT_ROOT/independent_validation_endpoint_report.json"
if [[ ! -s "$endpoint_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli independent-validation-endpoint-freeze \
      --purpose-partition-report "$purpose_report" \
      --injection-plan-report "$recipe_report" \
      --waveform-validation-report "$waveform_report" \
      --materialization-report "$materialization_report" \
      --snr-annotation-report "$snr_report" \
      --arrival-annotation-report "$arrival_report" \
      --output "$endpoint_report"
  )
fi
"$TASK_PYTHON" - "$endpoint_report" "$VALIDATION_COUNT" \
  "$MINIMUM_PURPOSE_GPS_BLOCKS" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path, count, minimum_blocks = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
components = report.get("component_reports", {})
expected_components = {
    "purpose_partition",
    "injection_plan",
    "waveform_validation",
    "materialization",
    "snr_annotation",
    "arrival_annotation",
}
if (
    report.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or not report.get("passed")
    or report.get("test_rows_read") != 0
    or report.get("test_evaluation") is not None
    or int(report.get("rows", -1)) != int(count)
    or int(report.get("purpose_gps_block_overlap", -1)) != 0
    or int(report.get("candidate_calibration_unique_gps_blocks", -1))
    < int(minimum_blocks)
    or int(report.get("injection_validation_unique_gps_blocks", -1))
    < int(minimum_blocks)
    or set(components) != expected_components
    or any(digest(item["path"]) != item["sha256"] for item in components.values())
    or digest(report["candidate_calibration_background_manifest_path"])
    != report["candidate_calibration_background_manifest_sha256"]
    or digest(report["injection_arrival_manifest_path"])
    != report["injection_arrival_manifest_sha256"]
):
    raise SystemExit("frozen independent validation endpoint failed final replay")
PY

printf '%s independent-validation-arrivals=%s\n' \
  "$(date -u +%FT%TZ)" "$arrival_manifest"
printf '%s candidate-calibration-background=%s\n' \
  "$(date -u +%FT%TZ)" "$calibration_manifest"
printf '%s independent-validation-endpoint=%s\n' \
  "$(date -u +%FT%TZ)" "$endpoint_report"
