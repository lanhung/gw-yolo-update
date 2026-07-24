#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_PIPELINE_REPORT
  BACKGROUND_MANIFEST
  NETWORK_CONFIG
  OUTPUT_ROOT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$SOURCE_PIPELINE_REPORT" \
  "$BACKGROUND_MANIFEST" \
  "$NETWORK_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD)" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "task code checkout differs from GWYOLO_CODE_COMMIT" >&2
  exit 2
fi

pipeline_dir=$(dirname "$SOURCE_PIPELINE_REPORT")
background_candidates="$pipeline_dir/background_candidates_calibrated.jsonl"
injection_candidates="$pipeline_dir/injection_candidates_calibrated.jsonl"
injection_score_report="$pipeline_dir/injection_score/injection_score_report.json"
expanded_args=()
successor_mode=pilot
if [[ -n "${EXPANDED_BACKGROUND_MERGE_REPORT:-}" \
  || -n "${BACKGROUND_PLAN_AUTHORIZATION:-}" ]]; then
  if [[ -z "${EXPANDED_BACKGROUND_MERGE_REPORT:-}" \
    || -z "${BACKGROUND_PLAN_AUTHORIZATION:-}" ]]; then
    echo "expanded successor requires merge report and authorization" >&2
    exit 2
  fi
  for input in \
    "$EXPANDED_BACKGROUND_MERGE_REPORT" \
    "$BACKGROUND_PLAN_AUTHORIZATION"; do
    if [[ ! -f "$input" ]]; then
      echo "expanded successor input is absent: $input" >&2
      exit 2
    fi
  done
  background_candidates=$("$TASK_PYTHON" - \
    "$EXPANDED_BACKGROUND_MERGE_REPORT" "$BACKGROUND_MANIFEST" <<'PY'
import json
import pathlib
import sys

merge = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
background = pathlib.Path(sys.argv[2]).resolve()
declared_background = pathlib.Path(
    str(merge.get("background_manifest_path", ""))
).resolve()
candidate = pathlib.Path(
    str(merge.get("candidate_manifests", {}).get("val", {}).get("path", ""))
).resolve()
if (
    merge.get("status") != "verified_merged_streamed_candidate_background"
    or declared_background != background
    or not background.is_file()
    or not candidate.is_file()
):
    raise SystemExit("expanded merge does not resolve the requested artifacts")
print(candidate)
PY
  )
  expanded_args+=(
    --expanded-background-merge-report
    "$EXPANDED_BACKGROUND_MERGE_REPORT"
    --background-plan-authorization
    "$BACKGROUND_PLAN_AUTHORIZATION"
  )
  successor_mode=expanded
fi
for input in \
  "$background_candidates" \
  "$injection_candidates" \
  "$injection_score_report"; do
  if [[ ! -f "$input" ]]; then
    echo "source pipeline artifact is absent: $input" >&2
    exit 2
  fi
done
injection_triggers=$("$TASK_PYTHON" - "$injection_score_report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
path = pathlib.Path(str(report.get("triggers_path", ""))).resolve()
if not path.is_file():
    raise SystemExit("source injection trigger manifest is absent")
print(path)
PY
)

mkdir -p "$OUTPUT_ROOT"
"$TASK_PYTHON" -m gwyolo.cli \
  candidate-search-validation-detector-set-recalibrate \
  --pipeline-report "$SOURCE_PIPELINE_REPORT" \
  --background-manifest "$BACKGROUND_MANIFEST" \
  --calibrated-background-candidate-manifest "$background_candidates" \
  --injection-trigger-manifest "$injection_triggers" \
  --calibrated-injection-candidate-manifest "$injection_candidates" \
  --network-config "$NETWORK_CONFIG" \
  --output-dir "$OUTPUT_ROOT" \
  --zero-count-confidence "${ZERO_COUNT_CONFIDENCE:-0.90}" \
  --exposure-safety-factor "${EXPOSURE_SAFETY_FACTOR:-1.0}" \
  "${expanded_args[@]}"

report="$OUTPUT_ROOT/candidate_validation_detector_set_block_pipeline_report.json"
"$TASK_PYTHON" - "$report" \
  "$successor_mode" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
calibration = report.get("frozen_search", {})
if (
    report.get("status") != "validation_only_clustered_candidate_search_pipeline"
    or report.get("scientific_claim_allowed") is not False
    or report.get("test_evaluation") is not None
    or calibration.get("selection_data")
    != "validation_variable_detector_set_block_permutations_only"
    or calibration.get("slide_schedule_audit", {}).get("schedule_kind")
    != "variable_detector_set_block_permutation"
    or calibration.get("background_dependence_audit", {}).get("status")
    != "detector_set_candidate_background_dependence_audit_v1"
    or (
        mode == "expanded"
        and calibration.get("publication_calibration_eligible") is not True
    )
):
    raise SystemExit("detector-set validation successor failed replay")
PY

if [[ "$successor_mode" == expanded ]]; then
  mapfile -t binding_settings < <(
    "$TASK_PYTHON" - "$BACKGROUND_PLAN_AUTHORIZATION" "$report" <<'PY'
import json
import pathlib
import sys

authorization = json.loads(
    pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
)
pipeline = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
print(authorization["independent_validation_endpoint"]["path"])
print(authorization["authorization_identity"]["target_far_per_year"])
print(pipeline["frozen_search"]["bootstrap_replicates"])
PY
  )
  if (( ${#binding_settings[@]} != 3 )); then
    echo "expanded successor did not resolve endpoint settings" >&2
    exit 2
  fi
  endpoint=${binding_settings[0]}
  target_far=${binding_settings[1]}
  bootstrap_replicates=${binding_settings[2]}
  binding="$OUTPUT_ROOT/frozen_validation_candidate_search_calibration_endpoint_bound.json"
  if [[ ! -s "$binding" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli \
      candidate-search-calibration-endpoint-bind \
      --independent-validation-endpoint "$endpoint" \
      --candidate-pipeline-report "$report" \
      --calibration-report "$OUTPUT_ROOT/frozen_candidate_search_calibration.json" \
      --expected-target-far-per-year "$target_far" \
      --minimum-bootstrap-replicates "$bootstrap_replicates" \
      --background-plan-authorization "$BACKGROUND_PLAN_AUTHORIZATION" \
      --expanded-background-merge-report "$EXPANDED_BACKGROUND_MERGE_REPORT" \
      --output "$binding"
  fi
  "$TASK_PYTHON" - "$binding" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status")
    != "frozen_validation_candidate_search_calibration_endpoint_bound"
    or report.get("passed") is not True
    or not isinstance(report.get("expanded_background_lineage"), dict)
):
    raise SystemExit("expanded detector-set endpoint binding failed")
PY
fi
