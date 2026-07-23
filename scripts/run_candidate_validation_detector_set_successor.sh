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
  --exposure-safety-factor "${EXPOSURE_SAFETY_FACTOR:-1.0}"

report="$OUTPUT_ROOT/candidate_validation_detector_set_block_pipeline_report.json"
"$TASK_PYTHON" - "$report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
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
):
    raise SystemExit("detector-set validation successor failed replay")
PY
