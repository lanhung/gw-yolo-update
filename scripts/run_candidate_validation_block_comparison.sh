#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  BACKGROUND_MANIFEST
  BASELINE_PIPELINE_REPORT
  PROMOTED_PIPELINE_REPORT
  BASELINE_CALIBRATED_CANDIDATES
  PROMOTED_CALIBRATED_CANDIDATES
  BASELINE_INJECTION_RANKING_REPORT
  PROMOTED_INJECTION_RANKING_REPORT
  PROMOTION_CONFIG
  OUTPUT_ROOT
  COMPARISON_OUTPUT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$BACKGROUND_MANIFEST" \
  "$BASELINE_PIPELINE_REPORT" \
  "$PROMOTED_PIPELINE_REPORT" \
  "$BASELINE_CALIBRATED_CANDIDATES" \
  "$PROMOTED_CALIBRATED_CANDIDATES" \
  "$BASELINE_INJECTION_RANKING_REPORT" \
  "$PROMOTED_INJECTION_RANKING_REPORT" \
  "$PROMOTION_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "task code directory is invalid: $TASK_CODE_DIR" >&2
  exit 2
fi

"$TASK_PYTHON" - "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$BACKGROUND_MANIFEST" "$BASELINE_PIPELINE_REPORT" \
  "$PROMOTED_PIPELINE_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys

endpoint_path, background_path, baseline_path, promoted_path = sys.argv[1:]
endpoint = json.loads(pathlib.Path(endpoint_path).read_text(encoding="utf-8"))
pipelines = [
    json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for path in (baseline_path, promoted_path)
]
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
background_hash = digest(background_path)
injection_hash = endpoint.get("injection_arrival_manifest_sha256")
if (
    endpoint.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or not endpoint.get("passed")
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or set(components) != expected_components
    or any(digest(item["path"]) != item["sha256"] for item in components.values())
    or endpoint.get("candidate_calibration_background_manifest_sha256")
    != background_hash
    or any(
        pipeline.get("status") != "validation_only_clustered_candidate_search_pipeline"
        or pipeline.get("test_evaluation") is not None
        or pipeline.get("run_identity", {}).get("background_manifest_sha256")
        != background_hash
        or pipeline.get("run_identity", {}).get("injection_manifest_sha256")
        != injection_hash
        for pipeline in pipelines
    )
    or len({pipeline.get("run_identity", {}).get("code_commit") for pipeline in pipelines})
    != 1
):
    raise SystemExit("block comparison inputs do not match the frozen independent endpoint")
PY

mkdir -p "$OUTPUT_ROOT"
for arm in baseline promoted; do
  if [[ "$arm" == "baseline" ]]; then
    pipeline_report=$BASELINE_PIPELINE_REPORT
    candidates=$BASELINE_CALIBRATED_CANDIDATES
    rankings=$BASELINE_INJECTION_RANKING_REPORT
  else
    pipeline_report=$PROMOTED_PIPELINE_REPORT
    candidates=$PROMOTED_CALIBRATED_CANDIDATES
    rankings=$PROMOTED_INJECTION_RANKING_REPORT
  fi
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-block-recalibrate \
      --pipeline-report "$pipeline_report" \
      --background-manifest "$BACKGROUND_MANIFEST" \
      --calibrated-candidate-manifest "$candidates" \
      --injection-ranking-report "$rankings" \
      --output-dir "$OUTPUT_ROOT/$arm"
  )
done

if [[ ! -s "$COMPARISON_OUTPUT" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-compare \
      --baseline-report \
      "$OUTPUT_ROOT/baseline/candidate_validation_block_pipeline_report.json" \
      --promoted-report \
      "$OUTPUT_ROOT/promoted/candidate_validation_block_pipeline_report.json" \
      --config "$PROMOTION_CONFIG" \
      --output "$COMPARISON_OUTPUT"
  )
fi
