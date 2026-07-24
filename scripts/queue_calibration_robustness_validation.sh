#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  REFERENCE_CODE_DIR
  REFERENCE_CODE_COMMIT
  FIVE_SEED_SUMMARY
  PROMOTED_BLOCK_PIPELINE_REPORT
  CALIBRATION_PLAN
  BACKGROUND_MANIFEST
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
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
adapter_config=${ADAPTER_CONFIG:-$script_dir/../configs/physical_overlap_finetune_glitch_adapter.yaml}
# The shared resolver enforces test_data_opened=false and exact config hashes.

upstream_pid=${UPSTREAM_PID:-}
while [[ ! -s "$FIVE_SEED_SUMMARY" || ! -s "$PROMOTED_BLOCK_PIPELINE_REPORT" \
  || ! -s "$BLOCK_SCHEDULE" || ! -s "$BASELINE_CALIBRATION_REPORT" ]]; do
  if [[ -n "$upstream_pid" ]] && ! kill -0 "$upstream_pid" 2>/dev/null; then
    echo "upstream publication DAG ended before calibration prerequisites existed" >&2
    exit 1
  fi
  sleep 30
done

selection_output=$(TASK_PYTHON="$TASK_PYTHON" bash \
  "$script_dir/resolve_promoted_overlap_model.sh" \
  "$FIVE_SEED_SUMMARY" "$UNIFORM_CONFIG" "$FAMILY_BALANCED_CONFIG" \
  "$adapter_config")
readarray -t selection <<< "$selection_output"
if (( ${#selection[@]} != 3 )); then
  echo "five-seed summary did not resolve one arm, checkpoint and config" >&2
  exit 2
fi
arm=${selection[0]}
checkpoint=${selection[1]}
model_config=${selection[2]}

physics_output=$("$TASK_PYTHON" - "$PROMOTED_BLOCK_PIPELINE_REPORT" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
frozen = report.get("frozen_search", {})
identity = report.get("run_identity", {})
if (
    report.get("status") != "validation_only_clustered_candidate_search_pipeline"
    or report.get("test_evaluation") is not None
    or frozen.get("status") != "frozen_validation_candidate_search_calibration"
    or frozen.get("publication_calibration_eligible") is not True
    or frozen.get("slide_schedule_audit", {}).get("passed") is not True
):
    raise SystemExit("promoted block calibration prerequisite did not pass")
print(report["physical_delay_limit_seconds"])
print(report["empirical_timing_uncertainty_seconds"])
print(report["coincidence_window_seconds"])
print(identity["reference_ifo"])
print(identity["second_ifo"])
print(identity["cluster_window_seconds"])
PY
)
readarray -t physics <<< "$physics_output"
if (( ${#physics[@]} != 6 )); then
  echo "promoted block report did not resolve its physical timing contract" >&2
  exit 2
fi

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  REFERENCE_CODE_DIR="$REFERENCE_CODE_DIR" \
  REFERENCE_CODE_COMMIT="$REFERENCE_CODE_COMMIT" \
  CALIBRATION_PLAN="$CALIBRATION_PLAN" \
  BACKGROUND_MANIFEST="$BACKGROUND_MANIFEST" \
  INJECTION_MANIFEST="$INJECTION_MANIFEST" \
  CHECKPOINT="$checkpoint" \
  MODEL_CONFIG="$model_config" \
  COHERENCE_CONFIG="$COHERENCE_CONFIG" \
  NETWORK_CONFIG="$NETWORK_CONFIG" \
  TIMING_CALIBRATION_REPORT="$TIMING_CALIBRATION_REPORT" \
  BLOCK_SCHEDULE="$BLOCK_SCHEDULE" \
  BASELINE_CALIBRATION_REPORT="$BASELINE_CALIBRATION_REPORT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  PHYSICAL_DELAY_LIMIT_SECONDS="${physics[0]}" \
  EMPIRICAL_TIMING_UNCERTAINTY_SECONDS="${physics[1]}" \
  COINCIDENCE_WINDOW_SECONDS="${physics[2]}" \
  REFERENCE_IFO="${physics[3]}" \
  SECOND_IFO="${physics[4]}" \
  CLUSTER_WINDOW_SECONDS="${physics[5]}" \
  bash "$TASK_CODE_DIR/scripts/run_calibration_robustness_validation.sh"
