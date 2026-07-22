#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_CALIBRATION
  BACKGROUND_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT
  CANDIDATE_PIPELINE_REPORT
  OUTPUT_ROOT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD)" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "background dependence replay checkout differs from its frozen commit" >&2
  exit 2
fi
for input in \
  "$TASK_PYTHON" \
  "$SOURCE_CALIBRATION" \
  "$BACKGROUND_MANIFEST" \
  "$INDEPENDENT_VALIDATION_ENDPOINT" \
  "$CANDIDATE_PIPELINE_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "background dependence replay input is absent: $input" >&2
    exit 2
  fi
done

mapfile -t settings < <(
  "$TASK_PYTHON" - "$SOURCE_CALIBRATION" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "frozen_validation_candidate_search_calibration":
    raise SystemExit("source calibration has the wrong status")
print(report["validation_time_slide_report_path"])
print(report["validation_injection_ranking_report_path"])
print(report["target_far_per_year"])
print(report["bootstrap_replicates"])
print(report["seed"])
PY
)
if (( ${#settings[@]} != 5 )); then
  echo "source calibration did not resolve five replay settings" >&2
  exit 2
fi
slide_report=${settings[0]}
injection_report=${settings[1]}
target_far=${settings[2]}
bootstrap_replicates=${settings[3]}
bootstrap_seed=${settings[4]}
for input in "$slide_report" "$injection_report"; do
  if [[ ! -f "$input" ]]; then
    echo "source calibration dependency is absent: $input" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"
calibration="$OUTPUT_ROOT/frozen_validation_candidate_search_calibration.json"
binding="$OUTPUT_ROOT/frozen_validation_candidate_search_calibration_endpoint_bound.json"
if [[ ! -s "$calibration" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibrate \
      --validation-time-slide-report "$slide_report" \
      --validation-background-manifest "$BACKGROUND_MANIFEST" \
      --validation-injection-ranking-report "$injection_report" \
      --target-far-per-year "$target_far" \
      --bootstrap-replicates "$bootstrap_replicates" \
      --seed "$bootstrap_seed" \
      --output "$calibration"
  )
fi
if [[ ! -s "$binding" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibration-endpoint-bind \
      --independent-validation-endpoint "$INDEPENDENT_VALIDATION_ENDPOINT" \
      --candidate-pipeline-report "$CANDIDATE_PIPELINE_REPORT" \
      --calibration-report "$calibration" \
      --expected-target-far-per-year "$target_far" \
      --minimum-bootstrap-replicates "$bootstrap_replicates" \
      --output "$binding"
  )
fi

"$TASK_PYTHON" - "$calibration" "$binding" <<'PY'
import json
import pathlib
import sys

calibration = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
binding = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
if (
    calibration.get("background_dependence_audit", {}).get("passed") is not True
    or binding.get("status")
    != "frozen_validation_candidate_search_calibration_endpoint_bound"
    or binding.get("passed") is not True
):
    raise SystemExit("background dependence replay did not pass its publication gate")
PY

