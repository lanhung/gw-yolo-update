#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  LOCKED_SUITE_PLAN
  LOCKED_ACCESS_LOG
  RAW_CALIBRATION_REPORT
  MASK_CALIBRATION_REPORT
  VALIDATION_RAW_MASK_COMPARISON_REPORT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$LOCKED_SUITE_PLAN" \
  "$LOCKED_ACCESS_LOG" \
  "$RAW_CALIBRATION_REPORT" \
  "$MASK_CALIBRATION_REPORT" \
  "$VALIDATION_RAW_MASK_COMPARISON_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 2
fi

mapfile -t suite_settings < <(
  "$TASK_PYTHON" - "$LOCKED_SUITE_PLAN" "$LOCKED_ACCESS_LOG" <<'PY'
import hashlib
import json
import pathlib
import sys

plan_path = pathlib.Path(sys.argv[1]).resolve()
access_path = pathlib.Path(sys.argv[2]).resolve()
plan = json.loads(plan_path.read_text(encoding="utf-8"))
access = json.loads(access_path.read_text(encoding="utf-8"))
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
frozen = access.get("frozen_artifacts", {}).get("locked_suite_plan", {})
if (
    plan.get("status") != "frozen_locked_evaluation_suite_plan"
    or access.get("status") != "locked_evaluation_corpus_opened_once"
    or frozen.get("path") != str(plan_path)
    or frozen.get("sha256") != digest(plan_path)
    or access.get("code_commit") != plan.get("code_commit")
):
    raise SystemExit("one-time access log does not bind the locked suite plan")
endpoints = plan["endpoints"]
for value in (
    plan["outputs"]["raw_candidate_search"],
    plan["outputs"]["mask_candidate_search"],
    plan["outputs"]["paired_raw_mask_search"],
    endpoints["minimum_test_live_time_years"],
    endpoints["minimum_test_injections"],
    endpoints["bootstrap_replicates"],
    endpoints["bootstrap_seed"],
    plan["inputs"]["raw_test_time_slide_report"],
    plan["inputs"]["mask_test_time_slide_report"],
    plan["inputs"]["raw_test_background_manifest"],
    plan["inputs"]["mask_test_background_manifest"],
    plan["inputs"]["raw_test_injection_ranking_report"],
    plan["inputs"]["mask_test_injection_ranking_report"],
):
    print(value)
PY
)
if (( ${#suite_settings[@]} != 13 )); then
  echo "locked suite plan did not resolve the search endpoints" >&2
  exit 2
fi
raw_output=${suite_settings[0]}
mask_output=${suite_settings[1]}
paired_output=${suite_settings[2]}
minimum_live_time=${suite_settings[3]}
minimum_injections=${suite_settings[4]}
bootstrap_replicates=${suite_settings[5]}
bootstrap_seed=${suite_settings[6]}
RAW_TEST_TIME_SLIDE_REPORT=${suite_settings[7]}
MASK_TEST_TIME_SLIDE_REPORT=${suite_settings[8]}
RAW_TEST_BACKGROUND_MANIFEST=${suite_settings[9]}
MASK_TEST_BACKGROUND_MANIFEST=${suite_settings[10]}
RAW_TEST_INJECTION_RANKING_REPORT=${suite_settings[11]}
MASK_TEST_INJECTION_RANKING_REPORT=${suite_settings[12]}
for input in \
  "$RAW_TEST_TIME_SLIDE_REPORT" \
  "$MASK_TEST_TIME_SLIDE_REPORT" \
  "$RAW_TEST_BACKGROUND_MANIFEST" \
  "$MASK_TEST_BACKGROUND_MANIFEST" \
  "$RAW_TEST_INJECTION_RANKING_REPORT" \
  "$MASK_TEST_INJECTION_RANKING_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "predeclared locked search input is absent: $input" >&2
    exit 2
  fi
done

run_arm() {
  local arm=$1
  local calibration=$2
  local slides=$3
  local background=$4
  local rankings=$5
  local output=$6
  if [[ ! -s "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src
      "$TASK_PYTHON" -m gwyolo.cli candidate-search-evaluate-frozen \
        --calibration-report "$calibration" \
        --test-time-slide-report "$slides" \
        --test-background-manifest "$background" \
        --test-injection-ranking-report "$rankings" \
        --minimum-test-live-time-years "$minimum_live_time" \
        --minimum-test-injections "$minimum_injections" \
        --bootstrap-replicates "$bootstrap_replicates" \
        --seed "$bootstrap_seed" \
        --locked-suite-plan "$LOCKED_SUITE_PLAN" \
        --access-log "$LOCKED_ACCESS_LOG" \
        --output-key "${arm}_candidate_search" \
        --output "$output"
    )
  fi
}

run_arm raw \
  "$RAW_CALIBRATION_REPORT" \
  "$RAW_TEST_TIME_SLIDE_REPORT" \
  "$RAW_TEST_BACKGROUND_MANIFEST" \
  "$RAW_TEST_INJECTION_RANKING_REPORT" \
  "$raw_output"
run_arm mask \
  "$MASK_CALIBRATION_REPORT" \
  "$MASK_TEST_TIME_SLIDE_REPORT" \
  "$MASK_TEST_BACKGROUND_MANIFEST" \
  "$MASK_TEST_INJECTION_RANKING_REPORT" \
  "$mask_output"

if [[ ! -s "$paired_output" ]]; then
  mkdir -p "$(dirname "$paired_output")"
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-compare-locked \
      --raw-locked-report "$raw_output" \
      --mask-locked-report "$mask_output" \
      --validation-comparison-report "$VALIDATION_RAW_MASK_COMPARISON_REPORT" \
      --locked-suite-plan "$LOCKED_SUITE_PLAN" \
      --access-log "$LOCKED_ACCESS_LOG" \
      --bootstrap-replicates "$bootstrap_replicates" \
      --seed "$bootstrap_seed" \
      --output "$paired_output"
  )
fi
