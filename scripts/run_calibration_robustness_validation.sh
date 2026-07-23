#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  REFERENCE_CODE_DIR
  REFERENCE_CODE_COMMIT
  CALIBRATION_PLAN
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  CHECKPOINT
  MODEL_CONFIG
  COHERENCE_CONFIG
  NETWORK_CONFIG
  TIMING_CALIBRATION_REPORT
  BLOCK_SCHEDULE
  BASELINE_CALIBRATION_REPORT
  OUTPUT_ROOT
  PHYSICAL_DELAY_LIMIT_SECONDS
  EMPIRICAL_TIMING_UNCERTAINTY_SECONDS
  COINCIDENCE_WINDOW_SECONDS
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for input in \
  "$TASK_PYTHON" \
  "$CALIBRATION_PLAN" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$CHECKPOINT" \
  "$MODEL_CONFIG" \
  "$COHERENCE_CONFIG" \
  "$NETWORK_CONFIG" \
  "$TIMING_CALIBRATION_REPORT" \
  "$BLOCK_SCHEDULE" \
  "$BASELINE_CALIBRATION_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
for code_dir in "$TASK_CODE_DIR" "$REFERENCE_CODE_DIR"; do
  if [[ ! -d "$code_dir/src/gwyolo" ]]; then
    echo "code directory is invalid: $code_dir" >&2
    exit 2
  fi
done

read -r -a model_ifos <<< "${MODEL_IFOS:-H1 L1 V1}"
read -r -a q_values <<< "${Q_VALUES:-4 8 16}"
target_sample_rate=${TARGET_SAMPLE_RATE:-1024}
context_duration=${CONTEXT_DURATION:-64}
chirp_threshold=${CHIRP_THRESHOLD:-0.3}
minimum_bins=${MINIMUM_BINS:-1}
reference_ifo=${REFERENCE_IFO:-H1}
second_ifo=${SECOND_IFO:-L1}
cluster_window_seconds=${CLUSTER_WINDOW_SECONDS:-0.1}
truth_association_window_seconds=${TRUTH_ASSOCIATION_WINDOW_SECONDS:-0.25}

robustness_config="$TASK_CODE_DIR/configs/calibration_perturbation_o4a_validation.yaml"
detector_set_mode=$("$TASK_PYTHON" - \
  "$TASK_CODE_DIR" "$GWYOLO_CODE_COMMIT" \
  "$REFERENCE_CODE_DIR" "$REFERENCE_CODE_COMMIT" \
  "$CALIBRATION_PLAN" "$BACKGROUND_MANIFEST" "$INJECTION_MANIFEST" \
  "$CHECKPOINT" "$MODEL_CONFIG" "$NETWORK_CONFIG" \
  "$TIMING_CALIBRATION_REPORT" \
  "$BLOCK_SCHEDULE" "$BASELINE_CALIBRATION_REPORT" \
  "$robustness_config" \
  "$PHYSICAL_DELAY_LIMIT_SECONDS" "$EMPIRICAL_TIMING_UNCERTAINTY_SECONDS" \
  "$COINCIDENCE_WINDOW_SECONDS" <<'PY'
import hashlib
import json
import math
import pathlib
import subprocess
import sys

(
    task_dir,
    task_commit,
    reference_dir,
    reference_commit,
    plan_path,
    background_path,
    injection_path,
    checkpoint_path,
    model_config_path,
    network_config_path,
    timing_path,
    schedule_path,
    baseline_path,
    robustness_config_path,
    physical_delay,
    timing_uncertainty,
    coincidence_window,
) = sys.argv[1:]
digest = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
git_head = lambda path: subprocess.run(
    ["git", "-C", path, "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
timing = json.loads(pathlib.Path(timing_path).read_text(encoding="utf-8"))
schedule = json.loads(pathlib.Path(schedule_path).read_text(encoding="utf-8"))
baseline = json.loads(pathlib.Path(baseline_path).read_text(encoding="utf-8"))
import yaml

robustness = yaml.safe_load(
    pathlib.Path(robustness_config_path).read_text(encoding="utf-8")
)["calibration_robustness"]
required_subsets = set(robustness.get("required_detector_subsets", []))
minimum_per_subset = int(
    robustness.get("minimum_injections_per_detector_subset", 1)
)
observed_subsets = {}
with pathlib.Path(injection_path).open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            row = json.loads(line)
            subset = "+".join(
                row.get("valid_ifos", row.get("ifos", []))
            )
            observed_subsets[subset] = observed_subsets.get(subset, 0) + 1
undersized_subsets = {
    subset: observed_subsets.get(subset, 0)
    for subset in sorted(required_subsets)
    if observed_subsets.get(subset, 0) < minimum_per_subset
}
if undersized_subsets:
    raise SystemExit(
        "calibration robustness injection corpus has undersized detector strata: "
        + ", ".join(
            f"{subset}={count}/{minimum_per_subset}"
            for subset, count in undersized_subsets.items()
        )
    )
identity = baseline.get("identity", {})
expected_window = float(physical_delay) + 2 * float(timing_uncertainty)
variable_detector_set = (
    schedule.get("status")
    == "frozen_detector_set_block_permutation_schedule"
)
expected_schedule_kind = (
    "variable_detector_set_block_permutation"
    if variable_detector_set
    else "gps_block_permutation"
)
detector_identity_valid = (
    identity.get("detector_set_policy")
    == "single_model_explicit_missing_ifo_validity_v1"
    and schedule.get("network_config_sha256")
    == digest(network_config_path)
    if variable_detector_set
    else (
        identity.get("physical_delay_limit_seconds")
        == float(physical_delay)
        and math.isclose(
            expected_window,
            float(coincidence_window),
            rel_tol=0,
            abs_tol=1e-12,
        )
    )
)
if (
    git_head(task_dir) != task_commit
    or git_head(reference_dir) != reference_commit
    or plan.get("status") != "frozen_validation_calibration_perturbation_plan"
    or plan.get("passed") is not True
    or plan.get("test_rows_read") != 0
    or len(plan.get("scenario_ids", [])) < 7
    or plan.get("manifests", {}).get("background", {}).get("sha256")
    != digest(background_path)
    or plan.get("manifests", {}).get("injection", {}).get("sha256")
    != digest(injection_path)
    or timing.get("status") != "validation_only_candidate_timing_calibration"
    or baseline.get("status") != "frozen_validation_candidate_search_calibration"
    or baseline.get("publication_calibration_eligible") is not True
    or baseline.get("slide_schedule_audit", {}).get("passed") is not True
    or baseline.get("slide_schedule_audit", {}).get("schedule_kind")
    != expected_schedule_kind
    or identity.get("candidate_checkpoint_sha256") != digest(checkpoint_path)
    or identity.get("candidate_config_sha256") != digest(model_config_path)
    or identity.get("timing_calibration_report_sha256") != digest(timing_path)
    or identity.get("empirical_timing_uncertainty_seconds")
    != float(timing_uncertainty)
    or schedule.get("status")
    not in {
        "frozen_candidate_block_permutation_schedule",
        "frozen_detector_set_block_permutation_schedule",
    }
    or schedule.get("background_manifest_sha256") != digest(background_path)
    or baseline.get("slide_schedule_audit", {}).get("schedule_sha256")
    != digest(schedule_path)
    or not detector_identity_valid
):
    raise SystemExit("calibration robustness inputs do not replay one frozen baseline")
print("1" if variable_detector_set else "0")
PY
)

mkdir -p "$OUTPUT_ROOT"
timing_transfer="$OUTPUT_ROOT/calibration_timing_transfer_compatibility.json"
if [[ ! -s "$timing_transfer" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli calibration-timing-transfer-compatibility-audit \
      --reference-code-dir "$REFERENCE_CODE_DIR" \
      --candidate-code-dir "$TASK_CODE_DIR" \
      --reference-commit "$REFERENCE_CODE_COMMIT" \
      --candidate-commit "$GWYOLO_CODE_COMMIT" \
      --output "$timing_transfer"
  )
fi

mapfile -t scenario_ids < <(
  "$TASK_PYTHON" - "$CALIBRATION_PLAN" <<'PY'
import json
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for scenario_id in plan["scenario_ids"]:
    print(scenario_id)
PY
)
if (( ${#scenario_ids[@]} < 7 )); then
  echo "calibration plan exposes fewer than seven scenarios" >&2
  exit 2
fi

receipt_args=()
for scenario_id in "${scenario_ids[@]}"; do
  scenario_root="$OUTPUT_ROOT/$scenario_id"
  background_root="$scenario_root/background"
  injection_root="$scenario_root/injection"
  mkdir -p "$background_root" "$injection_root"

  background_score="$background_root/score/trigger_score_report.json"
  if [[ ! -s "$background_score" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli trigger-score \
        --manifest "$BACKGROUND_MANIFEST" \
        --checkpoint "$CHECKPOINT" \
        --config "$MODEL_CONFIG" \
        --output-dir "$background_root/score" \
        --model-ifos "${model_ifos[@]}" \
        --q-values "${q_values[@]}" \
        --target-sample-rate "$target_sample_rate" \
        --context-duration "$context_duration" \
        --save-probabilities \
        --required-split val \
        --coherence-config "$COHERENCE_CONFIG" \
        --calibration-plan "$CALIBRATION_PLAN" \
        --calibration-scenario "$scenario_id"
    )
  fi
  if [[ ! -s "$background_root/candidates/candidate_extraction_report.json" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli candidate-extract \
        --triggers "$background_root/score/background_triggers.jsonl" \
        --output-dir "$background_root/candidates" \
        --chirp-threshold "$chirp_threshold" \
        --minimum-bins "$minimum_bins"
    )
  fi
  background_calibrated="$background_root/candidates_calibrated.jsonl"
  if [[ ! -s "$background_calibrated.report.json" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli candidate-timing-apply \
        --candidates "$background_root/candidates/single_ifo_candidates.jsonl" \
        --calibration-report "$TIMING_CALIBRATION_REPORT" \
        --calibration-perturbation-plan "$CALIBRATION_PLAN" \
        --calibration-timing-compatibility-report "$timing_transfer" \
        --output "$background_calibrated"
    )
  fi

  injection_score="$injection_root/score/injection_score_report.json"
  if [[ ! -s "$injection_score" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli injection-score \
        --manifest "$INJECTION_MANIFEST" \
        --checkpoint "$CHECKPOINT" \
        --config "$MODEL_CONFIG" \
        --output-dir "$injection_root/score" \
        --model-ifos "${model_ifos[@]}" \
        --q-values "${q_values[@]}" \
        --target-sample-rate "$target_sample_rate" \
        --save-probabilities \
        --required-split val \
        --coherence-config "$COHERENCE_CONFIG" \
        --calibration-plan "$CALIBRATION_PLAN" \
        --calibration-scenario "$scenario_id"
    )
  fi
  if [[ ! -s "$injection_root/candidates/injection_candidate_extraction_report.json" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli injection-candidate-extract \
        --injection-triggers "$injection_root/score/injection_triggers.jsonl" \
        --output-dir "$injection_root/candidates" \
        --chirp-threshold "$chirp_threshold" \
        --minimum-bins "$minimum_bins"
    )
  fi
  injection_calibrated="$injection_root/candidates_calibrated.jsonl"
  if [[ ! -s "$injection_calibrated.report.json" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli candidate-timing-apply \
        --candidates "$injection_root/candidates/single_ifo_injection_candidates.jsonl" \
        --calibration-report "$TIMING_CALIBRATION_REPORT" \
        --calibration-perturbation-plan "$CALIBRATION_PLAN" \
        --calibration-timing-compatibility-report "$timing_transfer" \
        --output "$injection_calibrated"
    )
  fi

  if [[ "$detector_set_mode" == 1 ]]; then
    ranking_report="$injection_root/rankings/val_variable_detector_set_injection_candidate_ranking_report.json"
  else
    ranking_report="$injection_root/rankings/val_injection_candidate_ranking_report.json"
  fi
  if [[ ! -s "$ranking_report" ]]; then
    if [[ "$detector_set_mode" == 1 ]]; then
      (
        cd "$TASK_CODE_DIR"
        export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
        "$TASK_PYTHON" -m gwyolo.cli detector-set-injection-candidate-rank \
          --injection-triggers "$injection_root/score/injection_triggers.jsonl" \
          --candidates "$injection_calibrated" \
          --config "$NETWORK_CONFIG" \
          --output-dir "$injection_root/rankings" \
          --split val \
          --empirical-timing-uncertainty-seconds "$EMPIRICAL_TIMING_UNCERTAINTY_SECONDS"
      )
    else
      (
        cd "$TASK_CODE_DIR"
        export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli injection-candidate-rank \
        --injection-triggers "$injection_root/score/injection_triggers.jsonl" \
        --candidates "$injection_calibrated" \
        --output-dir "$injection_root/rankings" \
        --split val \
        --reference-ifo "$reference_ifo" \
        --second-ifo "$second_ifo" \
        --physical-delay-limit-seconds "$PHYSICAL_DELAY_LIMIT_SECONDS" \
        --empirical-timing-uncertainty-seconds "$EMPIRICAL_TIMING_UNCERTAINTY_SECONDS" \
        --truth-association-window-seconds "$truth_association_window_seconds"
      )
    fi
  fi

  if [[ "$detector_set_mode" == 1 ]]; then
    background_search="$background_root/search/val_detector_set_block_permutation_report.json"
  else
    background_search="$background_root/search/val_candidate_time_slide_report.json"
  fi
  if [[ ! -s "$background_search" ]]; then
    if [[ "$detector_set_mode" == 1 ]]; then
      (
        cd "$TASK_CODE_DIR"
        export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
        "$TASK_PYTHON" -m gwyolo.cli detector-set-block-permutations \
          --candidates "$background_calibrated" \
          --background-manifest "$BACKGROUND_MANIFEST" \
          --schedule "$BLOCK_SCHEDULE" \
          --output-dir "$background_root/search" \
          --empirical-timing-uncertainty-seconds "$EMPIRICAL_TIMING_UNCERTAINTY_SECONDS" \
          --cluster-window-seconds "$cluster_window_seconds"
      )
    else
      (
        cd "$TASK_CODE_DIR"
        export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
        "$TASK_PYTHON" -m gwyolo.cli candidate-block-permutations \
          --candidates "$background_calibrated" \
          --background-manifest "$BACKGROUND_MANIFEST" \
          --schedule "$BLOCK_SCHEDULE" \
          --output-dir "$background_root/search" \
          --split val \
          --reference-ifo "$reference_ifo" \
          --shifted-ifo "$second_ifo" \
          --coincidence-window-seconds "$COINCIDENCE_WINDOW_SECONDS" \
          --cluster-window-seconds "$cluster_window_seconds" \
          --physical-delay-limit-seconds "$PHYSICAL_DELAY_LIMIT_SECONDS" \
          --empirical-timing-uncertainty-seconds "$EMPIRICAL_TIMING_UNCERTAINTY_SECONDS"
      )
    fi
  fi

  receipt="$scenario_root/scenario_receipt.json"
  if [[ ! -s "$receipt" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
      "$TASK_PYTHON" -m gwyolo.cli calibration-perturbation-scenario-freeze \
        --plan "$CALIBRATION_PLAN" \
        --background-score-report "$background_score" \
        --injection-score-report "$injection_score" \
        --background-timing-application-report "$background_calibrated.report.json" \
        --injection-timing-application-report "$injection_calibrated.report.json" \
        --background-search-report "$background_search" \
        --injection-ranking-report "$ranking_report" \
        --output "$receipt"
    )
  fi
  receipt_args+=(--scenario-receipt "$receipt")
done

result="$OUTPUT_ROOT/calibration_robustness.json"
if [[ ! -s "$result" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli calibration-perturbation-evaluate \
      --plan "$CALIBRATION_PLAN" \
      --baseline-calibration-report "$BASELINE_CALIBRATION_REPORT" \
      "${receipt_args[@]}" \
      --config "$robustness_config" \
      --output "$result"
  )
fi

"$TASK_PYTHON" - "$result" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_validation_calibration_perturbation_robustness"
    or report.get("scenario_count", 0) < 7
    or report.get("scenario_threshold_refits") != 0
    or report.get("test_rows_read") != 0
):
    raise SystemExit("calibration robustness aggregate is incomplete")
PY
