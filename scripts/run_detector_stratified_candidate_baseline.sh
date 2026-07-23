#!/usr/bin/env bash
set -euo pipefail

# Build a validation-only variable-detector candidate baseline from the frozen
# numeric H1/L1/V1 bank. Its 1000/year operating point is used only for
# detector-set and calibration robustness, never as the final search FAR.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  BACKGROUND_CHAIN_RECEIPT
  BACKGROUND_MANIFEST
  PHYSICAL_MATERIALIZATION_AUDIT
  INJECTION_MANIFEST
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
  NETWORK_CONFIG
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector-baseline variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$BACKGROUND_CHAIN_RECEIPT" \
  "$BACKGROUND_MANIFEST" \
  "$PHYSICAL_MATERIALIZATION_AUDIT" \
  "$INJECTION_MANIFEST" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$COHERENCE_CONFIG" \
  "$NETWORK_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "detector-baseline input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector-baseline checkout differs from its declared commit" >&2
  exit 3
fi

policy="$TASK_CODE_DIR/configs/detector_stratified_candidate_calibration.yaml"
if [[ ! -s "$policy" ]]; then
  echo "detector-baseline policy is absent: $policy" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"

mapfile -t settings < <(
  "$TASK_PYTHON" - \
    "$FIVE_SEED_SUMMARY" \
    "$BACKGROUND_CHAIN_RECEIPT" \
    "$BACKGROUND_MANIFEST" \
    "$PHYSICAL_MATERIALIZATION_AUDIT" \
    "$INJECTION_MANIFEST" \
    "$UNIFORM_CONFIG" \
    "$FAMILY_BALANCED_CONFIG" \
    "$policy" <<'PY'
import hashlib
import json
import pathlib
import sys

import yaml


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    summary_path,
    chain_path,
    background_path,
    physical_path,
    injection_path,
    uniform_path,
    family_path,
    policy_path,
) = sys.argv[1:]
summary = json.loads(pathlib.Path(summary_path).read_text(encoding="utf-8"))
chain = json.loads(pathlib.Path(chain_path).read_text(encoding="utf-8"))
physical = json.loads(pathlib.Path(physical_path).read_text(encoding="utf-8"))
policy = yaml.safe_load(
    pathlib.Path(policy_path).read_text(encoding="utf-8")
)["detector_stratified_candidate_calibration"]
required = set(policy["required_detector_subsets"])
minimum_background = int(
    policy["minimum_background_windows_per_detector_subset"]
)
minimum_injections = int(policy["minimum_injections_per_detector_subset"])
background_rows = [
    json.loads(line)
    for line in pathlib.Path(background_path)
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
injection_rows = [
    json.loads(line)
    for line in pathlib.Path(injection_path)
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]
background_counts = {}
numeric_banks = {}
for row in background_rows:
    subset = "+".join(row.get("ifos", []))
    background_counts[subset] = background_counts.get(subset, 0) + 1
    bank = row.get("background_bank", {})
    path = pathlib.Path(str(bank.get("path", "")))
    if (
        row.get("split") != "val"
        or row.get("physical_signal_present") is not False
        or not path.is_file()
        or digest(path) != bank.get("sha256")
    ):
        raise SystemExit("numeric detector background replay failed")
    numeric_banks[str(path.resolve())] = bank["sha256"]
injection_counts = {}
for row in injection_rows:
    subset = "+".join(row.get("ifos", []))
    injection_counts[subset] = injection_counts.get(subset, 0) + 1
    if row.get("split") != "val":
        raise SystemExit("detector injection corpus is not validation-only")
if (
    summary.get("status")
    != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("five_seed_stability", {}).get("passed") is not True
    or summary.get("test_data_opened") is not False
    or chain.get("status")
    != "verified_detector_stratified_validation_data_chain"
    or chain.get("passed") is not True
    or chain.get("candidate_scores_inspected") is not False
    or int(chain.get("test_rows_read", -1)) != 0
    or pathlib.Path(chain["background_report_path"]).resolve().parent
    != pathlib.Path(background_path).resolve().parent
    or physical.get("status")
    != "verified_detector_stratified_physical_injection_materialization"
    or physical.get("passed") is not True
    or physical.get("publication_calibration_eligible") is not True
    or physical.get("manifest_sha256") != digest(injection_path)
    or physical.get("candidate_scores_inspected") is not False
    or int(physical.get("test_rows_read", -1)) != 0
    or set(background_counts) != required
    or set(injection_counts) != required
    or any(
        background_counts[name] < minimum_background for name in required
    )
    or any(
        injection_counts[name] < minimum_injections for name in required
    )
    or {row["gps_block"] for row in background_rows}
    & {row["gps_block"] for row in injection_rows}
):
    raise SystemExit("detector-stratified baseline preflight failed")
arm = summary["promoted_arm"]
checkpoint = pathlib.Path(summary["selected_checkpoint_path"])
if arm == "uniform":
    config = pathlib.Path(uniform_path)
elif arm == "family_balanced":
    config = pathlib.Path(family_path)
else:
    raise SystemExit("five-seed summary selected an unknown arm")
if (
    not checkpoint.is_file()
    or digest(checkpoint) != summary.get("selected_checkpoint_sha256")
    or digest(config)
    != summary.get("common_artifact_hashes", {}).get("config_file_sha256")
):
    raise SystemExit("five-seed checkpoint/config replay failed")
print(checkpoint)
print(config)
for key in (
    "target_far_per_year",
    "zero_count_confidence",
    "exposure_safety_factor",
    "bootstrap_replicates",
    "seed",
    "minimum_timing_matches",
    "maximum_timing_uncertainty_seconds",
    "chirp_threshold",
    "minimum_bins",
    "slide_count",
    "slide_step_seconds",
    "cluster_window_seconds",
):
    print(policy[key])
PY
)
if (( ${#settings[@]} != 14 )); then
  echo "detector-baseline preflight did not resolve 14 settings" >&2
  exit 3
fi
checkpoint=${settings[0]}
model_config=${settings[1]}
target_far=${settings[2]}
zero_count_confidence=${settings[3]}
exposure_safety_factor=${settings[4]}
bootstrap_replicates=${settings[5]}
seed=${settings[6]}
minimum_timing_matches=${settings[7]}
maximum_timing_uncertainty=${settings[8]}
chirp_threshold=${settings[9]}
minimum_bins=${settings[10]}
slide_count=${settings[11]}
slide_step=${settings[12]}
cluster_window=${settings[13]}

pipeline_root="$OUTPUT_ROOT/pipeline"
pipeline_report="$pipeline_root/candidate_validation_pipeline_report.json"
if [[ ! -s "$pipeline_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-validation-pipeline \
      --background-manifest "$BACKGROUND_MANIFEST" \
      --injection-manifest "$INJECTION_MANIFEST" \
      --checkpoint "$checkpoint" \
      --config "$model_config" \
      --coherence-config "$COHERENCE_CONFIG" \
      --output-dir "$pipeline_root" \
      --chirp-threshold "$chirp_threshold" \
      --minimum-bins "$minimum_bins" \
      --timing-association-window-seconds 0.25 \
      --timing-uncertainty-quantile 0.99 \
      --minimum-timing-matches "$minimum_timing_matches" \
      --maximum-timing-uncertainty-seconds "$maximum_timing_uncertainty" \
      --truth-association-window-seconds 0.25 \
      --slide-count "$slide_count" \
      --slide-step-seconds "$slide_step" \
      --cluster-window-seconds "$cluster_window" \
      --target-far-per-year "$target_far" \
      --bootstrap-replicates "$bootstrap_replicates" \
      --seed "$seed" \
      --model-selection-report "$FIVE_SEED_SUMMARY"
  )
fi

detector_root="$OUTPUT_ROOT/detector-set"
detector_report="$detector_root/candidate_validation_detector_set_block_pipeline_report.json"
if [[ ! -s "$detector_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli \
      candidate-search-validation-detector-set-recalibrate \
      --pipeline-report "$pipeline_report" \
      --background-manifest "$BACKGROUND_MANIFEST" \
      --calibrated-background-candidate-manifest \
        "$pipeline_root/background_candidates_calibrated.jsonl" \
      --injection-trigger-manifest \
        "$pipeline_root/injection_score/injection_triggers.jsonl" \
      --calibrated-injection-candidate-manifest \
        "$pipeline_root/injection_candidates_calibrated.jsonl" \
      --network-config "$NETWORK_CONFIG" \
      --output-dir "$detector_root" \
      --zero-count-confidence "$zero_count_confidence" \
      --exposure-safety-factor "$exposure_safety_factor"
  )
fi

"$TASK_PYTHON" - \
  "$detector_report" \
  "$detector_root/detector_set_block_permutation_schedule.json" \
  "$detector_root/frozen_candidate_search_calibration.json" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$FIVE_SEED_SUMMARY" \
  "$BACKGROUND_CHAIN_RECEIPT" \
  "$PHYSICAL_MATERIALIZATION_AUDIT" \
  "$policy" \
  "$OUTPUT_ROOT/detector_stratified_candidate_baseline_receipt.json" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

import yaml


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    report_path,
    schedule_path,
    calibration_path,
    background_path,
    injection_path,
    summary_path,
    chain_path,
    physical_path,
    policy_path,
    output_path,
    commit,
) = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
schedule = json.loads(pathlib.Path(schedule_path).read_text(encoding="utf-8"))
calibration = json.loads(
    pathlib.Path(calibration_path).read_text(encoding="utf-8")
)
policy = yaml.safe_load(
    pathlib.Path(policy_path).read_text(encoding="utf-8")
)["detector_stratified_candidate_calibration"]
target_far = float(policy["target_far_per_year"])
if (
    report.get("status")
    != "validation_only_clustered_candidate_search_pipeline"
    or report.get("test_evaluation") is not None
    or report.get("run_identity", {}).get("detector_set_policy")
    != "single_model_explicit_missing_ifo_validity_v1"
    or report.get("run_identity", {}).get("target_far_per_year")
    != target_far
    or schedule.get("status")
    != "frozen_detector_set_block_permutation_schedule"
    or schedule.get("candidate_scores_inspected") is not False
    or schedule.get("schedule_exposure_target_reached") is not True
    or schedule.get("required_detector_subsets_covered") is not True
    or schedule.get("relative_window_slot_policy")
    != "within_block_gps_order_v1"
    or schedule.get("background_manifest_sha256") != digest(background_path)
    or schedule.get("target_far_per_year") != target_far
    or schedule.get("zero_count_confidence")
    != float(policy["zero_count_confidence"])
    or schedule.get("exposure_safety_factor")
    != float(policy["exposure_safety_factor"])
    or calibration.get("status")
    != "frozen_validation_candidate_search_calibration"
    or calibration.get("publication_calibration_eligible") is not True
    or calibration.get("slide_schedule_audit", {}).get("passed") is not True
    or calibration.get("slide_schedule_audit", {}).get("schedule_kind")
    != "variable_detector_set_block_permutation"
    or calibration.get("target_far_per_year") != target_far
):
    raise SystemExit("detector-stratified candidate baseline failed replay")
result = {
    "status": "completed_detector_stratified_candidate_calibration_baseline",
    "passed": True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "test_rows_read": 0,
    "purpose": policy["purpose"],
    "final_search_far_claim_allowed": False,
    "robustness_only_target_far_per_year": target_far,
    "code_commit": commit,
    "inputs": {
        name: {"path": str(pathlib.Path(path).resolve()), "sha256": digest(path)}
        for name, path in {
            "background_manifest": background_path,
            "injection_manifest": injection_path,
            "five_seed_summary": summary_path,
            "background_chain_receipt": chain_path,
            "physical_materialization_audit": physical_path,
            "policy": policy_path,
        }.items()
    },
    "pipeline_report": {
        "path": str(pathlib.Path(report_path).resolve()),
        "sha256": digest(report_path),
    },
    "block_schedule": {
        "path": str(pathlib.Path(schedule_path).resolve()),
        "sha256": digest(schedule_path),
        "gps_blocks": schedule["gps_blocks"],
        "selected_shift_count": schedule["selected_shift_count"],
        "selected_equivalent_live_time_years": schedule[
            "selected_equivalent_live_time_years"
        ],
        "far_resolution_one_count_per_year": schedule[
            "far_resolution_one_count_per_year"
        ],
        "eligible_windows_by_detector_subset": schedule[
            "eligible_windows_by_detector_subset"
        ],
    },
    "baseline_calibration": {
        "path": str(pathlib.Path(calibration_path).resolve()),
        "sha256": digest(calibration_path),
    },
}
target = pathlib.Path(output_path)
part = target.with_suffix(target.suffix + ".part")
part.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY

