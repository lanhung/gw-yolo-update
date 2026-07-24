#!/usr/bin/env bash
set -euo pipefail

# Validation-only paired raw/mask successor over the frozen numeric
# H1/L1/V1 detector bank. The 1000/year operating point is a detector-set
# robustness diagnostic and cannot be promoted as the final search FAR.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  DETECTOR_BASELINE_RECEIPT
  MASK_VALIDATION_RECEIPT
  MASK_TIMING_RECEIPT
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
  NETWORK_CONFIG
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required numeric raw/mask variable is unset: $variable" >&2
    exit 2
  fi
done
adapter_config=${ADAPTER_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_finetune_glitch_adapter.yaml}
for path in \
  "$TASK_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$DETECTOR_BASELINE_RECEIPT" \
  "$MASK_VALIDATION_RECEIPT" \
  "$MASK_TIMING_RECEIPT" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$COHERENCE_CONFIG" \
  "$NETWORK_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "numeric raw/mask detector-set input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "numeric raw/mask checkout differs from its declared commit" >&2
  exit 3
fi

policy="$TASK_CODE_DIR/configs/detector_stratified_candidate_calibration.yaml"
if [[ ! -s "$policy" ]]; then
  echo "numeric raw/mask calibration policy is absent" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"

mapfile -t settings < <(
  "$TASK_PYTHON" - \
    "$FIVE_SEED_SUMMARY" \
    "$DETECTOR_BASELINE_RECEIPT" \
    "$MASK_VALIDATION_RECEIPT" \
    "$MASK_TIMING_RECEIPT" \
    "$BACKGROUND_MANIFEST" \
    "$INJECTION_MANIFEST" \
    "$UNIFORM_CONFIG" \
    "$FAMILY_BALANCED_CONFIG" \
    "$adapter_config" \
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
    baseline_path,
    validation_path,
    timing_path,
    background_path,
    injection_path,
    uniform_path,
    family_path,
    adapter_path,
    policy_path,
) = sys.argv[1:]
summary = json.loads(pathlib.Path(summary_path).read_text(encoding="utf-8"))
baseline = json.loads(pathlib.Path(baseline_path).read_text(encoding="utf-8"))
validation = json.loads(
    pathlib.Path(validation_path).read_text(encoding="utf-8")
)
timing = json.loads(pathlib.Path(timing_path).read_text(encoding="utf-8"))
pipeline_path = pathlib.Path(
    validation["artifacts"]["pipeline_report"]["path"]
).resolve()
pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
policy = yaml.safe_load(
    pathlib.Path(policy_path).read_text(encoding="utf-8")
)["detector_stratified_candidate_calibration"]
if (
    summary.get("status")
    != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("five_seed_stability", {}).get("passed") is not True
    or summary.get("test_data_opened") is not False
    or baseline.get("status")
    != "completed_detector_stratified_candidate_calibration_baseline"
    or baseline.get("passed") is not True
    or baseline.get("scientific_claim_allowed") is not False
    or baseline.get("final_search_far_claim_allowed") is not False
    or baseline.get("inputs", {})
    .get("background_manifest", {})
    .get("sha256")
    != digest(background_path)
    or baseline.get("inputs", {})
    .get("injection_manifest", {})
    .get("sha256")
    != digest(injection_path)
    or baseline.get("robustness_only_target_far_per_year")
    != float(policy["target_far_per_year"])
    or validation.get("status")
    != "completed_validation_only_mask_deglitch_gate"
    or validation.get("execution_passed") is not True
    or validation.get("development_gates_passed") is not True
    or validation.get("locked_test_allowed") is not False
    or validation.get("artifacts", {})
    .get("pipeline_report", {})
    .get("sha256")
    != digest(pipeline_path)
    or timing.get("status")
    != "completed_validation_only_mask_timing_gate"
    or timing.get("coherent_background_scale_allowed") is not True
    or timing.get("mask_validation_receipt_sha256")
    != digest(validation_path)
    or pipeline.get("status")
    != "validation_only_end_to_end_mask_search_pipeline"
    or pipeline.get("development_gates_passed") is not True
):
    raise SystemExit("numeric raw/mask detector-set preflight failed")
arm = summary["promoted_arm"]
if arm == "uniform":
    config = pathlib.Path(uniform_path)
elif arm == "family_balanced":
    config = pathlib.Path(family_path)
elif arm == "glitch_adapter":
    config = pathlib.Path(adapter_path)
else:
    raise SystemExit("five-seed summary selected an unknown arm")
checkpoint = pathlib.Path(summary["selected_checkpoint_path"])
if (
    not checkpoint.is_file()
    or digest(checkpoint) != summary.get("selected_checkpoint_sha256")
    or digest(config)
    != summary.get("common_artifact_hashes", {}).get("config_file_sha256")
):
    raise SystemExit("numeric raw/mask model replay failed")
schedule = pathlib.Path(baseline["block_schedule"]["path"]).resolve()
if (
    not schedule.is_file()
    or digest(schedule) != baseline["block_schedule"]["sha256"]
):
    raise SystemExit("numeric raw/mask block schedule replay failed")
print(checkpoint)
print(config)
print(schedule)
print(pipeline["strength"])
for key in (
    "target_far_per_year",
    "bootstrap_replicates",
    "seed",
    "minimum_timing_matches",
    "maximum_timing_uncertainty_seconds",
    "chirp_threshold",
    "minimum_bins",
    "cluster_window_seconds",
):
    print(policy[key])
print(
    pipeline["minimum_contaminated_efficiency_gain"]
)
PY
)
if (( ${#settings[@]} != 13 )); then
  echo "numeric raw/mask preflight did not resolve 13 settings" >&2
  exit 3
fi
checkpoint=${settings[0]}
model_config=${settings[1]}
schedule=${settings[2]}
strength=${settings[3]}
target_far=${settings[4]}
bootstrap_replicates=${settings[5]}
seed=${settings[6]}
minimum_timing_matches=${settings[7]}
maximum_timing_uncertainty=${settings[8]}
chirp_threshold=${settings[9]}
minimum_bins=${settings[10]}
cluster_window=${settings[11]}
minimum_gain=${settings[12]}

while :; do
  gpu_pids=$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true
  )
  [[ -z "$gpu_pids" ]] && break
  sleep 30
done

cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
model_ifos=(H1 L1 V1)
q_values=(4 8 16)

raw_background="$OUTPUT_ROOT/raw/background_score"
raw_injection="$OUTPUT_ROOT/raw/injection_score"
if [[ ! -s "$raw_background/trigger_score_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli trigger-score \
    --manifest "$BACKGROUND_MANIFEST" \
    --checkpoint "$checkpoint" \
    --config "$model_config" \
    --output-dir "$raw_background" \
    --model-ifos "${model_ifos[@]}" \
    --q-values "${q_values[@]}" \
    --target-sample-rate 1024 \
    --context-duration 64 \
    --save-probabilities \
    --required-split val \
    --coherence-config "$COHERENCE_CONFIG"
fi
if [[ ! -s "$raw_injection/injection_score_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli injection-score \
    --manifest "$INJECTION_MANIFEST" \
    --checkpoint "$checkpoint" \
    --config "$model_config" \
    --output-dir "$raw_injection" \
    --model-ifos "${model_ifos[@]}" \
    --q-values "${q_values[@]}" \
    --target-sample-rate 1024 \
    --save-probabilities \
    --required-split val \
    --coherence-config "$COHERENCE_CONFIG"
fi

background_cleaning="$OUTPUT_ROOT/mask/background_cleaning"
injection_cleaning="$OUTPUT_ROOT/mask/injection_cleaning"
if [[ ! -s "$background_cleaning/learned_background_deglitch_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli learned-background-deglitch \
    --background-manifest "$BACKGROUND_MANIFEST" \
    --scored-manifest "$raw_background/background_triggers.jsonl" \
    --output-dir "$background_cleaning" \
    --strength "$strength" \
    --model-ifos "${model_ifos[@]}" \
    --target-sample-rate 1024 \
    --context-duration 64 \
    --required-split val
fi
if [[ ! -s "$injection_cleaning/learned_deglitch_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli learned-deglitch \
    --materialized-manifest "$INJECTION_MANIFEST" \
    --scored-manifest "$raw_injection/injection_triggers.jsonl" \
    --output-dir "$injection_cleaning" \
    --strength "$strength"
fi

mask_background="$OUTPUT_ROOT/mask/background_score"
mask_injection="$OUTPUT_ROOT/mask/injection_score"
if [[ ! -s "$mask_background/trigger_score_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli trigger-score \
    --manifest "$background_cleaning/learned_background_deglitch.jsonl" \
    --checkpoint "$checkpoint" \
    --config "$model_config" \
    --output-dir "$mask_background" \
    --model-ifos "${model_ifos[@]}" \
    --q-values "${q_values[@]}" \
    --target-sample-rate 1024 \
    --context-duration 64 \
    --required-split val \
    --coherence-config "$COHERENCE_CONFIG"
fi
if [[ ! -s "$mask_injection/injection_score_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli injection-score \
    --manifest "$injection_cleaning/learned_deglitch.jsonl" \
    --checkpoint "$checkpoint" \
    --config "$model_config" \
    --output-dir "$mask_injection" \
    --model-ifos "${model_ifos[@]}" \
    --q-values "${q_values[@]}" \
    --target-sample-rate 1024 \
    --required-split val \
    --coherence-config "$COHERENCE_CONFIG"
fi

rankings=()
calibrations=()
timings=()
for arm in raw mask; do
  background_score="$OUTPUT_ROOT/$arm/background_score/background_triggers.jsonl"
  injection_score="$OUTPUT_ROOT/$arm/injection_score/injection_triggers.jsonl"
  background_candidates="$OUTPUT_ROOT/$arm/background_candidates"
  injection_candidates="$OUTPUT_ROOT/$arm/injection_candidates"
  timing="$OUTPUT_ROOT/$arm/candidate_timing_calibration.json"
  calibrated_background="$OUTPUT_ROOT/$arm/background_candidates_calibrated.jsonl"
  calibrated_injection="$OUTPUT_ROOT/$arm/injection_candidates_calibrated.jsonl"
  if [[ ! -s "$background_candidates/candidate_extraction_report.json" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli candidate-extract \
      --triggers "$background_score" \
      --output-dir "$background_candidates" \
      --chirp-threshold "$chirp_threshold" \
      --minimum-bins "$minimum_bins"
  fi
  if [[ ! -s "$injection_candidates/injection_candidate_extraction_report.json" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli injection-candidate-extract \
      --injection-triggers "$injection_score" \
      --output-dir "$injection_candidates" \
      --chirp-threshold "$chirp_threshold" \
      --minimum-bins "$minimum_bins"
  fi
  if [[ ! -s "$timing" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli candidate-timing-calibrate \
      --injection-triggers "$injection_score" \
      --output "$timing" \
      --chirp-threshold "$chirp_threshold" \
      --minimum-bins "$minimum_bins" \
      --association-window-seconds 0.25 \
      --uncertainty-quantile 0.99 \
      --minimum-matches-per-method "$minimum_timing_matches" \
      --maximum-empirical-timing-uncertainty-seconds \
        "$maximum_timing_uncertainty"
  fi
  if [[ ! -s "$calibrated_background" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli candidate-timing-apply \
      --candidates "$background_candidates/single_ifo_candidates.jsonl" \
      --calibration-report "$timing" \
      --output "$calibrated_background"
  fi
  if [[ ! -s "$calibrated_injection" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli candidate-timing-apply \
      --candidates \
        "$injection_candidates/single_ifo_injection_candidates.jsonl" \
      --calibration-report "$timing" \
      --output "$calibrated_injection"
  fi
  uncertainty=$(
    "$TASK_PYTHON" - "$timing" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
method = "local_whitened_strain_envelope_per_mask_cluster_v1"
values = report["methods"][method]
if values.get("calibration_gate_passed") is not True:
    raise SystemExit("numeric raw/mask timing gate failed")
print(values["empirical_timing_uncertainty_seconds"])
PY
  )
  block="$OUTPUT_ROOT/$arm/detector_set_block_background"
  if [[ ! -s "$block/val_detector_set_block_permutation_report.json" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli detector-set-block-permutations \
      --candidates "$calibrated_background" \
      --background-manifest "$BACKGROUND_MANIFEST" \
      --schedule "$schedule" \
      --output-dir "$block" \
      --empirical-timing-uncertainty-seconds "$uncertainty" \
      --cluster-window-seconds "$cluster_window"
  fi
  ranking="$OUTPUT_ROOT/$arm/detector_set_injection_rankings"
  ranking_report="$ranking/val_variable_detector_set_injection_candidate_ranking_report.json"
  if [[ ! -s "$ranking_report" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli detector-set-injection-candidate-rank \
      --injection-triggers "$injection_score" \
      --candidates "$calibrated_injection" \
      --config "$NETWORK_CONFIG" \
      --output-dir "$ranking" \
      --split val \
      --empirical-timing-uncertainty-seconds "$uncertainty"
  fi
  calibration="$OUTPUT_ROOT/$arm/frozen_validation_candidate_search_calibration.json"
  if [[ ! -s "$calibration" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibrate \
      --validation-time-slide-report \
        "$block/val_detector_set_block_permutation_report.json" \
      --validation-background-manifest "$BACKGROUND_MANIFEST" \
      --validation-injection-ranking-report "$ranking_report" \
      --target-far-per-year "$target_far" \
      --output "$calibration" \
      --bootstrap-replicates "$bootstrap_replicates" \
      --seed "$seed"
  fi
  rankings+=("$ranking_report")
  calibrations+=("$calibration")
  timings+=("$timing")
done

successor="$OUTPUT_ROOT/numeric_raw_mask_detector_set_ranking_successor.json"
if [[ ! -s "$successor" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli \
    numeric-raw-mask-detector-set-ranking-successor-freeze \
    --mask-validation-receipt "$MASK_VALIDATION_RECEIPT" \
    --mask-timing-receipt "$MASK_TIMING_RECEIPT" \
    --raw-variable-ranking-report "${rankings[0]}" \
    --mask-variable-ranking-report "${rankings[1]}" \
    --raw-timing-report "${timings[0]}" \
    --mask-timing-report "${timings[1]}" \
    --background-deglitch-report \
      "$background_cleaning/learned_background_deglitch_report.json" \
    --injection-deglitch-report \
      "$injection_cleaning/learned_deglitch_report.json" \
    --network-config "$NETWORK_CONFIG" \
    --output "$successor"
fi

comparison="$OUTPUT_ROOT/paired_validation_candidate_comparison.json"
if [[ ! -s "$comparison" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-compare \
    --raw-calibration-report "${calibrations[0]}" \
    --mask-calibration-report "${calibrations[1]}" \
    --mask-validation-receipt "$MASK_VALIDATION_RECEIPT" \
    --mask-timing-receipt "$MASK_TIMING_RECEIPT" \
    --detector-set-ranking-successor "$successor" \
    --minimum-absolute-weighted-efficiency-gain "$minimum_gain" \
    --bootstrap-replicates "$bootstrap_replicates" \
    --seed "$seed" \
    --output "$comparison"
fi

"$TASK_PYTHON" - \
  "$DETECTOR_BASELINE_RECEIPT" \
  "$MASK_VALIDATION_RECEIPT" \
  "$MASK_TIMING_RECEIPT" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$successor" \
  "${calibrations[0]}" \
  "${calibrations[1]}" \
  "$comparison" \
  "$GWYOLO_CODE_COMMIT" \
  "$OUTPUT_ROOT/numeric_raw_mask_detector_set_successor_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


paths = [pathlib.Path(value).resolve() for value in sys.argv[1:10]]
commit = sys.argv[10]
output = pathlib.Path(sys.argv[11])
if output.exists():
    raise SystemExit("numeric raw/mask successor receipts are immutable")
(
    baseline_path,
    validation_path,
    timing_path,
    background_path,
    injection_path,
    successor_path,
    raw_path,
    mask_path,
    comparison_path,
) = paths
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
successor = json.loads(successor_path.read_text(encoding="utf-8"))
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
calibrations = {
    "raw": json.loads(raw_path.read_text(encoding="utf-8")),
    "mask": json.loads(mask_path.read_text(encoding="utf-8")),
}
if (
    baseline.get("status")
    != "completed_detector_stratified_candidate_calibration_baseline"
    or baseline.get("passed") is not True
    or baseline.get("inputs", {})
    .get("background_manifest", {})
    .get("sha256")
    != digest(background_path)
    or baseline.get("inputs", {})
    .get("injection_manifest", {})
    .get("sha256")
    != digest(injection_path)
    or successor.get("status")
    != "numeric_variable_detector_set_raw_mask_ranking_successor_v1"
    or successor.get("scientific_claim_allowed") is not False
    or successor.get("test_rows_read") != 0
    or comparison.get("status")
    != "validation_only_paired_raw_mask_candidate_calibration_comparison"
    or comparison.get("scientific_claim_allowed") is not False
    or comparison.get("locked_test_allowed") is not False
    or comparison.get("test_rows_read") != 0
    or any(
        report.get("status")
        != "frozen_validation_candidate_search_calibration"
        or report.get("publication_calibration_eligible") is not True
        or report.get("slide_schedule_audit", {}).get("schedule_kind")
        != "variable_detector_set_block_permutation"
        for report in calibrations.values()
    )
):
    raise SystemExit("numeric raw/mask detector-set successor failed replay")
receipt = {
    "status": "completed_validation_only_numeric_detector_set_raw_mask_successor",
    "passed": comparison.get("passed") is True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "final_search_far_claim_allowed": False,
    "locked_test_allowed": False,
    "test_rows_read": 0,
    "test_evaluation": None,
    "purpose": (
        "four-detector-subset numeric validation robustness at 1000/year; "
        "not the final continuous-search FAR"
    ),
    "code_commit": commit,
    "inputs": {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in {
            "detector_baseline_receipt": baseline_path,
            "mask_validation_receipt": validation_path,
            "mask_timing_receipt": timing_path,
            "background_manifest": background_path,
            "injection_manifest": injection_path,
        }.items()
    },
    "ranking_successor": {
        "path": str(successor_path),
        "sha256": digest(successor_path),
    },
    "calibrations": {
        "raw": {"path": str(raw_path), "sha256": digest(raw_path)},
        "mask": {"path": str(mask_path), "sha256": digest(mask_path)},
    },
    "paired_comparison": {
        "path": str(comparison_path),
        "sha256": digest(comparison_path),
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
part = output.with_suffix(output.suffix + ".part")
part.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(part, output)
PY
