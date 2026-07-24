#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_RAW_MASK_RECEIPT
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
  "$SOURCE_RAW_MASK_RECEIPT" \
  "$NETWORK_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "raw/mask detector-set input is absent: $input" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD)" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "raw/mask detector-set checkout differs from its declared commit" >&2
  exit 2
fi
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT

mapfile -t settings < <(
  "$TASK_PYTHON" - "$SOURCE_RAW_MASK_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    receipt.get("status")
    != "completed_validation_only_raw_mask_continuous_background"
    or receipt.get("scientific_claim_allowed") is not False
    or int(receipt.get("test_rows_read", -1)) != 0
):
    raise SystemExit("source raw/mask receipt has the wrong contract")
merge_path = pathlib.Path(receipt["merge_report"]["path"]).resolve()
merge = json.loads(merge_path.read_text(encoding="utf-8"))
timing_path = pathlib.Path(
    receipt["mask_timing_receipt"]["path"]
).resolve()
timing = json.loads(timing_path.read_text(encoding="utf-8"))
validation_path = pathlib.Path(
    receipt["mask_validation_receipt"]["path"]
).resolve()
authorization_path = pathlib.Path(
    receipt["inputs"]["background_plan_authorization"]["path"]
).resolve()
authorization = json.loads(
    authorization_path.read_text(encoding="utf-8")
)
comparison = json.loads(
    pathlib.Path(
        receipt["paired_validation_comparison"]["path"]
    ).read_text(encoding="utf-8")
)
if (
    merge.get("status")
    != "verified_merged_streamed_raw_mask_candidate_background"
    or merge.get("complete_parent_plan") is not True
    or int(merge.get("split_counts", {}).get("test", -1)) != 0
    or authorization.get("status")
    != "authorized_validation_candidate_continuous_background_plan"
    or authorization.get("passed") is not True
    or timing.get("status")
    != "completed_validation_only_mask_timing_gate"
):
    raise SystemExit("source raw/mask lineage failed replay")
print(merge_path)
print(merge["background_manifest_path"])
print(authorization_path)
print(timing_path)
print(validation_path)
print(authorization["authorization_identity"]["target_far_per_year"])
print(authorization["authorization_identity"]["zero_count_confidence"])
print(
    comparison["continuous_background_mask_gain_gate"][
        "minimum_absolute_weighted_efficiency_gain"
    ]
)
print(comparison["bootstrap_replicates"])
print(comparison["seed"])
method = timing["required_method"]
for arm in ("raw", "mask"):
    arm_merge_path = pathlib.Path(
        merge["arm_merges"][arm]["report_path"]
    ).resolve()
    arm_merge = json.loads(arm_merge_path.read_text(encoding="utf-8"))
    candidate = pathlib.Path(
        arm_merge["candidate_manifests"]["val"]["path"]
    ).resolve()
    source_ranking = pathlib.Path(
        timing["injection_ranking_reports"][arm]["path"]
    ).resolve()
    score_path = pathlib.Path(
        timing[f"{arm}_score_report"]["path"]
    ).resolve()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    trigger = pathlib.Path(score["triggers_path"]).resolve()
    candidate_report = pathlib.Path(
        timing["injection_ranking_reports"][arm][
            "candidate_extraction_report_path"
        ]
    ).resolve()
    injection_candidate = (
        candidate_report.parent.parent
        / f"{arm}_injection_candidates_calibrated.jsonl"
    ).resolve()
    uncertainty = timing["timing_reports"][arm]["report"]["methods"][
        method
    ]["empirical_timing_uncertainty_seconds"]
    for value in (
        arm_merge_path,
        candidate,
        source_ranking,
        trigger,
        injection_candidate,
        uncertainty,
    ):
        print(value)
PY
)
if (( ${#settings[@]} != 22 )); then
  echo "source raw/mask receipt did not resolve 22 settings" >&2
  exit 2
fi
merge_report=${settings[0]}
background_manifest=${settings[1]}
authorization=${settings[2]}
timing_receipt=${settings[3]}
validation_receipt=${settings[4]}
target_far=${settings[5]}
zero_count_confidence=${settings[6]}
minimum_gain=${settings[7]}
comparison_replicates=${settings[8]}
comparison_seed=${settings[9]}
for index in 0 1 2 3 4 10 11 12 13 14 16 17 18 19 20; do
  if [[ ! -f "${settings[$index]}" ]]; then
    echo "resolved raw/mask artifact is absent: ${settings[$index]}" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"
schedule="$OUTPUT_ROOT/detector_set_block_permutation_schedule.json"
if [[ ! -s "$schedule" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli \
    detector-set-block-permutation-schedule-freeze \
    --background-manifest "$background_manifest" \
    --network-config "$NETWORK_CONFIG" \
    --output "$schedule" \
    --split val \
    --target-far-per-year "$target_far" \
    --zero-count-confidence "$zero_count_confidence"
fi

variable_rankings=()
calibrations=()
for arm_index in 0 1; do
  if (( arm_index == 0 )); then
    arm=raw
    offset=10
  else
    arm=mask
    offset=16
  fi
  background_candidates=${settings[$((offset + 1))]}
  injection_triggers=${settings[$((offset + 3))]}
  injection_candidates=${settings[$((offset + 4))]}
  uncertainty=${settings[$((offset + 5))]}
  block_dir="$OUTPUT_ROOT/$arm/detector_set_block_background"
  ranking_dir="$OUTPUT_ROOT/$arm/detector_set_injection_rankings"
  "$TASK_PYTHON" -m gwyolo.cli detector-set-block-permutations \
    --candidates "$background_candidates" \
    --background-manifest "$background_manifest" \
    --schedule "$schedule" \
    --output-dir "$block_dir" \
    --empirical-timing-uncertainty-seconds "$uncertainty" \
    --cluster-window-seconds 0.1
  "$TASK_PYTHON" -m gwyolo.cli detector-set-injection-candidate-rank \
    --injection-triggers "$injection_triggers" \
    --candidates "$injection_candidates" \
    --config "$NETWORK_CONFIG" \
    --output-dir "$ranking_dir" \
    --split val \
    --empirical-timing-uncertainty-seconds "$uncertainty"
  ranking_report="$ranking_dir/val_variable_detector_set_injection_candidate_ranking_report.json"
  variable_rankings+=("$ranking_report")
  calibration="$OUTPUT_ROOT/$arm/frozen_validation_candidate_search_calibration.json"
  if [[ ! -s "$calibration" ]]; then
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibrate \
      --validation-time-slide-report \
        "$block_dir/val_detector_set_block_permutation_report.json" \
      --validation-background-manifest "$background_manifest" \
      --validation-injection-ranking-report "$ranking_report" \
      --target-far-per-year "$target_far" \
      --output "$calibration" \
      --bootstrap-replicates 10000 \
      --seed 20260720
  fi
  calibrations+=("$calibration")
done

ranking_successor="$OUTPUT_ROOT/raw_mask_detector_set_ranking_successor.json"
if [[ ! -s "$ranking_successor" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli \
    raw-mask-detector-set-ranking-successor-freeze \
    --mask-timing-receipt "$timing_receipt" \
    --raw-variable-ranking-report "${variable_rankings[0]}" \
    --mask-variable-ranking-report "${variable_rankings[1]}" \
    --network-config "$NETWORK_CONFIG" \
    --output "$ranking_successor"
fi

comparison="$OUTPUT_ROOT/paired_validation_candidate_comparison.json"
if [[ ! -s "$comparison" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-compare \
    --raw-calibration-report "${calibrations[0]}" \
    --mask-calibration-report "${calibrations[1]}" \
    --mask-validation-receipt "$validation_receipt" \
    --mask-timing-receipt "$timing_receipt" \
    --detector-set-ranking-successor "$ranking_successor" \
    --minimum-absolute-weighted-efficiency-gain "$minimum_gain" \
    --bootstrap-replicates "$comparison_replicates" \
    --seed "$comparison_seed" \
    --output "$comparison"
fi

binding="$OUTPUT_ROOT/bound_validation_raw_mask_continuous_background_evidence.json"
if [[ ! -s "$binding" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli \
    candidate-search-raw-mask-endpoint-bind \
    --raw-mask-background-receipt "$SOURCE_RAW_MASK_RECEIPT" \
    --raw-calibration-report "${calibrations[0]}" \
    --mask-calibration-report "${calibrations[1]}" \
    --paired-comparison-report "$comparison" \
    --output "$binding"
fi

"$TASK_PYTHON" - "$binding" "$ranking_successor" "$merge_report" <<'PY'
import json
import pathlib
import sys

binding = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
successor = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
merge = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
if (
    binding.get("status")
    != "bound_validation_raw_mask_continuous_background_evidence"
    or binding.get("passed") is not True
    or successor.get("status")
    != "variable_detector_set_raw_mask_ranking_successor_v1"
    or merge.get("complete_parent_plan") is not True
    or any(
        binding.get("background_dependence_audits", {})
        .get(arm, {})
        .get("status")
        != "detector_set_candidate_background_dependence_audit_v1"
        for arm in ("raw", "mask")
    )
):
    raise SystemExit("raw/mask detector-set successor failed publication replay")
PY
