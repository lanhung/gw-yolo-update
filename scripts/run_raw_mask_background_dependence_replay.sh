#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_RAW_MASK_RECEIPT
  OUTPUT_ROOT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD)" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "raw/mask dependence replay checkout differs from its frozen commit" >&2
  exit 2
fi
if [[ ! -f "$TASK_PYTHON" || ! -f "$SOURCE_RAW_MASK_RECEIPT" ]]; then
  echo "raw/mask dependence replay source is absent" >&2
  exit 2
fi

mapfile -t settings < <(
  "$TASK_PYTHON" - "$SOURCE_RAW_MASK_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "completed_validation_only_raw_mask_continuous_background":
    raise SystemExit("source raw/mask receipt has the wrong status")
merge_path = pathlib.Path(receipt["merge_report"]["path"])
merge = json.loads(merge_path.read_text(encoding="utf-8"))
raw_path = pathlib.Path(receipt["calibrations"]["raw"]["path"])
mask_path = pathlib.Path(receipt["calibrations"]["mask"]["path"])
raw = json.loads(raw_path.read_text(encoding="utf-8"))
mask = json.loads(mask_path.read_text(encoding="utf-8"))
comparison_path = pathlib.Path(receipt["paired_validation_comparison"]["path"])
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
for value in (
    merge["background_manifest_path"],
    raw["validation_time_slide_report_path"],
    raw["validation_injection_ranking_report_path"],
    mask["validation_time_slide_report_path"],
    mask["validation_injection_ranking_report_path"],
    raw["target_far_per_year"],
    raw["bootstrap_replicates"],
    raw["seed"],
    receipt["mask_validation_receipt"]["path"],
    receipt["mask_timing_receipt"]["path"],
    comparison["continuous_background_mask_gain_gate"][
        "minimum_absolute_weighted_efficiency_gain"
    ],
    comparison["bootstrap_replicates"],
    comparison["seed"],
):
    print(value)
PY
)
if (( ${#settings[@]} != 13 )); then
  echo "source raw/mask receipt did not resolve thirteen replay settings" >&2
  exit 2
fi
background_manifest=${settings[0]}
raw_slide=${settings[1]}
raw_ranking=${settings[2]}
mask_slide=${settings[3]}
mask_ranking=${settings[4]}
target_far=${settings[5]}
calibration_replicates=${settings[6]}
calibration_seed=${settings[7]}
mask_validation=${settings[8]}
mask_timing=${settings[9]}
minimum_gain=${settings[10]}
comparison_replicates=${settings[11]}
comparison_seed=${settings[12]}
for input in \
  "$background_manifest" "$raw_slide" "$raw_ranking" "$mask_slide" \
  "$mask_ranking" "$mask_validation" "$mask_timing"; do
  if [[ ! -f "$input" ]]; then
    echo "raw/mask dependence replay dependency is absent: $input" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT/raw" "$OUTPUT_ROOT/mask"
raw_calibration="$OUTPUT_ROOT/raw/frozen_validation_candidate_search_calibration.json"
mask_calibration="$OUTPUT_ROOT/mask/frozen_validation_candidate_search_calibration.json"
for arm in raw mask; do
  if [[ "$arm" == raw ]]; then
    slide=$raw_slide
    ranking=$raw_ranking
    calibration=$raw_calibration
  else
    slide=$mask_slide
    ranking=$mask_ranking
    calibration=$mask_calibration
  fi
  if [[ ! -s "$calibration" ]]; then
    (
      cd "$TASK_CODE_DIR"
      export PYTHONPATH=src GWYOLO_CODE_COMMIT
      "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibrate \
        --validation-time-slide-report "$slide" \
        --validation-background-manifest "$background_manifest" \
        --validation-injection-ranking-report "$ranking" \
        --target-far-per-year "$target_far" \
        --bootstrap-replicates "$calibration_replicates" \
        --seed "$calibration_seed" \
        --output "$calibration"
    )
  fi
done

comparison="$OUTPUT_ROOT/paired_validation_candidate_comparison.json"
if [[ ! -s "$comparison" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-compare \
      --raw-calibration-report "$raw_calibration" \
      --mask-calibration-report "$mask_calibration" \
      --mask-validation-receipt "$mask_validation" \
      --mask-timing-receipt "$mask_timing" \
      --minimum-absolute-weighted-efficiency-gain "$minimum_gain" \
      --bootstrap-replicates "$comparison_replicates" \
      --seed "$comparison_seed" \
      --output "$comparison"
  )
fi

binding="$OUTPUT_ROOT/bound_validation_raw_mask_continuous_background_evidence.json"
if [[ ! -s "$binding" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-raw-mask-endpoint-bind \
      --raw-mask-background-receipt "$SOURCE_RAW_MASK_RECEIPT" \
      --raw-calibration-report "$raw_calibration" \
      --mask-calibration-report "$mask_calibration" \
      --paired-comparison-report "$comparison" \
      --output "$binding"
  )
fi

"$TASK_PYTHON" - "$binding" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "bound_validation_raw_mask_continuous_background_evidence"
    or report.get("passed") is not True
    or any(
        report.get("background_dependence_audits", {}).get(arm, {}).get("passed")
        is not True
        for arm in ("raw", "mask")
    )
):
    raise SystemExit("raw/mask dependence replay did not pass its publication gate")
PY
