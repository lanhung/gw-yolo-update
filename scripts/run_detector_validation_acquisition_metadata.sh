#!/usr/bin/env bash
set -euo pipefail

# Freeze score-blind O3 development acquisition metadata for detector subsets
# that are undersized in the source-safe validation bank. Strain is not
# downloaded by this script.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FROZEN_TRAIN_MANIFEST
  FROZEN_VALIDATION_MANIFEST
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector acquisition variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$FROZEN_TRAIN_MANIFEST" \
  "$FROZEN_VALIDATION_MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "detector acquisition input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector acquisition metadata requires its exact checkout" >&2
  exit 3
fi

run=${DETECTOR_ACQUISITION_RUN:-O3b}
target_pairs=${TARGET_PAIRS_PER_SUBSET:-32}
seed=${DETECTOR_ACQUISITION_SEED:-20260723}
inventory_h1v1="$OUTPUT_ROOT/${run}-H1-V1-inventory.json"
inventory_l1v1="$OUTPUT_ROOT/${run}-L1-V1-inventory.json"
plan_h1v1="$OUTPUT_ROOT/${run}-H1-V1-source-disjoint.json"
plan_l1v1="$OUTPUT_ROOT/${run}-L1-V1-source-disjoint.json"

mkdir -p "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
if [[ ! -s "$inventory_h1v1" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli gwosc-run-plan \
    --run "$run" \
    --detectors H1 V1 \
    --sample-rate-khz 4 \
    --seed "$seed" \
    --output "$inventory_h1v1"
fi
if [[ ! -s "$plan_h1v1" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli detector-validation-acquisition-plan \
    --inventory-plan "$inventory_h1v1" \
    --frozen-network-manifest "$FROZEN_TRAIN_MANIFEST" \
    --frozen-network-manifest "$FROZEN_VALIDATION_MANIFEST" \
    --target-pairs "$target_pairs" \
    --seed "$seed" \
    --output "$plan_h1v1"
fi
if [[ ! -s "$inventory_l1v1" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli gwosc-run-plan \
    --run "$run" \
    --detectors L1 V1 \
    --sample-rate-khz 4 \
    --seed "$seed" \
    --output "$inventory_l1v1"
fi
if [[ ! -s "$plan_l1v1" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli detector-validation-acquisition-plan \
    --inventory-plan "$inventory_l1v1" \
    --frozen-network-manifest "$FROZEN_TRAIN_MANIFEST" \
    --frozen-network-manifest "$FROZEN_VALIDATION_MANIFEST" \
    --exclude-plan "$plan_h1v1" \
    --target-pairs "$target_pairs" \
    --seed "$((seed + 1))" \
    --output "$plan_l1v1"
fi

"$TASK_PYTHON" - "$plan_h1v1" "$plan_l1v1" <<'PY'
import json
import pathlib
import sys

plans = [
    json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for path in sys.argv[1:]
]
if any(
    plan.get("status") != "development_acquisition_plan"
    or plan.get("candidate_scores_inspected") is not False
    or plan.get("test_data_opened") is not False
    or plan.get("locked_evaluation_data") is not False
    or plan.get("selection_rule")
    != "source_file_and_gps_disjoint_stratified_v1"
    for plan in plans
):
    raise SystemExit("detector acquisition metadata failed replay")
left_gps = {row["gps_start"] for row in plans[0]["pairs"]}
right_gps = {row["gps_start"] for row in plans[1]["pairs"]}
if left_gps & right_gps:
    raise SystemExit("detector acquisition subset plans share GPS starts")
print(
    json.dumps(
        {
            "+".join(plan["detectors"]): {
                "selected_pairs": plan["selected_pairs"],
                "eligible_pairs_after_exclusion": (
                    plan["eligible_pairs_after_exclusion"]
                ),
                "selected_gps_span": plan["selected_gps_span"],
            }
            for plan in plans
        },
        indent=2,
        sort_keys=True,
    )
)
PY
