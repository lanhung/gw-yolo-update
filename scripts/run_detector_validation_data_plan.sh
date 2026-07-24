#!/usr/bin/env bash
set -euo pipefail

# Score-blind pre-access data audit. It exports one real numeric background
# bank per source-safe validation GPS block and freezes equal-count physical
# injection recipes only when every detector subset clears its data floor.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  NETWORK_VALIDATION_MANIFEST
  SOURCE_SAFE_CORPUS_AUDIT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector-validation data variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$NETWORK_VALIDATION_MANIFEST" \
  "$SOURCE_SAFE_CORPUS_AUDIT"; do
  if [[ ! -s "$path" ]]; then
    echo "detector-validation data input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector-validation data plan requires its exact checkout" >&2
  exit 3
fi

minimum_per_subset=${MINIMUM_PER_DETECTOR_SUBSET:-25}
injections_per_subset=${INJECTIONS_PER_DETECTOR_SUBSET:-100}
analysis_duration=${ANALYSIS_DURATION_SECONDS:-4}
seed=${DETECTOR_VALIDATION_SEED:-20260723}
background_root="$OUTPUT_ROOT/background"
plan_root="$OUTPUT_ROOT/injections"
background_report="$background_root/detector_validation_background_report.json"
background_manifest="$background_root/background_windows.jsonl"

mkdir -p "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
if [[ ! -s "$background_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli detector-validation-background-export \
    --network-manifest "$NETWORK_VALIDATION_MANIFEST" \
    --corpus-audit "$SOURCE_SAFE_CORPUS_AUDIT" \
    --output-dir "$background_root" \
    --analysis-duration-seconds "$analysis_duration" \
    --minimum-per-detector-subset "$minimum_per_subset"
fi

ready=$(
  "$TASK_PYTHON" - "$background_report" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status")
    != "exported_source_safe_detector_validation_background_bank"
    or report.get("scientific_claim_allowed") is not False
    or report.get("test_rows_read") != 0
    or report.get("candidate_scores_inspected") is not False
    or report.get("physical_signal_present") is not False
    or report.get("physical_signal_projection_required") is not True
):
    raise SystemExit("detector-validation background audit failed replay")
print("1" if report.get("passed") is True else "0")
PY
)

if [[ "$ready" == 1 && ! -s "$plan_root/detector_stratified_injection_plan.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli detector-validation-injection-plan \
    --background-manifest "$background_manifest" \
    --background-report "$background_report" \
    --output-dir "$plan_root" \
    --injections-per-detector-subset "$injections_per_subset" \
    --seed "$seed"
fi

"$TASK_PYTHON" - "$background_report" "$plan_root" <<'PY'
import json
import pathlib
import sys

background = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
plan_path = pathlib.Path(sys.argv[2]) / "detector_stratified_injection_plan.json"
print(
    json.dumps(
        {
            "background_ready": background["passed"],
            "detector_subset_counts": background["detector_subset_counts"],
            "detector_subset_deficits": background["detector_subset_deficits"],
            "injection_plan_frozen": plan_path.is_file(),
            "scientific_claim_allowed": False,
            "test_rows_read": 0,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
