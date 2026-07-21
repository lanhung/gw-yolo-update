#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  PROMOTION_REPORT
  SOURCE_SAFE_ROOT
  FIVE_SEED_ROOT
  SUMMARY_OUTPUT
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
  CANDIDATE_OUTPUT_ROOT
  GWYOLO_CODE_COMMIT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

arm=$("$TASK_PYTHON" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
arm = report.get("promoted_arm")
if not report.get("passed") or arm not in {"uniform", "family_balanced"}:
    raise SystemExit("overlap sampling promotion did not select an arm")
print(arm)
' "$PROMOTION_REPORT")
if [[ "$arm" == uniform ]]; then
  original_report="$SOURCE_SAFE_ROOT/uniform-seed20260720/overlap_finetune_report.json"
else
  original_report="$SOURCE_SAFE_ROOT/family-balanced-seed20260720/overlap_finetune_report.json"
fi
reports=(--report "$original_report")
for seed in 20260721 20260722 20260723 20260724; do
  reports+=(--report "$FIVE_SEED_ROOT/${arm}-seed${seed}/overlap_finetune_report.json")
done
"$TASK_PYTHON" -m gwyolo.cli physical-overlap-five-seed-summarize \
  --promotion-report "$PROMOTION_REPORT" \
  "${reports[@]}" \
  --output "$SUMMARY_OUTPUT"

export FIVE_SEED_SUMMARY="$SUMMARY_OUTPUT"
export OUTPUT_ROOT="$CANDIDATE_OUTPUT_ROOT"
bash scripts/run_promoted_candidate_validation.sh
