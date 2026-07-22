#!/usr/bin/env bash
set -euo pipefail

# Aggregate exactly the ten predeclared validation reports. This runner is the
# final fail-closed boundary before a locked-suite freeze; it never opens O4b.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_SAFE_CORPUS
  INDEPENDENT_VALIDATION_ENDPOINT
  FIVE_SEED_MODEL
  GROUP_SAFE_DATA_SCALING
  CONTINUOUS_CANDIDATE_CALIBRATION
  PAIRED_RAW_MASK_VT
  CALIBRATION_PERTURBATION_ROBUSTNESS
  DETECTOR_SET_OOD_TRANSFER
  PAIRED_DINGO_AMPLFI_PE_PORTFOLIO
  LOCKED_CORPUS_UNOPENED
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
observed_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi
evidence_paths=(
  "$SOURCE_SAFE_CORPUS"
  "$INDEPENDENT_VALIDATION_ENDPOINT"
  "$FIVE_SEED_MODEL"
  "$GROUP_SAFE_DATA_SCALING"
  "$CONTINUOUS_CANDIDATE_CALIBRATION"
  "$PAIRED_RAW_MASK_VT"
  "$CALIBRATION_PERTURBATION_ROBUSTNESS"
  "$DETECTOR_SET_OOD_TRANSFER"
  "$PAIRED_DINGO_AMPLFI_PE_PORTFOLIO"
  "$LOCKED_CORPUS_UNOPENED"
)
for path in "${evidence_paths[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "required publication evidence is absent: $path" >&2
    exit 3
  fi
done

output_json="$OUTPUT_ROOT/publication_validation_evidence.json"
output_markdown="$OUTPUT_ROOT/publication_validation_evidence.md"
if [[ -e "$output_json" || -e "$output_markdown" ]]; then
  echo "publication validation ledger outputs are immutable" >&2
  exit 4
fi
mkdir -p "$OUTPUT_ROOT"

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
"$TASK_PYTHON" -m gwyolo.cli publication-evidence-audit \
  --config configs/publication_validation_evidence.yaml \
  --evidence "source_safe_corpus=$SOURCE_SAFE_CORPUS" \
  --evidence "independent_validation_endpoint=$INDEPENDENT_VALIDATION_ENDPOINT" \
  --evidence "five_seed_model=$FIVE_SEED_MODEL" \
  --evidence "group_safe_data_scaling=$GROUP_SAFE_DATA_SCALING" \
  --evidence "continuous_candidate_calibration=$CONTINUOUS_CANDIDATE_CALIBRATION" \
  --evidence "paired_raw_mask_vt=$PAIRED_RAW_MASK_VT" \
  --evidence "calibration_perturbation_robustness=$CALIBRATION_PERTURBATION_ROBUSTNESS" \
  --evidence "detector_set_ood_transfer=$DETECTOR_SET_OOD_TRANSFER" \
  --evidence "paired_dingo_amplfi_pe_portfolio=$PAIRED_DINGO_AMPLFI_PE_PORTFOLIO" \
  --evidence "locked_corpus_unopened=$LOCKED_CORPUS_UNOPENED" \
  --output "$output_json" \
  --markdown "$output_markdown" \
  --require-ready

"$TASK_PYTHON" - "$output_json" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import pathlib
import sys


path = pathlib.Path(sys.argv[1]).resolve()
commit = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
requirements = report.get("requirements", [])
if (
    report.get("status") != "publication_evidence_ready"
    or report.get("publication_ready") is not True
    or report.get("locked_final_evidence_complete") is not False
    or report.get("scientific_claim_allowed") is not False
    or report.get("phase") != "validation_freeze"
    or report.get("summary", {}).get("required_total") != 10
    or report.get("summary", {}).get("required_passed") != 10
    or len(requirements) != 10
    or any(row.get("state") != "passed" for row in requirements)
    or any(
        artifact.get("passed") is not True
        for row in requirements
        for artifact in row.get("artifact_replay", [])
    )
    or report.get("code_commit") != commit
):
    raise SystemExit("publication validation ledger failed final 10/10 replay")
print(json.dumps(report, indent=2, sort_keys=True))
PY
