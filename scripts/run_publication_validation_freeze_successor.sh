#!/usr/bin/env bash
set -euo pipefail

# Build the immutable 10/10 validation ledger and freeze every locked-suite
# path. This is intentionally a pre-access boundary: it never creates the O4b
# access log and never reads a test manifest or strain sample.

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
  VALIDATION_LEDGER_OUTPUT_ROOT
  LOCKED_SUITE_OUTPUT_ROOT
  LOCKED_SUITE_PLAN_OUTPUT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "publication validation freeze requires its exact checkout" >&2
  exit 3
fi
evidence=(
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
for path in "$TASK_PYTHON" "${evidence[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "required publication validation freeze input is absent: $path" >&2
    exit 3
  fi
done

ledger="$VALIDATION_LEDGER_OUTPUT_ROOT/publication_validation_evidence.json"
if [[ ! -s "$ledger" ]]; then
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    SOURCE_SAFE_CORPUS="$SOURCE_SAFE_CORPUS" \
    INDEPENDENT_VALIDATION_ENDPOINT="$INDEPENDENT_VALIDATION_ENDPOINT" \
    FIVE_SEED_MODEL="$FIVE_SEED_MODEL" \
    GROUP_SAFE_DATA_SCALING="$GROUP_SAFE_DATA_SCALING" \
    CONTINUOUS_CANDIDATE_CALIBRATION="$CONTINUOUS_CANDIDATE_CALIBRATION" \
    PAIRED_RAW_MASK_VT="$PAIRED_RAW_MASK_VT" \
    CALIBRATION_PERTURBATION_ROBUSTNESS="$CALIBRATION_PERTURBATION_ROBUSTNESS" \
    DETECTOR_SET_OOD_TRANSFER="$DETECTOR_SET_OOD_TRANSFER" \
    PAIRED_DINGO_AMPLFI_PE_PORTFOLIO="$PAIRED_DINGO_AMPLFI_PE_PORTFOLIO" \
    LOCKED_CORPUS_UNOPENED="$LOCKED_CORPUS_UNOPENED" \
    OUTPUT_ROOT="$VALIDATION_LEDGER_OUTPUT_ROOT" \
    bash "$TASK_CODE_DIR/scripts/run_publication_validation_ledger.sh"
fi

if [[ ! -e "$LOCKED_SUITE_PLAN_OUTPUT" ]]; then
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src
  export GWYOLO_CODE_COMMIT
  "$TASK_PYTHON" -m gwyolo.cli locked-evaluation-suite-freeze \
    --validation-evidence-report "$ledger" \
    --config "$TASK_CODE_DIR/configs/locked_evaluation_suite_gwtc5.yaml" \
    --output-root "$LOCKED_SUITE_OUTPUT_ROOT" \
    --code-commit "$GWYOLO_CODE_COMMIT" \
    --output "$LOCKED_SUITE_PLAN_OUTPUT"
fi

"$TASK_PYTHON" - "$ledger" "$LOCKED_SUITE_PLAN_OUTPUT" \
  "$LOCKED_CORPUS_UNOPENED" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ledger_path, plan_path, unopened_path = map(pathlib.Path, sys.argv[1:4])
commit = sys.argv[4]
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
plan = json.loads(plan_path.read_text(encoding="utf-8"))
unopened = json.loads(unopened_path.read_text(encoding="utf-8"))
access_log = pathlib.Path(unopened["access_log_path"])
if (
    ledger.get("status") != "publication_evidence_ready"
    or ledger.get("publication_ready") is not True
    or ledger.get("summary", {}).get("required_passed") != 10
    or ledger.get("summary", {}).get("required_total") != 10
    or plan.get("status") != "frozen_locked_evaluation_suite_plan"
    or plan.get("passed") is not True
    or plan.get("test_rows_read") != 0
    or plan.get("locked_corpus_opened") is not False
    or plan.get("validation_evidence", {}).get("sha256") != digest(ledger_path)
    or plan.get("code_commit") != commit
    or unopened.get("evaluation_opened") is not False
    or access_log.exists()
):
    raise SystemExit("publication validation-to-locked freeze replay failed")
print(json.dumps(plan, indent=2, sort_keys=True))
PY
