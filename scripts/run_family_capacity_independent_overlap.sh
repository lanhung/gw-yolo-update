#!/usr/bin/env bash
set -euo pipefail

# Rebuild the validation-only chirp+glitch overlap after the final family-safe
# resplit. Historical independent overlaps are not reusable because a new
# source-component split can move an old validation glitch into model training.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BASE_OVERLAP_ROOT
  BASE_GRAVITYSPY_CORPUS_AUDIT
  BASE_VALIDATION_GLITCH_MANIFEST
  RARE_OVERLAP_ROOT
  RARE_GRAVITYSPY_CORPUS_AUDIT
  RARE_VALIDATION_GLITCH_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  MATERIALIZATION_CONFIG
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$MATERIALIZATION_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "required family-capacity independent-overlap input is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "family-capacity independent overlap requires its exact checkout" >&2
  exit 3
fi

if ! resolved=$(
  "$TASK_PYTHON" - \
    base "$BASE_OVERLAP_ROOT" "$BASE_GRAVITYSPY_CORPUS_AUDIT" \
      "$BASE_VALIDATION_GLITCH_MANIFEST" \
    rare "$RARE_OVERLAP_ROOT" "$RARE_GRAVITYSPY_CORPUS_AUDIT" \
      "$RARE_VALIDATION_GLITCH_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


eligible = []
values = sys.argv[1:]
if len(values) % 4:
    raise SystemExit("family-capacity overlap candidates are malformed")
for offset in range(0, len(values), 4):
    name, raw_root, raw_audit, raw_validation = values[offset : offset + 4]
    root = pathlib.Path(raw_root).resolve()
    audit = pathlib.Path(raw_audit).resolve()
    validation = pathlib.Path(raw_validation).resolve()
    train_overlap = root / "train-overlaps/physical_overlap_train_manifest.jsonl"
    summary_path = root / "five-seed/five_seed_overlap_summary.json"
    receipt_path = root / "source_safe_overlap_chain_receipt.json"
    if not summary_path.is_file() and not receipt_path.is_file():
        continue
    for path in (audit, validation, train_overlap, summary_path, receipt_path):
        if not path.is_file():
            raise SystemExit(f"{name} branch is partially materialized: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    corpus = json.loads(audit.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "completed_five_seed_source_safe_overlap_validation"
        or summary.get("passed") is not True
        or summary.get("five_seed_stability", {}).get("passed") is not True
        or summary.get("test_data_opened") is not False
        or receipt.get("execution_passed") is not True
        or receipt.get("five_seed_promoted") is not True
        or receipt.get("five_seed_summary", {}).get("sha256") != digest(summary_path)
        or receipt.get("inputs", {}).get("gravityspy_corpus_audit", {}).get("sha256")
        != digest(audit)
        or receipt.get("inputs", {}).get("overlap_train_manifest", {}).get("sha256")
        != digest(train_overlap)
        or corpus.get("status")
        != "verified_group_safe_gravityspy_aligned_network_corpus"
        or corpus.get("passed") is not True
        or corpus.get("validation_manifest_sha256") != digest(validation)
        or any(corpus.get("split_audit", {}).get("cross_split_overlaps", {}).values())
    ):
        raise SystemExit(f"{name} family-safe overlap branch failed replay")
    eligible.append((name, root, audit, validation, train_overlap, summary_path))
if len(eligible) != 1:
    raise SystemExit(f"expected one passing family-safe model branch, found {len(eligible)}")
print("\t".join(str(value) for value in eligible[0]))
PY
); then
  echo "family-capacity independent-overlap branch resolution failed" >&2
  exit 4
fi
IFS=$'\t' read -r \
  selected_branch selected_root selected_audit selected_validation \
  selected_train_overlap selected_summary <<<"$resolved"

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
env \
  TASK_PYTHON="$TASK_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  VALIDATION_GLITCH_MANIFEST="$selected_validation" \
  GRAVITYSPY_CORPUS_AUDIT="$selected_audit" \
  TRAIN_OVERLAP_MANIFEST="$selected_train_overlap" \
  MATERIALIZATION_CONFIG="$MATERIALIZATION_CONFIG" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SEED="${SEED:-20260726}" \
  MINIMUM_OVERLAP_ROWS="${MINIMUM_OVERLAP_ROWS:-100}" \
  bash scripts/run_independent_pe_overlap.sh

"$TASK_PYTHON" - \
  "$OUTPUT_ROOT/independent_pe_overlap_report.json" \
  "$selected_branch" "$selected_audit" "$selected_validation" \
  "$selected_train_overlap" "$selected_summary" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


report_path, branch, raw_audit, raw_validation, raw_train, raw_summary = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
audit = pathlib.Path(raw_audit).resolve()
validation = pathlib.Path(raw_validation).resolve()
train = pathlib.Path(raw_train).resolve()
if (
    report.get("status") != "verified_independent_validation_pe_overlap"
    or report.get("passed") is not True
    or report.get("test_rows_read") != 0
    or pathlib.Path(report["gravityspy_corpus_audit_path"]).resolve() != audit
    or report.get("gravityspy_corpus_audit_sha256") != digest(audit)
    or pathlib.Path(report["validation_glitch_manifest_path"]).resolve() != validation
    or report.get("validation_glitch_manifest_sha256") != digest(validation)
    or pathlib.Path(report["training_overlap_manifest_path"]).resolve() != train
    or report.get("training_overlap_manifest_sha256") != digest(train)
):
    raise SystemExit("family-capacity independent overlap failed final identity replay")
report["family_capacity_branch"] = branch
report["five_seed_summary_path"] = str(pathlib.Path(raw_summary).resolve())
report["five_seed_summary_sha256"] = digest(pathlib.Path(raw_summary))
print(json.dumps(report, indent=2, sort_keys=True))
PY
