#!/usr/bin/env bash
set -euo pipefail

# Materialize the full variable-detector overlap corpus only after both
# detector-expanded signal banks and the hash-bound capacity preflight pass.
# This is a detector-set robustness data product, not same-distribution scaling.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  GRAVITYSPY_TRAIN_MANIFEST
  GRAVITYSPY_VALIDATION_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  EXPANDED_TRAIN_MANIFEST
  EXPANDED_TRAIN_REPORT
  EXPANDED_VALIDATION_MANIFEST
  EXPANDED_VALIDATION_REPORT
  EXPANSION_READINESS_AUDIT
  EXPANSION_CAPACITY_REPORT
  OVERLAP_CONFIG
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required detector-set overlap variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$GRAVITYSPY_TRAIN_MANIFEST" \
  "$GRAVITYSPY_VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$EXPANDED_TRAIN_MANIFEST" \
  "$EXPANDED_TRAIN_REPORT" \
  "$EXPANDED_VALIDATION_MANIFEST" \
  "$EXPANDED_VALIDATION_REPORT" \
  "$EXPANSION_READINESS_AUDIT" \
  "$EXPANSION_CAPACITY_REPORT" \
  "$OVERLAP_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "detector-set overlap input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "detector-set overlap materialization requires its exact checkout" >&2
  exit 3
fi
minimum_free_kb=${MINIMUM_FREE_KB:-4194304}
if ! [[ "$minimum_free_kb" =~ ^[1-9][0-9]*$ ]]; then
  echo "MINIMUM_FREE_KB must be a positive integer" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
available_kb=$(df -Pk "$OUTPUT_ROOT" | awk 'NR == 2 {print $4}')
if (( available_kb < minimum_free_kb )); then
  echo "insufficient free space for detector-set overlap materialization" >&2
  exit 1
fi

"$TASK_PYTHON" - \
  "$EXPANDED_TRAIN_REPORT" \
  "$EXPANDED_TRAIN_MANIFEST" \
  "$EXPANDED_VALIDATION_REPORT" \
  "$EXPANDED_VALIDATION_MANIFEST" \
  "$EXPANSION_READINESS_AUDIT" \
  "$EXPANSION_CAPACITY_REPORT" \
  "$GRAVITYSPY_TRAIN_MANIFEST" \
  "$GRAVITYSPY_VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" <<'PY'
import hashlib
import json
import pathlib
import sys


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    train_report_path,
    train_manifest,
    validation_report_path,
    validation_manifest,
    readiness_path,
    capacity_path,
    glitch_train,
    glitch_validation,
    corpus_audit_path,
) = sys.argv[1:]
train = load(train_report_path)
validation = load(validation_report_path)
readiness = load(readiness_path)
capacity = load(capacity_path)
corpus = load(corpus_audit_path)
train_glitches = sum(
    bool(line.strip())
    for line in pathlib.Path(glitch_train).read_text(encoding="utf-8").splitlines()
)
validation_glitches = sum(
    bool(line.strip())
    for line in pathlib.Path(glitch_validation).read_text(encoding="utf-8").splitlines()
)
for report, manifest, split, expected_minimum in (
    (train, train_manifest, "train", train_glitches),
    (validation, validation_manifest, "val", validation_glitches),
):
    if (
        report.get("status") != "verified_physical_detector_set_expansion"
        or report.get("passed") is not True
        or report.get("selected_split") != split
        or report.get("test_rows_read") != 0
        or report.get("test_evaluation") is not None
        or report.get("same_distribution_data_scaling_claim_allowed") is not False
        or report.get("manifest_sha256") != digest(manifest)
        or int(report.get("rows", -1)) < expected_minimum
    ):
        raise SystemExit("detector-expanded signal bank failed replay")
expected_report_hashes = {digest(train_report_path), digest(validation_report_path)}
if (
    readiness.get("status") != "audited_detector_set_signal_bank_readiness"
    or readiness.get("passed") is not True
    or readiness.get("signal_overlap_materialization_authorized") is not True
    or readiness.get("detector_complete_clean_training_authorized") is not False
    or readiness.get("detector_set_robustness_ablation_ready") is not False
    or readiness.get("test_rows_read") != 0
    or readiness.get("test_evaluation") is not None
    or {row.get("sha256") for row in readiness.get("reports", [])}
    != expected_report_hashes
):
    raise SystemExit("detector expansion readiness audit failed replay")
if (
    capacity.get("status") != "audited_physical_overlap_expansion_capacity"
    or capacity.get("passed") is not True
    or capacity.get("test_rows_read") != 0
    or capacity.get("test_evaluation") is not None
    or int(capacity.get("maximum_all_detector_set_physical_groups", -1))
    != train_glitches
    or int(capacity.get("maximum_same_distribution_physical_groups", -1))
    >= train_glitches
    or capacity.get("next_scale_training_authorized") is not False
    or capacity.get("inputs", {})
    .get("candidate_injection_audit", {})
    .get("sha256")
    != digest(train_report_path)
):
    raise SystemExit("detector-set overlap capacity preflight failed replay")
if (
    corpus.get("status")
    != "verified_group_safe_gravityspy_aligned_network_corpus"
    or corpus.get("passed") is not True
    or corpus.get("train_manifest_sha256") != digest(glitch_train)
    or corpus.get("validation_manifest_sha256") != digest(glitch_validation)
    or any(corpus.get("split_audit", {}).get("cross_split_overlaps", {}).values())
):
    raise SystemExit("Gravity Spy source corpus failed group-safe replay")
PY

train_root="$OUTPUT_ROOT/train-overlaps"
validation_root="$OUTPUT_ROOT/val-overlaps"
train_manifest="$train_root/physical_overlap_train_manifest.jsonl"
validation_manifest="$validation_root/physical_overlap_val_manifest.jsonl"
cd "$TASK_CODE_DIR"
export PYTHONPATH=src GWYOLO_CODE_COMMIT
if [[ ! -s "$train_root/physical_overlap_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-materialize \
    --gravityspy-manifest "$GRAVITYSPY_TRAIN_MANIFEST" \
    --injection-manifest "$EXPANDED_TRAIN_MANIFEST" \
    --config "$OVERLAP_CONFIG" \
    --output-dir "$train_root" \
    --split train \
    --seed "${OVERLAP_SEED:-20260720}" \
    --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT"
fi
if [[ ! -s "$validation_root/physical_overlap_report.json" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-materialize \
    --gravityspy-manifest "$GRAVITYSPY_VALIDATION_MANIFEST" \
    --injection-manifest "$EXPANDED_VALIDATION_MANIFEST" \
    --config "$OVERLAP_CONFIG" \
    --output-dir "$validation_root" \
    --split val \
    --seed "${OVERLAP_SEED:-20260720}" \
    --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT"
fi

joint_audit="$OUTPUT_ROOT/train_validation_group_audit.json"
"$TASK_PYTHON" -m gwyolo.cli physical-overlap-audit \
  --manifest "$train_manifest" \
  --manifest "$validation_manifest" \
  --output "$joint_audit"

receipt="$OUTPUT_ROOT/detector_set_overlap_materialization_receipt.json"
"$TASK_PYTHON" - \
  "$receipt" \
  "$train_root/physical_overlap_report.json" \
  "$validation_root/physical_overlap_report.json" \
  "$joint_audit" \
  "$EXPANSION_READINESS_AUDIT" \
  "$EXPANSION_CAPACITY_REPORT" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

(
    target,
    train_path,
    validation_path,
    audit_path,
    readiness_path,
    capacity_path,
    commit,
) = sys.argv[1:]


def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def artifact(path):
    value = pathlib.Path(path).resolve()
    return {
        "path": str(value),
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
    }


train = load(train_path)
validation = load(validation_path)
audit = load(audit_path)
capacity = load(capacity_path)
if (
    train.get("status") != "verified_real_glitch_physical_overlap_training_data"
    or validation.get("status")
    != "verified_real_glitch_physical_overlap_training_data"
    or train.get("split") != "train"
    or validation.get("split") != "val"
    or int(train.get("rows", -1))
    != int(capacity.get("maximum_all_detector_set_physical_groups", -2))
    or int(train.get("single_ifo_rows", -1)) != 0
    or int(validation.get("single_ifo_rows", -1)) != 0
    or audit.get("status") != "passed_physical_overlap_group_audit"
    or audit.get("passed") is not True
    or any(
        values
        for pair in audit.get("cross_split_overlaps", {}).values()
        for values in pair.values()
    )
):
    raise SystemExit("detector-set overlap materialization failed final replay")
result = {
    "status": "verified_detector_set_overlap_robustness_corpus",
    "passed": True,
    "scientific_claim_allowed": False,
    "same_distribution_data_scaling_claim_allowed": False,
    "scientific_blocker": (
        "detector-set overlap data require validation-selected training, empirical-noise "
        "O4 transfer, continuous-background search and locked evaluation"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "rows_by_split": {"train": train["rows"], "val": validation["rows"]},
    "detector_subset_counts_by_split": {
        "train": train["detector_subset_counts"],
        "val": validation["detector_subset_counts"],
    },
    "artifacts": {
        "train_report": artifact(train_path),
        "validation_report": artifact(validation_path),
        "joint_group_audit": artifact(audit_path),
        "expansion_readiness_audit": artifact(readiness_path),
        "capacity_report": artifact(capacity_path),
    },
    "code_commit": commit,
}
target = pathlib.Path(target)
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, target)
PY
