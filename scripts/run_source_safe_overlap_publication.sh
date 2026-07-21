#!/usr/bin/env bash
set -euo pipefail

# Build and validate the complete validation-only real-glitch overlap model-selection
# chain. A valid negative promotion is retained without opening test data or starting
# five-seed training. Every accepted input and output is replayed by hash.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  TRAIN_GLITCH_MANIFEST
  VALIDATION_GLITCH_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

materialization_config=${MATERIALIZATION_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_factory.yaml}
uniform_config=${UNIFORM_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_finetune.yaml}
family_balanced_config=${FAMILY_BALANCED_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_finetune_family_balanced.yaml}
promotion_config=${PROMOTION_CONFIG:-$TASK_CODE_DIR/configs/physical_overlap_sampling_promotion.yaml}
seed=${SEED:-20260720}
for path in \
  "$TASK_PYTHON" \
  "$TRAIN_GLITCH_MANIFEST" \
  "$VALIDATION_GLITCH_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$materialization_config" \
  "$uniform_config" \
  "$family_balanced_config" \
  "$promotion_config"; do
  if [[ ! -s "$path" ]]; then
    echo "required source-safe overlap artifact is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi
if ! [[ "$seed" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED must be a positive integer" >&2
  exit 4
fi

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
export CUDA_VISIBLE_DEVICES="${OVERLAP_CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_ROOT/logs"

"$TASK_PYTHON" - \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$TRAIN_GLITCH_MANIFEST" \
  "$VALIDATION_GLITCH_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


audit_path, train_path, validation_path = map(pathlib.Path, sys.argv[1:])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
overlaps = audit.get("split_audit", {}).get("cross_split_overlaps")
if (
    audit.get("status")
    != "verified_group_safe_gravityspy_aligned_network_corpus"
    or audit.get("passed") is not True
    or not isinstance(overlaps, dict)
    or not overlaps
    or any(overlaps.values())
    or audit.get("train_manifest_sha256") != digest(train_path)
    or audit.get("validation_manifest_sha256") != digest(validation_path)
):
    raise SystemExit("source-safe Gravity Spy corpus replay failed")
PY

env \
  TASK_PYTHON="$TASK_PYTHON" \
  TRAIN_GLITCH_MANIFEST="$TRAIN_GLITCH_MANIFEST" \
  VAL_GLITCH_MANIFEST="$VALIDATION_GLITCH_MANIFEST" \
  TRAIN_INJECTION_MANIFEST="$CLEAN_TRAIN_MANIFEST" \
  VAL_INJECTION_MANIFEST="$CLEAN_VALIDATION_MANIFEST" \
  PRETRAINED_CHECKPOINT="$PRETRAINED_CHECKPOINT" \
  MATERIALIZATION_CONFIG="$materialization_config" \
  UNIFORM_CONFIG="$uniform_config" \
  FAMILY_BALANCED_CONFIG="$family_balanced_config" \
  CLEAN_VALIDATION_FEATURE_CACHE_DIR="${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" \
  GLITCH_CORPUS_AUDIT="$GRAVITYSPY_CORPUS_AUDIT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  SEED="$seed" \
  bash scripts/run_recovered_overlap_ablation.sh \
  >"$OUTPUT_ROOT/logs/one-seed-ablation.log" 2>&1

overlap_train="$OUTPUT_ROOT/train-overlaps/physical_overlap_train_manifest.jsonl"
overlap_validation="$OUTPUT_ROOT/val-overlaps/physical_overlap_val_manifest.jsonl"
uniform_report="$OUTPUT_ROOT/uniform-seed$seed/overlap_finetune_report.json"
family_report="$OUTPUT_ROOT/family-balanced-seed$seed/overlap_finetune_report.json"
for path in "$overlap_train" "$overlap_validation" "$uniform_report" "$family_report"; do
  if [[ ! -s "$path" ]]; then
    echo "one-seed overlap ablation omitted an artifact: $path" >&2
    exit 5
  fi
done

"$TASK_PYTHON" - \
  "$uniform_report" \
  "$family_report" \
  "$overlap_train" \
  "$overlap_validation" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT" \
  "$uniform_config" \
  "$family_balanced_config" \
  "$seed" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    uniform_report,
    family_report,
    overlap_train,
    overlap_validation,
    clean_train,
    clean_validation,
    pretrained,
    uniform_config,
    family_config,
) = map(pathlib.Path, sys.argv[1:10])
seed = int(sys.argv[10])
commit = sys.argv[11]
common = {
    "overlap_train_manifest_sha256": digest(overlap_train),
    "overlap_validation_manifest_sha256": digest(overlap_validation),
    "clean_train_manifest_sha256": digest(clean_train),
    "clean_validation_manifest_sha256": digest(clean_validation),
    "pretrained_checkpoint_sha256": digest(pretrained),
}
for report_path, config_path in (
    (uniform_report, uniform_config),
    (family_report, family_config),
):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = pathlib.Path(report.get("checkpoint_path", ""))
    if (
        report.get("status")
        != "validation_selected_real_glitch_overlap_finetune"
        or report.get("code_commit") != commit
        or report.get("scientific_claim_allowed") is not False
        or report.get("search_claim_allowed") is not False
        or int(report.get("seed", -1)) != seed
        or report.get("config_file_sha256") != digest(config_path)
        or any(report.get(key) != value for key, value in common.items())
        or not checkpoint.is_file()
        or report.get("checkpoint_sha256") != digest(checkpoint)
    ):
        raise SystemExit(f"one-seed overlap report replay failed: {report_path}")
PY

promotion_report="$OUTPUT_ROOT/overlap_sampling_promotion.json"
"$TASK_PYTHON" -m gwyolo.cli physical-overlap-sampling-promote \
  --uniform-report "$uniform_report" \
  --family-balanced-report "$family_report" \
  --overlap-train-manifest "$overlap_train" \
  --overlap-validation-manifest "$overlap_validation" \
  --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT" \
  --config "$promotion_config" \
  --output "$promotion_report" \
  >"$OUTPUT_ROOT/logs/promotion.log" 2>&1

arm=$("$TASK_PYTHON" - "$promotion_report" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "validation_only_overlap_sampling_promotion"
    or report.get("code_commit") != sys.argv[2]
    or report.get("test_data_opened") is not False
    or report.get("scientific_claim_allowed") is not False
):
    raise SystemExit("overlap promotion receipt is invalid")
if report.get("passed"):
    arm = report.get("promoted_arm")
    if not report.get("scale_to_five_seeds") or arm not in {
        "uniform",
        "family_balanced",
    }:
        raise SystemExit("overlap promotion has an invalid promoted arm")
    print(arm)
elif report.get("promoted_arm") is not None or report.get("scale_to_five_seeds"):
    raise SystemExit("negative overlap promotion contains an enabled arm")
PY
)

five_seed_summary=-
if [[ -n "$arm" ]]; then
  five_seed_root="$OUTPUT_ROOT/five-seed"
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    PROMOTION_REPORT="$promotion_report" \
    ORIGINAL_UNIFORM_REPORT="$uniform_report" \
    ORIGINAL_FAMILY_BALANCED_REPORT="$family_report" \
    OVERLAP_TRAIN_MANIFEST="$overlap_train" \
    OVERLAP_VALIDATION_MANIFEST="$overlap_validation" \
    CLEAN_TRAIN_MANIFEST="$CLEAN_TRAIN_MANIFEST" \
    CLEAN_VALIDATION_MANIFEST="$CLEAN_VALIDATION_MANIFEST" \
    PRETRAINED_CHECKPOINT="$PRETRAINED_CHECKPOINT" \
    UNIFORM_CONFIG="$uniform_config" \
    FAMILY_BALANCED_CONFIG="$family_balanced_config" \
    CLEAN_VALIDATION_FEATURE_CACHE_DIR="${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" \
    OUTPUT_ROOT="$five_seed_root" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    bash scripts/run_overlap_five_seed_promotion.sh \
    >"$OUTPUT_ROOT/logs/five-seed.log" 2>&1
  five_seed_summary="$five_seed_root/five_seed_overlap_summary.json"
  if [[ ! -s "$five_seed_summary" ]]; then
    echo "promoted overlap arm omitted its five-seed summary" >&2
    exit 6
  fi
fi

"$TASK_PYTHON" - \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$overlap_train" \
  "$overlap_validation" \
  "$uniform_report" \
  "$family_report" \
  "$promotion_report" \
  "$five_seed_summary" \
  "$OUTPUT_ROOT/source_safe_overlap_chain_receipt.json" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


paths = [pathlib.Path(value) for value in sys.argv[1:7]]
summary_arg = sys.argv[7]
output = pathlib.Path(sys.argv[8])
commit = sys.argv[9]
promotion = json.loads(paths[-1].read_text(encoding="utf-8"))
if (
    promotion.get("code_commit") != commit
    or promotion.get("corpus_audit_sha256") != digest(paths[0])
    or promotion.get("overlap_manifest_hashes")
    != {"train": digest(paths[1]), "val": digest(paths[2])}
    or promotion.get("input_report_hashes")
    != {"uniform": digest(paths[3]), "family_balanced": digest(paths[4])}
):
    raise SystemExit("overlap promotion provenance replay failed")
summary = None
if summary_arg != "-":
    summary_path = pathlib.Path(summary_arg)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status")
        != "completed_five_seed_source_safe_overlap_validation"
        or summary.get("code_commit") != commit
        or summary.get("passed") is not True
        or summary.get("test_data_opened") is not False
        or summary.get("scientific_claim_allowed") is not False
        or summary.get("promoted_arm") != promotion.get("promoted_arm")
    ):
        raise SystemExit("five-seed overlap summary replay failed")
    summary_entry = {"path": str(summary_path), "sha256": digest(summary_path)}
else:
    if promotion.get("passed") or promotion.get("scale_to_five_seeds"):
        raise SystemExit("passing promotion omitted the five-seed summary")
    summary_entry = None
receipt = {
    "status": (
        "completed_source_safe_overlap_five_seed_chain"
        if summary_entry
        else "completed_source_safe_overlap_negative_promotion"
    ),
    "execution_passed": True,
    "five_seed_promoted": summary_entry is not None,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "test_rows_read": 0,
    "code_commit": commit,
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
    "inputs": {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in zip(
            (
                "gravityspy_corpus_audit",
                "overlap_train_manifest",
                "overlap_validation_manifest",
                "uniform_report",
                "family_balanced_report",
                "promotion_report",
            ),
            paths,
        )
    },
    "five_seed_summary": summary_entry,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
