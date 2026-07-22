#!/usr/bin/env bash
set -euo pipefail

# Resolve exactly one family-safe corpus/model branch, retain a detector-set OOD
# result independently of model promotion, and run the expensive scaling curve
# only after the five-seed validation gate passes. No test data are read.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BASE_OVERLAP_ROOT
  BASE_GRAVITYSPY_CORPUS_AUDIT
  BASE_TRAIN_MANIFEST
  BASE_VALIDATION_MANIFEST
  RARE_OVERLAP_ROOT
  RARE_GRAVITYSPY_CORPUS_AUDIT
  RARE_TRAIN_MANIFEST
  RARE_VALIDATION_MANIFEST
  CLEAN_TRAIN_MANIFEST
  CLEAN_VALIDATION_MANIFEST
  PRETRAINED_CHECKPOINT
  SCALING_OUTPUT_ROOT
  HARD_ENDPOINT_OUTPUT_ROOT
  OOD_OUTPUT_ROOT
  SUCCESSOR_RECEIPT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$CLEAN_TRAIN_MANIFEST" \
  "$CLEAN_VALIDATION_MANIFEST" \
  "$PRETRAINED_CHECKPOINT"; do
  if [[ ! -s "$path" ]]; then
    echo "required family-capacity successor input is absent: $path" >&2
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

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT

if ! resolved=$(
  "$TASK_PYTHON" - \
    base "$BASE_OVERLAP_ROOT" "$BASE_GRAVITYSPY_CORPUS_AUDIT" \
      "$BASE_TRAIN_MANIFEST" "$BASE_VALIDATION_MANIFEST" \
    rare "$RARE_OVERLAP_ROOT" "$RARE_GRAVITYSPY_CORPUS_AUDIT" \
      "$RARE_TRAIN_MANIFEST" "$RARE_VALIDATION_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


eligible = []
values = sys.argv[1:]
if len(values) % 5:
    raise SystemExit("family-capacity candidate arguments are malformed")
for offset in range(0, len(values), 5):
    name, raw_root, raw_audit, raw_train, raw_validation = values[offset : offset + 5]
    root = pathlib.Path(raw_root).resolve()
    receipt_path = root / "source_safe_overlap_chain_receipt.json"
    if not receipt_path.is_file():
        continue
    audit = pathlib.Path(raw_audit).resolve()
    train = pathlib.Path(raw_train).resolve()
    validation = pathlib.Path(raw_validation).resolve()
    for path in (audit, train, validation):
        if not path.is_file():
            raise SystemExit(f"{name} completed without required corpus input: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "gravityspy_corpus_audit": audit,
        "overlap_train_manifest": root / "train-overlaps/physical_overlap_train_manifest.jsonl",
        "overlap_validation_manifest": root / "val-overlaps/physical_overlap_val_manifest.jsonl",
    }
    if (
        not str(receipt.get("status", "")).startswith("completed_source_safe_overlap_")
        or receipt.get("execution_passed") is not True
        or receipt.get("test_rows_read") != 0
    ):
        raise SystemExit(f"{name} overlap receipt is not an accepted validation-only result")
    for label, path in expected.items():
        identity = receipt.get("inputs", {}).get(label, {})
        if (
            not path.is_file()
            or pathlib.Path(identity.get("path", "")).resolve() != path.resolve()
            or identity.get("sha256") != digest(path)
        ):
            raise SystemExit(f"{name} overlap receipt failed {label} replay")
    summary_identity = receipt.get("five_seed_summary")
    summary_path = None
    summary_passed = False
    if summary_identity is not None:
        summary_path = pathlib.Path(summary_identity.get("path", "")).resolve()
        if not summary_path.is_file() or summary_identity.get("sha256") != digest(summary_path):
            raise SystemExit(f"{name} five-seed summary failed hash replay")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") != "completed_five_seed_source_safe_overlap_validation"
            or summary.get("test_data_opened") is not False
        ):
            raise SystemExit(f"{name} five-seed summary failed validation-boundary replay")
        summary_passed = summary.get("passed") is True
    eligible.append(
        (
            name,
            root,
            audit,
            train,
            validation,
            expected["overlap_train_manifest"],
            expected["overlap_validation_manifest"],
            summary_path,
            summary_passed,
        )
    )
if len(eligible) != 1:
    raise SystemExit(f"expected exactly one completed family-safe branch, found {len(eligible)}")
row = eligible[0]
print("\t".join("-" if value is None else str(value) for value in row))
PY
); then
  echo "family-capacity branch resolution failed" >&2
  exit 4
fi
IFS=$'\t' read -r \
  selected_branch selected_root selected_audit selected_train selected_validation \
  overlap_train overlap_validation five_seed_summary five_seed_passed <<<"$resolved"

ood_endpoint="$OOD_OUTPUT_ROOT/network_ood_validation_endpoint.json"
if [[ ! -s "$ood_endpoint" ]]; then
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    TRAIN_MANIFEST="$selected_train" \
    VALIDATION_MANIFEST="$selected_validation" \
    GRAVITYSPY_CORPUS_AUDIT="$selected_audit" \
    OUTPUT_ROOT="$OOD_OUTPUT_ROOT" \
    bash scripts/run_network_ood_validation.sh
fi
if [[ ! -s "$ood_endpoint" ]]; then
  echo "family-capacity OOD successor omitted its endpoint" >&2
  exit 5
fi

scaling_status=not_authorized_by_five_seed_gate
hard_status=not_run
scaling_summary="$SCALING_OUTPUT_ROOT/physical_overlap_data_scaling_summary.json"
hard_report="$HARD_ENDPOINT_OUTPUT_ROOT/physical_overlap_data_scaling_hard_endpoint_bound.json"
if [[ "$five_seed_passed" == True ]]; then
  if [[ ! -s "$scaling_summary" ]]; then
    env \
      TASK_PYTHON="$TASK_PYTHON" \
      TASK_CODE_DIR="$TASK_CODE_DIR" \
      GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
      OVERLAP_TRAIN_MANIFEST="$overlap_train" \
      OVERLAP_VALIDATION_MANIFEST="$overlap_validation" \
      GRAVITYSPY_CORPUS_AUDIT="$selected_audit" \
      CLEAN_TRAIN_MANIFEST="$CLEAN_TRAIN_MANIFEST" \
      CLEAN_VALIDATION_MANIFEST="$CLEAN_VALIDATION_MANIFEST" \
      PRETRAINED_CHECKPOINT="$PRETRAINED_CHECKPOINT" \
      CLEAN_VALIDATION_FEATURE_CACHE_DIR="${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" \
      OUTPUT_ROOT="$SCALING_OUTPUT_ROOT" \
      OVERLAP_SCALES="${OVERLAP_SCALES:-250 500 1000}" \
      OVERLAP_SCALE_SEEDS="${OVERLAP_SCALE_SEEDS:-20260728 20260729 20260730 20260731 20260732}" \
      bash scripts/run_physical_overlap_data_scaling.sh
  fi
  if [[ ! -s "$scaling_summary" ]]; then
    echo "family-capacity scaling successor omitted its summary" >&2
    exit 6
  fi
  scaling_status=completed

  next_scale=$(
    "$TASK_PYTHON" - "$scaling_summary" <<'PY'
import json
import pathlib
import sys


report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed_group_safe_physical_overlap_data_scaling_curve"
    or report.get("passed") is not True
    or report.get("test_rows_read") != 0
):
    raise SystemExit("scaling summary is not a complete validation-only curve")
lower, upper = (int(value) for value in report["promotion_data_doubling"])
maximum = max(int(value) for value in report["scales"])
upper_bound = int(2.5 * upper)
candidate = min(upper_bound, max(2 * upper, maximum + 1))
if lower <= 0 or upper / lower < 1.8 or candidate <= maximum:
    print("-")
else:
    print(candidate)
PY
  )
  if [[ "$next_scale" != - ]]; then
    if [[ ! -s "$hard_report" ]]; then
      env \
        TASK_PYTHON="$TASK_PYTHON" \
        TASK_CODE_DIR="$TASK_CODE_DIR" \
        GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
        SCALING_OUTPUT_ROOT="$SCALING_OUTPUT_ROOT" \
        OVERLAP_VALIDATION_MANIFEST="$overlap_validation" \
        GRAVITYSPY_CORPUS_AUDIT="$selected_audit" \
        OUTPUT_ROOT="$HARD_ENDPOINT_OUTPUT_ROOT" \
        NEXT_PHYSICAL_SCALE="$next_scale" \
        bash scripts/run_physical_overlap_scaling_hard_endpoint.sh
    fi
    if [[ ! -s "$hard_report" ]]; then
      echo "family-capacity hard-endpoint successor omitted its report" >&2
      exit 7
    fi
    hard_status=completed
  else
    hard_status=no_bounded_next_scale
  fi
fi

"$TASK_PYTHON" - \
  "$SUCCESSOR_RECEIPT" "$selected_branch" "$selected_root" "$selected_audit" \
  "$ood_endpoint" "$scaling_status" "$scaling_summary" "$hard_status" "$hard_report" \
  "$five_seed_summary" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

from gwyolo.io import atomic_write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


(
    output,
    branch,
    root,
    audit,
    ood,
    scaling_status,
    scaling,
    hard_status,
    hard,
    five_seed,
    commit,
) = sys.argv[1:]
artifacts = {}
for label, raw_path in (("corpus_audit", audit), ("ood_endpoint", ood)):
    path = pathlib.Path(raw_path).resolve()
    if not path.is_file():
        raise SystemExit(f"successor receipt input is absent: {label}")
    artifacts[label] = {"path": str(path), "sha256": digest(path)}
for label, status, raw_path in (
    ("scaling_summary", scaling_status, scaling),
    ("hard_endpoint", hard_status, hard),
):
    if status == "completed":
        path = pathlib.Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit(f"completed successor artifact is absent: {label}")
        artifacts[label] = {"path": str(path), "sha256": digest(path)}
result = {
    "status": "completed_family_capacity_scaling_ood_successor",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "validation-only OOD and scaling evidence cannot replace continuous-background "
        "FAR/IFAR/<VT> or the one-time locked evaluation"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "code_commit": commit,
    "selected_family_capacity_branch": branch,
    "selected_overlap_root": str(pathlib.Path(root).resolve()),
    "five_seed_summary": None if five_seed == "-" else five_seed,
    "scaling_status": scaling_status,
    "hard_endpoint_status": hard_status,
    "artifacts": artifacts,
}
atomic_write_json(output, result)
print(json.dumps(result, indent=2, sort_keys=True))
PY
