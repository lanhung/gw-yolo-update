#!/usr/bin/env bash
set -euo pipefail

# Import the content-addressed detector-set training bundle, prove that the
# variable-H1/L1/V1 train/validation split remains group-safe after path
# projection, then run exactly one validation-selected glitch-adapter seed.
# This is a robustness arm, not a same-distribution data-scaling result.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BUNDLE_RECEIPT
  IMPORT_ROOT
  OUTPUT_ROOT
  QUEUE_RECEIPT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required portable detector-set training variable is unset: $variable" >&2
    exit 2
  fi
done
for path in "$TASK_PYTHON" "$BUNDLE_RECEIPT"; do
  if [[ ! -s "$path" ]]; then
    echo "portable detector-set training input is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "portable detector-set training checkout differs from its declared commit" >&2
  exit 3
fi

seed=${TRAINING_SEED:-20260720}
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "TRAINING_SEED must be a non-negative integer" >&2
  exit 2
fi
assigned_gpu=${GWYOLO_ASSIGNED_GPU_INDEX:-0}
if [[ ! "$assigned_gpu" =~ ^[0-9]+$ ]]; then
  echo "GWYOLO_ASSIGNED_GPU_INDEX must be a non-negative integer" >&2
  exit 2
fi

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
mkdir -p "$IMPORT_ROOT" "$OUTPUT_ROOT"
inputs="$IMPORT_ROOT/detector_set_training_inputs.json"
if [[ ! -s "$inputs" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli detector-set-training-bundle-import \
    --bundle-receipt "$BUNDLE_RECEIPT" \
    --output-dir "$IMPORT_ROOT"
fi

preflight="$OUTPUT_ROOT/detector_set_training_preflight.json"
if [[ ! -s "$preflight" ]]; then
  "$TASK_PYTHON" - "$inputs" "$BUNDLE_RECEIPT" "$preflight" \
    "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


inputs_path, bundle_path, output_path, commit = map(pathlib.Path, sys.argv[1:])
projected = json.loads(inputs_path.read_text(encoding="utf-8"))
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
if (
    projected.get("status") != "projected_detector_set_training_input_bundle"
    or projected.get("passed") is not True
    or projected.get("detector_complete_clean_training_authorized") is not False
    or projected.get("test_rows_read") != 0
    or projected.get("test_evaluation") is not None
    or projected.get("bundle_receipt", {}).get("sha256") != digest(bundle_path)
    or projected.get("object_count") != bundle.get("object_count")
    or projected.get("object_bytes") != bundle.get("object_bytes")
):
    raise SystemExit("projected detector-set training inputs failed replay")

manifests = projected.get("manifests", {})
expected = {
    "overlap_train",
    "overlap_validation",
    "clean_train",
    "clean_validation",
}
if set(manifests) != expected:
    raise SystemExit("projected detector-set manifest inventory is incomplete")


def load_manifest(label):
    identity = manifests[label]
    path = pathlib.Path(identity["path"]).resolve()
    if not path.is_file() or digest(path) != identity["sha256"]:
        raise SystemExit(f"projected detector-set manifest drift: {label}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != identity["rows"] or len(rows) != bundle["manifests"][label]["rows"]:
        raise SystemExit(f"projected detector-set row-count drift: {label}")
    return rows


overlap_train = load_manifest("overlap_train")
overlap_validation = load_manifest("overlap_validation")
clean_train = load_manifest("clean_train")
clean_validation = load_manifest("clean_validation")
if len(overlap_train) <= 1170:
    raise SystemExit("portable arm did not expand beyond the original H1/L1 capacity")

allowed_subsets = {"H1L1", "H1L1V1", "H1V1", "L1V1"}
subset_counts = {}
for label, rows, split in (
    ("train", overlap_train, "train"),
    ("val", overlap_validation, "val"),
):
    if any(row.get("split") != split for row in rows):
        raise SystemExit(f"{label} overlap manifest crosses its frozen split")
    counts = Counter("".join(row.get("available_ifos", [])) for row in rows)
    if set(counts) != allowed_subsets:
        raise SystemExit(f"{label} overlap manifest lacks a predeclared detector subset")
    subset_counts[label] = dict(sorted(counts.items()))

leakage_fields = (
    "mixture_id",
    "injection_id",
    "waveform_id",
    "glitch_id",
    "injection_gps_block",
    "network_gps_block",
)
cross_split_overlaps = {}
for field in leakage_fields:
    overlap = sorted(
        {str(row[field]) for row in overlap_train}
        & {str(row[field]) for row in overlap_validation}
    )
    cross_split_overlaps[field] = overlap
if any(cross_split_overlaps.values()):
    raise SystemExit("portable detector-set overlap manifests contain split leakage")

for label, rows, split in (
    ("clean_train", clean_train, "train"),
    ("clean_validation", clean_validation, "val"),
):
    if any(row.get("split") != split for row in rows):
        raise SystemExit(f"{label} crosses its frozen split")
    if any(set(row.get("ifos", [])) != {"H1", "L1"} for row in rows):
        raise SystemExit(f"{label} is not the declared H1/L1-only clean control")

result = {
    "status": "verified_portable_detector_set_training_preflight",
    "passed": True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "same_distribution_data_scaling_claim_allowed": False,
    "detector_complete_clean_training_authorized": False,
    "training_scope": (
        "variable-detector physical-overlap training plus H1/L1-only clean regularization"
    ),
    "scientific_blocker": (
        "detector-complete empirical-noise clean training, continuous-background "
        "FAR/IFAR/<VT>, five seeds and locked evaluation remain required"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "rows": {
        "overlap_train": len(overlap_train),
        "overlap_validation": len(overlap_validation),
        "clean_train": len(clean_train),
        "clean_validation": len(clean_validation),
    },
    "detector_subset_counts": subset_counts,
    "cross_split_overlaps": cross_split_overlaps,
    "inputs": {
        "projected_receipt": {
            "path": str(inputs_path.resolve()),
            "sha256": digest(inputs_path),
        },
        "bundle_receipt": {
            "path": str(bundle_path.resolve()),
            "sha256": digest(bundle_path),
        },
    },
    "code_commit": str(commit),
}
target = pathlib.Path(output_path)
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, target)
PY
fi

config=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["configs"]["finetune"])' \
    "$inputs"
)
overlap_train=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["manifests"]["overlap_train"]["path"])' \
    "$inputs"
)
overlap_validation=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["manifests"]["overlap_validation"]["path"])' \
    "$inputs"
)
clean_train=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["manifests"]["clean_train"]["path"])' \
    "$inputs"
)
clean_validation=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["manifests"]["clean_validation"]["path"])' \
    "$inputs"
)
checkpoint=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["path"])' \
    "$inputs"
)

run="$OUTPUT_ROOT/seed${seed}"
report="$run/overlap_finetune_report.json"
if [[ ! -s "$report" ]]; then
  while :; do
    gpu_pids=$(
      nvidia-smi -i "$assigned_gpu" \
        --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d' || true
    )
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  cache_args=()
  if [[ -n "${CLEAN_VALIDATION_FEATURE_CACHE_DIR:-}" ]]; then
    cache_args=(
      --clean-validation-feature-cache-dir
      "$CLEAN_VALIDATION_FEATURE_CACHE_DIR"
    )
  fi
  env CUDA_VISIBLE_DEVICES="$assigned_gpu" \
    "$TASK_PYTHON" -m gwyolo.cli physical-overlap-finetune \
    --config "$config" \
    --overlap-train-manifest "$overlap_train" \
    --overlap-validation-manifest "$overlap_validation" \
    --clean-train-manifest "$clean_train" \
    --clean-validation-manifest "$clean_validation" \
    --pretrained-checkpoint "$checkpoint" \
    "${cache_args[@]}" \
    --output-dir "$run" \
    --seed "$seed"
fi

"$TASK_PYTHON" - "$QUEUE_RECEIPT" "$inputs" "$preflight" "$report" \
  "$GWYOLO_CODE_COMMIT" "$seed" "$assigned_gpu" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


target_value, inputs_value, preflight_value, report_value, commit, raw_seed, gpu = (
    sys.argv[1:]
)
inputs_path = pathlib.Path(inputs_value).resolve()
preflight_path = pathlib.Path(preflight_value).resolve()
report_path = pathlib.Path(report_value).resolve()
inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
report = json.loads(report_path.read_text(encoding="utf-8"))
seed = int(raw_seed)
scope = report.get("training_scope", {})
if (
    preflight.get("status") != "verified_portable_detector_set_training_preflight"
    or preflight.get("passed") is not True
    or report.get("status") != "validation_selected_real_glitch_overlap_finetune"
    or report.get("seed") != seed
    or report.get("code_commit") != commit
    or report.get("teacher_architecture") != "detector_set"
    or report.get("checkpoint_selection_metric") != "validation_loss"
    or report.get("split_audit", {}).get("passed") is not True
    or report.get("clean_split_audit", {}).get("passed") is not True
    or scope.get("scope") != "glitch_adapter_only"
):
    raise SystemExit("portable detector-set training report failed terminal replay")
identity = report.get("run_identity", {})
expected_hashes = {
    "overlap_train_manifest_sha256": inputs["manifests"]["overlap_train"]["sha256"],
    "overlap_validation_manifest_sha256": inputs["manifests"]["overlap_validation"][
        "sha256"
    ],
    "clean_train_manifest_sha256": inputs["manifests"]["clean_train"]["sha256"],
    "clean_validation_manifest_sha256": inputs["manifests"]["clean_validation"][
        "sha256"
    ],
    "pretrained_checkpoint_sha256": inputs["checkpoint"]["sha256"],
}
if any(identity.get(key) != value for key, value in expected_hashes.items()):
    raise SystemExit("portable detector-set training identity differs from its import")
checkpoint = pathlib.Path(report["checkpoint_path"]).resolve()
if not checkpoint.is_file() or digest(checkpoint) != report["checkpoint_sha256"]:
    raise SystemExit("portable detector-set selected checkpoint drifted")

result = {
    "status": "completed_variable_detector_overlap_training_seed",
    "execution_passed": True,
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "same_distribution_data_scaling_claim_allowed": False,
    "detector_complete_clean_training_authorized": False,
    "training_scope": preflight["training_scope"],
    "scientific_blocker": (
        "one validation-selected robustness seed cannot support a search, "
        "same-distribution scaling, five-seed or locked-evaluation claim"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "seed": seed,
    "assigned_gpu_index": int(gpu),
    "artifacts": {
        "projected_inputs": {
            "path": str(inputs_path),
            "sha256": digest(inputs_path),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": digest(preflight_path),
        },
        "training_report": {
            "path": str(report_path),
            "sha256": digest(report_path),
        },
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": report["checkpoint_sha256"],
        },
    },
    "validation_summary": {
        "best_epoch": report["best_epoch"],
        "best_validation_overlap_mean_iou": report[
            "best_validation_overlap_mean_iou"
        ],
        "calibrated_overlap_validation": report["calibrated_overlap_validation"],
        "calibrated_clean_validation": report["calibrated_clean_validation"],
        "validation_selected_thresholds": report["validation_selected_thresholds"],
    },
    "code_commit": commit,
}
target = pathlib.Path(target_value)
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("portable detector-set training receipt has another identity")
else:
    part = target.with_suffix(target.suffix + ".part")
    part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
