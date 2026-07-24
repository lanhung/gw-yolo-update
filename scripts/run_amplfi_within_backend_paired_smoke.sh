#!/usr/bin/env bash
set -euo pipefail

# Run a strict validation-only AMPLFI clean/contaminated/mask-conditioned
# posterior smoke without implying an absolute comparison to DINGO.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PE_INPUT_ROOT
  AMPLFI_PYTHON
  AMPLFI_MODEL_METADATA
  AMPLFI_NATIVE_PRIOR
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

native_report="$PE_INPUT_ROOT/amplfi-native/native_conditioning_report.json"
smoke_summary="$PE_INPUT_ROOT/paired_pe_smoke_summary.json"
for path in \
  "$TASK_PYTHON" \
  "$AMPLFI_PYTHON" \
  "$AMPLFI_MODEL_METADATA" \
  "$AMPLFI_NATIVE_PRIOR" \
  "$native_report" \
  "$smoke_summary"; do
  if [[ ! -s "$path" ]]; then
    echo "required AMPLFI paired artifact is absent: $path" >&2
    exit 3
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

if ! native_manifest=$(
  "$TASK_PYTHON" - "$smoke_summary" "$native_report" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


summary_path, report_path = map(pathlib.Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
report = json.loads(report_path.read_text(encoding="utf-8"))
if summary.get("status") != "paired_pe_native_inputs_smoke_complete":
    raise SystemExit("paired PE input smoke is incomplete")
if report != summary.get("reports", {}).get("amplfi_native"):
    raise SystemExit("AMPLFI native report differs from the paired PE smoke summary")
manifest = pathlib.Path(report["manifest_path"])
if not manifest.is_file() or digest(manifest) != report.get("manifest_sha256"):
    raise SystemExit("AMPLFI native manifest hash mismatch")
if report.get("run_identity", {}).get("required_split") != "val":
    raise SystemExit("AMPLFI paired smoke is not validation-only")
print(manifest)
PY
); then
  echo "AMPLFI paired native manifest resolution failed" >&2
  exit 4
fi
if [[ -z "$native_manifest" ]]; then
  echo "AMPLFI paired native manifest resolution failed" >&2
  exit 4
fi

wait_for_idle_gpu() {
  gpu_query_args=()
  if [[ -n "${GWYOLO_ASSIGNED_GPU_INDEX:-}" ]]; then
    if [[ ! "$GWYOLO_ASSIGNED_GPU_INDEX" =~ ^[0-9]+$ ]]; then
      echo "GWYOLO_ASSIGNED_GPU_INDEX must be a non-negative integer" >&2
      exit 2
    fi
    if [[ "${PE_CUDA_VISIBLE_DEVICES:-0}" != "$GWYOLO_ASSIGNED_GPU_INDEX" ]]; then
      echo "assigned PE GPU index and PE_CUDA_VISIBLE_DEVICES differ" >&2
      exit 2
    fi
    gpu_query_args=(-i "$GWYOLO_ASSIGNED_GPU_INDEX")
  fi
  while true; do
    gpu_pids=$(
      nvidia-smi "${gpu_query_args[@]}" \
        --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d' || true
    )
    [[ -z "$gpu_pids" ]] && return
    sleep 30
  done
}

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
export CUDA_VISIBLE_DEVICES="${PE_CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_ROOT/logs"

wait_for_idle_gpu
"$TASK_PYTHON" -m gwyolo.cli amplfi-common-batch \
  --native-manifest "$native_manifest" \
  --model-metadata "$AMPLFI_MODEL_METADATA" \
  --native-prior "$AMPLFI_NATIVE_PRIOR" \
  --python-executable "$AMPLFI_PYTHON" \
  --runner-script scripts/run_amplfi_common_event.py \
  --output-dir "$OUTPUT_ROOT/amplfi" \
  --required-split val \
  --num-samples "${AMPLFI_NUM_SAMPLES:-10000}" \
  --sample-batch-size "${AMPLFI_BATCH_SIZE:-1000}" \
  --device cuda \
  --seed "${PE_SEED:-20260721}" \
  >"$OUTPUT_ROOT/logs/amplfi-batch.log" 2>&1

"$TASK_PYTHON" -m gwyolo.cli pe-robustness-evaluate \
  --manifest "$OUTPUT_ROOT/amplfi/amplfi_posterior_manifest.jsonl" \
  --output "$OUTPUT_ROOT/amplfi_within_backend_robustness.json" \
  --credible-level "${PE_CREDIBLE_LEVEL:-0.9}" \
  --bootstrap-replicates "${PE_BOOTSTRAP_REPLICATES:-10000}" \
  --bootstrap-seed "${PE_BOOTSTRAP_SEED:-20260721}" \
  --within-backend-only \
  >"$OUTPUT_ROOT/logs/amplfi-evaluation.log" 2>&1

"$TASK_PYTHON" - "$OUTPUT_ROOT" "$GWYOLO_CODE_COMMIT" "$AMPLFI_MODEL_METADATA" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


root = pathlib.Path(sys.argv[1]).resolve()
commit = sys.argv[2]
metadata_path = pathlib.Path(sys.argv[3]).resolve()
paths = {
    "model_metadata": metadata_path,
    "posterior_batch": root / "amplfi/amplfi_batch_report.json",
    "robustness": root / "amplfi_within_backend_robustness.json",
}
batch = json.loads(paths["posterior_batch"].read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
robustness = json.loads(paths["robustness"].read_text(encoding="utf-8"))
manifest = pathlib.Path(batch.get("manifest_path", "")).resolve()
if (
    metadata.get("backend") != "AMPLFI"
    or metadata.get("selection_split") != "validation"
    or batch.get("status") != "real_amplfi_common_batch_complete"
    or batch.get("run_identity", {}).get("required_split") != "val"
    or not manifest.is_file()
    or batch.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest()
    or robustness.get("comparison_scope") != "strict_within_backend_paired"
    or robustness.get("within_backend_provenance_gate") is not True
    or robustness.get("cross_backend_matched_input_gate") is not False
    or robustness.get("dingo_amplfi_joint_gate") is not False
    or robustness.get("publication_provenance_required") is not True
    or robustness.get("bootstrap_replicates", 0) < 10000
    or pathlib.Path(robustness.get("manifest_path", "")).resolve() != manifest
    or robustness.get("manifest_sha256") != batch.get("manifest_sha256")
):
    raise SystemExit("AMPLFI within-backend smoke violated its evidence boundary")
with manifest.open("r", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
if (
    not rows
    or any(row.get("split") != "val" for row in rows)
    or any(row.get("backend") != "AMPLFI" for row in rows)
    or len(rows) != int(batch.get("rows", -1))
):
    raise SystemExit("AMPLFI within-backend posterior manifest is not validation-only")
paired_injections = int(batch["paired_injections"])
bootstrap_replicates = int(robustness["bootstrap_replicates"])
minimum_publication_validation_injections = 100
evaluation_tier = (
    "publication_validation"
    if paired_injections >= minimum_publication_validation_injections
    and bootstrap_replicates >= 10000
    else "bounded_smoke"
)
if evaluation_tier == "publication_validation":
    blocker = (
        "validation-only within-AMPLFI deltas meet the predeclared event/bootstrap "
        "floors; portfolio promotion and locked evaluation remain required, and "
        "absolute DINGO/AMPLFI ranking remains forbidden"
    )
else:
    blocker = (
        "bounded within-AMPLFI smoke is below the predeclared 100-injection and/or "
        "10000-bootstrap publication floors; it is not an absolute DINGO/AMPLFI "
        "comparison"
    )
artifacts = {
    label: {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for label, path in paths.items()
}
result = {
    "status": "validation_only_amplfi_within_backend_paired_smoke_complete",
    "scientific_claim_allowed": False,
    "scientific_blocker": blocker,
    "evaluation_tier": evaluation_tier,
    "minimum_publication_validation_injections": (
        minimum_publication_validation_injections
    ),
    "comparison_scope": "strict_within_backend_paired",
    "cross_backend_absolute_comparison_allowed": False,
    "test_rows_read": 0,
    "assigned_gpu_index": os.environ.get("GWYOLO_ASSIGNED_GPU_INDEX"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "code_commit": commit,
    "paired_injections": paired_injections,
    "bootstrap_replicates": bootstrap_replicates,
    "artifacts": artifacts,
}
target = root / "amplfi_within_backend_paired_smoke_summary.json"
temporary = target.with_suffix(".json.part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
