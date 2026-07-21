#!/usr/bin/env bash
set -euo pipefail

# Run validation-only DINGO and AMPLFI posterior batches from a completed
# common-input smoke, then execute the strict hash-bound joint evaluation.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PE_INPUT_ROOT
  DINGO_PYTHON
  DINGO_MODEL_METADATA
  DINGO_MODEL_INIT
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

dingo_native_report="$PE_INPUT_ROOT/dingo-native/native_conditioning_report.json"
amplfi_native_report="$PE_INPUT_ROOT/amplfi-native/native_conditioning_report.json"
smoke_summary="$PE_INPUT_ROOT/paired_pe_smoke_summary.json"
for path in \
  "$TASK_PYTHON" \
  "$DINGO_PYTHON" \
  "$AMPLFI_PYTHON" \
  "$DINGO_MODEL_METADATA" \
  "$DINGO_MODEL_INIT" \
  "$AMPLFI_MODEL_METADATA" \
  "$AMPLFI_NATIVE_PRIOR" \
  "$dingo_native_report" \
  "$amplfi_native_report" \
  "$smoke_summary"; do
  if [[ ! -s "$path" ]]; then
    echo "required joint PE artifact is absent: $path" >&2
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

readarray -t native_manifests < <(
  "$TASK_PYTHON" - "$smoke_summary" "$dingo_native_report" "$amplfi_native_report" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


summary_path, dingo_path, amplfi_path = map(pathlib.Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "paired_pe_native_inputs_smoke_complete":
    raise SystemExit("paired PE input smoke is incomplete")
reports = summary.get("reports", {})
for backend, path in (("dingo_native", dingo_path), ("amplfi_native", amplfi_path)):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report != reports.get(backend):
        raise SystemExit(f"{backend} report differs from the paired PE smoke summary")
    manifest = pathlib.Path(report["manifest_path"])
    if not manifest.is_file() or digest(manifest) != report["manifest_sha256"]:
        raise SystemExit(f"{backend} manifest hash mismatch")
    if report.get("run_identity", {}).get("required_split") != "val":
        raise SystemExit(f"{backend} is not validation-only")
    print(manifest)
PY
)
if [[ "${#native_manifests[@]}" -ne 2 ]]; then
  echo "joint PE native manifest resolution failed" >&2
  exit 4
fi

wait_for_idle_gpu() {
  while true; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
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
"$TASK_PYTHON" -m gwyolo.cli dingo-common-batch \
  --native-manifest "${native_manifests[0]}" \
  --model-metadata "$DINGO_MODEL_METADATA" \
  --model-init "$DINGO_MODEL_INIT" \
  --python-executable "$DINGO_PYTHON" \
  --runner-script scripts/run_dingo_common_event.py \
  --output-dir "$OUTPUT_ROOT/dingo" \
  --required-split val \
  --num-samples "${DINGO_NUM_SAMPLES:-10000}" \
  --batch-size "${DINGO_BATCH_SIZE:-1000}" \
  --num-gnpe-iterations "${DINGO_GNPE_ITERATIONS:-30}" \
  --device cuda \
  --seed "${PE_SEED:-20260721}" \
  >"$OUTPUT_ROOT/logs/dingo-batch.log" 2>&1

wait_for_idle_gpu
"$TASK_PYTHON" -m gwyolo.cli amplfi-common-batch \
  --native-manifest "${native_manifests[1]}" \
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

"$TASK_PYTHON" -m gwyolo.cli pe-robustness-joint-evaluate \
  --dingo-batch-report "$OUTPUT_ROOT/dingo/dingo_batch_report.json" \
  --amplfi-batch-report "$OUTPUT_ROOT/amplfi/amplfi_batch_report.json" \
  --manifest-output "$OUTPUT_ROOT/paired_pe_posteriors.jsonl" \
  --output "$OUTPUT_ROOT/paired_pe_robustness_report.json" \
  --credible-level "${PE_CREDIBLE_LEVEL:-0.9}" \
  --bootstrap-replicates "${PE_BOOTSTRAP_REPLICATES:-10000}" \
  --bootstrap-seed "${PE_BOOTSTRAP_SEED:-20260721}" \
  >"$OUTPUT_ROOT/logs/joint-evaluation.log" 2>&1

"$TASK_PYTHON" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys


root = pathlib.Path(sys.argv[1]).resolve()
artifacts = {
    "dingo_batch": root / "dingo/dingo_batch_report.json",
    "amplfi_batch": root / "amplfi/amplfi_batch_report.json",
    "joint_evaluation": root / "paired_pe_robustness_report.json",
}
identities = {}
for name, path in artifacts.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    identities[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "report": json.loads(path.read_text(encoding="utf-8")),
    }
joint = identities["joint_evaluation"]["report"]
if (
    joint.get("status")
    != "paired_dingo_amplfi_pe_robustness_evaluation_complete"
    or joint.get("dingo_amplfi_joint_gate") is not True
    or joint.get("cross_backend_matched_input_gate") is not True
):
    raise SystemExit("joint PE evaluation did not pass its mandatory input gates")
summary = {
    "status": "validation_only_joint_paired_pe_complete",
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "bounded validation PE must satisfy the frozen sample-size and promotion protocol "
        "before a locked test claim"
    ),
    "artifacts": identities,
}
target = root / "joint_paired_pe_summary.json"
temporary = target.with_suffix(".json.part")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
