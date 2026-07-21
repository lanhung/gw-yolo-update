#!/usr/bin/env bash
set -euo pipefail

# Evaluate the official DINGO release on validation-only paired conditions under
# its native prior/waveform. This is a within-backend robustness smoke and must
# never be joined to the common-prior DINGO/AMPLFI absolute comparison.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  PE_INPUT_ROOT
  DINGO_PYTHON
  DINGO_SOURCE_CONFIG
  DINGO_ACQUISITION_REPORT
  DINGO_MODEL_LOAD_RECEIPT
  DINGO_NATIVE_CONDITIONING_CONFIG
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

dingo_native_report="$PE_INPUT_ROOT/dingo-native/native_conditioning_report.json"
smoke_summary="$PE_INPUT_ROOT/paired_pe_smoke_summary.json"
for path in \
  "$TASK_PYTHON" \
  "$DINGO_PYTHON" \
  "$DINGO_SOURCE_CONFIG" \
  "$DINGO_ACQUISITION_REPORT" \
  "$DINGO_MODEL_LOAD_RECEIPT" \
  "$DINGO_NATIVE_CONDITIONING_CONFIG" \
  "$dingo_native_report" \
  "$smoke_summary"; do
  if [[ ! -s "$path" ]]; then
    echo "required official-native DINGO artifact is absent: $path" >&2
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

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
export CUDA_VISIBLE_DEVICES="${PE_CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_ROOT/logs"
metadata="$OUTPUT_ROOT/dingo_official_native_model_metadata.json"

"$TASK_PYTHON" -m gwyolo.cli dingo-official-native-model-freeze \
  --source-config "$DINGO_SOURCE_CONFIG" \
  --acquisition-report "$DINGO_ACQUISITION_REPORT" \
  --model-load-receipt "$DINGO_MODEL_LOAD_RECEIPT" \
  --native-conditioning-config "$DINGO_NATIVE_CONDITIONING_CONFIG" \
  --output "$metadata" \
  >"$OUTPUT_ROOT/logs/model-metadata.log" 2>&1

if ! resolved_output=$(
  "$TASK_PYTHON" - "$smoke_summary" "$dingo_native_report" "$metadata" <<'PY'
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


summary_path, native_report_path, metadata_path = map(pathlib.Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
native_report = json.loads(native_report_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if summary.get("status") != "paired_pe_native_inputs_smoke_complete":
    raise SystemExit("paired PE input smoke is incomplete")
if native_report != summary.get("reports", {}).get("dingo_native"):
    raise SystemExit("DINGO native report differs from paired PE smoke summary")
manifest = pathlib.Path(native_report["manifest_path"])
if not manifest.is_file() or digest(manifest) != native_report.get("manifest_sha256"):
    raise SystemExit("DINGO native manifest hash mismatch")
if native_report.get("run_identity", {}).get("required_split") != "val":
    raise SystemExit("DINGO native manifest is not validation-only")
if (
    metadata.get("status") != "verified_official_dingo_native_model_metadata"
    or metadata.get("within_backend_paired_robustness_allowed") is not True
    or metadata.get("cross_backend_absolute_comparison_allowed") is not False
    or metadata.get("common_prior_equivalent") is not False
):
    raise SystemExit("official DINGO native comparison boundary is absent")
for label in ("training_settings", "initialization_model"):
    artifact = metadata.get("artifacts", {}).get(label, {})
    path = pathlib.Path(artifact.get("path", ""))
    if not path.is_file() or digest(path) != artifact.get("sha256"):
        raise SystemExit(f"official DINGO metadata artifact failed replay: {label}")
print(manifest)
print(metadata["artifacts"]["training_settings"]["path"])
print(metadata["artifacts"]["initialization_model"]["path"])
print(metadata["load_runtime_version"])
PY
); then
  echo "official-native DINGO input resolution failed" >&2
  exit 4
fi
readarray -t resolved <<<"$resolved_output"
if (( ${#resolved[@]} != 4 )) || [[ -z "${resolved[0]}" || -z "${resolved[3]}" ]]; then
  echo "official-native DINGO input resolution failed" >&2
  exit 4
fi

runtime_version=$(
  "$DINGO_PYTHON" -c 'import dingo; print(dingo.__version__)'
)
if [[ "$runtime_version" != "${resolved[3]}" ]]; then
  echo "DINGO runtime version differs from the passing model-load receipt" >&2
  exit 5
fi

wait_for_idle_gpu() {
  while true; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && return
    sleep 30
  done
}

wait_for_idle_gpu
"$TASK_PYTHON" -m gwyolo.cli dingo-common-batch \
  --native-manifest "${resolved[0]}" \
  --model-metadata "$metadata" \
  --native-prior "${resolved[1]}" \
  --model-init "${resolved[2]}" \
  --python-executable "$DINGO_PYTHON" \
  --runner-script scripts/run_dingo_common_event.py \
  --output-dir "$OUTPUT_ROOT/dingo" \
  --required-split val \
  --num-samples "${DINGO_NUM_SAMPLES:-10000}" \
  --batch-size "${DINGO_BATCH_SIZE:-1000}" \
  --num-gnpe-iterations "${DINGO_GNPE_ITERATIONS:-30}" \
  --device cuda \
  --seed "${PE_SEED:-20260721}" \
  --comparison-mode official_native \
  >"$OUTPUT_ROOT/logs/dingo-native-batch.log" 2>&1

"$TASK_PYTHON" -m gwyolo.cli pe-robustness-evaluate \
  --manifest "$OUTPUT_ROOT/dingo/dingo_posterior_manifest.jsonl" \
  --output "$OUTPUT_ROOT/dingo_official_native_robustness.json" \
  --credible-level "${PE_CREDIBLE_LEVEL:-0.9}" \
  --bootstrap-replicates "${PE_BOOTSTRAP_REPLICATES:-2000}" \
  --bootstrap-seed "${PE_BOOTSTRAP_SEED:-20260721}" \
  >"$OUTPUT_ROOT/logs/dingo-native-evaluation.log" 2>&1

"$TASK_PYTHON" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys


root = pathlib.Path(sys.argv[1]).resolve()
paths = {
    "model_metadata": root / "dingo_official_native_model_metadata.json",
    "posterior_batch": root / "dingo/dingo_batch_report.json",
    "robustness": root / "dingo_official_native_robustness.json",
}
artifacts = {}
for label, path in paths.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    artifacts[label] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
metadata = json.loads(paths["model_metadata"].read_text(encoding="utf-8"))
batch = json.loads(paths["posterior_batch"].read_text(encoding="utf-8"))
if (
    metadata.get("cross_backend_absolute_comparison_allowed") is not False
    or batch.get("status")
    != "real_dingo_official_native_paired_robustness_batch_complete"
    or batch.get("run_identity", {}).get("comparison_mode") != "official_native"
):
    raise SystemExit("official-native DINGO smoke violated its comparison boundary")
result = {
    "status": "validation_only_dingo_official_native_paired_smoke_complete",
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "three-event smoke under the official native prior/waveform; it is not a "
        "common-prior DINGO/AMPLFI absolute comparison"
    ),
    "cross_backend_absolute_comparison_allowed": False,
    "artifacts": artifacts,
}
target = root / "dingo_official_native_paired_smoke_summary.json"
temporary = target.with_suffix(".json.part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
