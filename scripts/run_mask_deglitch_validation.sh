#!/usr/bin/env bash
set -euo pipefail

# Validation-only six-arm raw/mask-conditioned search comparison. The runner
# binds the selected five-seed model to an independent real-glitch overlap bank
# and a purpose-disjoint background before any cleaning or threshold fitting.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  MODEL_SELECTION_OVERLAP_MANIFEST
  MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  INDEPENDENT_PE_OVERLAP_REPORT
  INDEPENDENT_OVERLAP_AUDIT
  OVERLAP_MANIFEST
  BACKGROUND_MANIFEST
  INJECTION_MANIFEST
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

protocol_config=${PROTOCOL_CONFIG:-$TASK_CODE_DIR/configs/mask_deglitch_validation.yaml}
for path in \
  "$TASK_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$MODEL_SELECTION_OVERLAP_MANIFEST" \
  "$MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$INDEPENDENT_PE_OVERLAP_REPORT" \
  "$INDEPENDENT_OVERLAP_AUDIT" \
  "$OVERLAP_MANIFEST" \
  "$BACKGROUND_MANIFEST" \
  "$INJECTION_MANIFEST" \
  "$protocol_config"; do
  if [[ ! -s "$path" ]]; then
    echo "required mask-deglitch validation artifact is absent: $path" >&2
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
export CUDA_VISIBLE_DEVICES="${MASK_CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUTPUT_ROOT/logs"

if ! preflight=$(
  "$TASK_PYTHON" - \
    "$FIVE_SEED_SUMMARY" \
    "$UNIFORM_CONFIG" \
    "$FAMILY_BALANCED_CONFIG" \
    "$MODEL_SELECTION_OVERLAP_MANIFEST" \
    "$MODEL_SELECTION_CLEAN_VALIDATION_MANIFEST" \
    "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    "$INDEPENDENT_PE_OVERLAP_REPORT" \
    "$INDEPENDENT_OVERLAP_AUDIT" \
    "$OVERLAP_MANIFEST" \
    "$BACKGROUND_MANIFEST" \
    "$INJECTION_MANIFEST" \
    "$protocol_config" \
    "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys
import yaml


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    summary_path,
    uniform_config,
    balanced_config,
    selection_overlap,
    selection_clean,
    endpoint_path,
    independent_report_path,
    audit_path,
    overlap_path,
    background_path,
    injection_path,
    protocol_path,
    commit,
) = sys.argv[1:]
summary = json.loads(pathlib.Path(summary_path).read_text(encoding="utf-8"))
if (
    summary.get("status")
    != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("test_data_opened") is not False
):
    raise SystemExit("five-seed summary is not a validation-only promoted model")
selection_commit = str(summary.get("code_commit", ""))
if not selection_commit:
    raise SystemExit("five-seed summary omits its model-selection code commit")
arm = summary.get("promoted_arm")
if arm == "uniform":
    model_config = uniform_config
elif arm == "family_balanced":
    model_config = balanced_config
else:
    raise SystemExit("five-seed summary selected an unknown arm")
checkpoint = str(summary["selected_checkpoint_path"])
if digest(checkpoint) != str(summary["selected_checkpoint_sha256"]):
    raise SystemExit("five-seed selected checkpoint hash mismatch")
matches = []
for identity in summary.get("finetune_reports", []):
    path = str(identity["path"])
    if digest(path) != str(identity["sha256"]):
        raise SystemExit("five-seed finetune report hash mismatch")
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if str(report.get("code_commit", "")) != selection_commit:
        raise SystemExit("five-seed report differs from the selection code commit")
    if str(report.get("checkpoint_path")) == checkpoint:
        matches.append((path, report))
if len(matches) != 1:
    raise SystemExit("five-seed checkpoint does not resolve to one report")
model_report_path, model_report = matches[0]
expected_model = {
    "checkpoint_sha256": digest(checkpoint),
    "config_file_sha256": digest(model_config),
    "overlap_validation_manifest_sha256": digest(selection_overlap),
    "clean_validation_manifest_sha256": digest(selection_clean),
}
if (
    model_report.get("status")
    != "validation_selected_real_glitch_overlap_finetune"
    or model_report.get("code_commit") != selection_commit
    or any(model_report.get(key) != value for key, value in expected_model.items())
):
    raise SystemExit("selected overlap model differs from its frozen inputs")

endpoint = json.loads(pathlib.Path(endpoint_path).read_text(encoding="utf-8"))
components = endpoint.get("component_reports", {})
expected_components = {
    "purpose_partition",
    "injection_plan",
    "waveform_validation",
    "materialization",
    "snr_annotation",
    "arrival_annotation",
}
if (
    endpoint.get("status")
    != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or endpoint.get("passed") is not True
    or endpoint.get("test_rows_read") != 0
    or endpoint.get("test_evaluation") is not None
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or set(components) != expected_components
    or any(digest(item["path"]) != item["sha256"] for item in components.values())
    or pathlib.Path(
        endpoint["candidate_calibration_background_manifest_path"]
    ).resolve()
    != pathlib.Path(background_path).resolve()
    or endpoint.get("candidate_calibration_background_manifest_sha256")
    != digest(background_path)
    or pathlib.Path(endpoint["injection_arrival_manifest_path"]).resolve()
    != pathlib.Path(injection_path).resolve()
    or endpoint.get("injection_arrival_manifest_sha256") != digest(injection_path)
):
    raise SystemExit("mask validation inputs differ from the independent endpoint")

independent = json.loads(
    pathlib.Path(independent_report_path).read_text(encoding="utf-8")
)
if (
    independent.get("status") != "verified_independent_validation_pe_overlap"
    or independent.get("passed") is not True
    or independent.get("test_rows_read") != 0
    or independent.get("test_evaluation") is not None
    or independent.get("overlap_manifest_sha256") != digest(overlap_path)
    or independent.get("joint_overlap_audit_sha256") != digest(audit_path)
    or independent.get("independent_validation_endpoint_report_sha256")
    != digest(endpoint_path)
    or independent.get("injection_arrival_manifest_sha256")
    != digest(injection_path)
):
    raise SystemExit("independent overlap receipt failed replay")
audit = json.loads(pathlib.Path(audit_path).read_text(encoding="utf-8"))
cross = audit.get("cross_split_overlaps", {})
if (
    audit.get("status") != "passed_physical_overlap_group_audit"
    or audit.get("passed") is not True
    or audit.get("manifest_sha256_by_split", {}).get("val") != digest(overlap_path)
    or not cross
    or any(values for pair in cross.values() for values in pair.values())
):
    raise SystemExit("independent overlap audit does not prove zero leakage")

protocol = yaml.safe_load(pathlib.Path(protocol_path).read_text(encoding="utf-8"))
settings = protocol.get("mask_deglitch_validation")
if not isinstance(settings, dict):
    raise SystemExit("mask-deglitch protocol configuration is missing")
maximum_false = int(settings["maximum_validation_false_alarms"])
strength = float(settings["strength"])
clean_margin = float(settings["clean_noninferiority_margin"])
minimum_gain = float(settings["minimum_contaminated_efficiency_gain"])
replicates = int(settings["bootstrap_replicates"])
seed = int(settings["seed"])
model_ifos = [str(value) for value in settings["model_ifos"]]
q_values = [float(value) for value in settings["q_values"]]
sample_rate = int(settings["target_sample_rate"])
duration = float(settings["context_duration"])
if (
    maximum_false < 0
    or not 0 <= strength <= 1
    or not 0 <= clean_margin < 1
    or not 0 <= minimum_gain < 1
    or replicates < 1000
    or seed <= 0
    or not model_ifos
    or not q_values
    or sample_rate <= 0
    or duration <= 0
):
    raise SystemExit("mask-deglitch protocol values are invalid")
print(model_report_path)
print(model_config)
print(checkpoint)
print(maximum_false)
print(strength)
print(clean_margin)
print(minimum_gain)
print(replicates)
print(seed)
print(" ".join(model_ifos))
print(" ".join(str(value) for value in q_values))
print(sample_rate)
print(duration)
PY
); then
  echo "mask-deglitch validation preflight failed" >&2
  exit 4
fi
readarray -t resolved <<<"$preflight"
if (( ${#resolved[@]} != 13 )); then
  echo "mask-deglitch validation preflight returned an invalid result" >&2
  exit 4
fi
model_report=${resolved[0]}
model_config=${resolved[1]}
checkpoint=${resolved[2]}
maximum_false=${resolved[3]}
strength=${resolved[4]}
clean_margin=${resolved[5]}
minimum_gain=${resolved[6]}
replicates=${resolved[7]}
seed=${resolved[8]}
read -r -a model_ifos <<<"${resolved[9]}"
read -r -a q_values <<<"${resolved[10]}"
sample_rate=${resolved[11]}
duration=${resolved[12]}

contamination_root="$OUTPUT_ROOT/contamination"
contamination_report="$contamination_root/contaminated_injection_report.json"
if [[ ! -s "$contamination_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli physical-overlap-contamination \
    --overlap-manifest "$OVERLAP_MANIFEST" \
    --injection-manifest "$INJECTION_MANIFEST" \
    --output-dir "$contamination_root" \
    --required-split val \
    >"$OUTPUT_ROOT/logs/contamination.log" 2>&1
fi
if ! contamination_paths=$(
  "$TASK_PYTHON" - \
    "$contamination_report" \
    "$OVERLAP_MANIFEST" \
    "$INJECTION_MANIFEST" \
    "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


report_path, overlap_path, injection_path, commit = sys.argv[1:]
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
contaminated = pathlib.Path(report.get("manifest_path", ""))
clean = pathlib.Path(report.get("paired_clean_manifest_path", ""))
if (
    report.get("status")
    != "verified_real_glitch_contaminated_injection_overrides"
    or report.get("scientific_claim_allowed") is not False
    or report.get("split") != "val"
    or report.get("code_commit") != commit
    or int(report.get("rows", 0)) < 100
    or report.get("overlap_manifest_sha256") != digest(overlap_path)
    or report.get("injection_manifest_sha256") != digest(injection_path)
    or not contaminated.is_file()
    or report.get("manifest_sha256") != digest(contaminated)
    or not clean.is_file()
    or report.get("paired_clean_manifest_sha256") != digest(clean)
):
    raise SystemExit("contamination report failed replay")
print(clean)
print(contaminated)
PY
); then
  echo "mask-deglitch contamination replay failed" >&2
  exit 5
fi
readarray -t contamination <<<"$contamination_paths"
if (( ${#contamination[@]} != 2 )); then
  echo "mask-deglitch contamination paths are invalid" >&2
  exit 5
fi

while :; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$gpu_pids" ]] && break
  sleep 30
done

pipeline_root="$OUTPUT_ROOT/pipeline"
"$TASK_PYTHON" -m gwyolo.cli mask-search-validation-pipeline \
  --background-manifest "$BACKGROUND_MANIFEST" \
  --clean-injection-manifest "${contamination[0]}" \
  --contaminated-injection-manifest "${contamination[1]}" \
  --checkpoint "$checkpoint" \
  --config "$model_config" \
  --output-dir "$pipeline_root" \
  --maximum-validation-false-alarms "$maximum_false" \
  --strength "$strength" \
  --clean-noninferiority-margin "$clean_margin" \
  --minimum-contaminated-efficiency-gain "$minimum_gain" \
  --bootstrap-replicates "$replicates" \
  --seed "$seed" \
  --model-ifos "${model_ifos[@]}" \
  --q-values "${q_values[@]}" \
  --target-sample-rate "$sample_rate" \
  --context-duration "$duration" \
  >"$OUTPUT_ROOT/logs/pipeline.log" 2>&1

pipeline_report="$pipeline_root/mask_search_pipeline_report.json"
"$TASK_PYTHON" - \
  "$pipeline_report" \
  "$contamination_report" \
  "$FIVE_SEED_SUMMARY" \
  "$model_report" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$INDEPENDENT_PE_OVERLAP_REPORT" \
  "$INDEPENDENT_OVERLAP_AUDIT" \
  "$protocol_config" \
  "$BACKGROUND_MANIFEST" \
  "${contamination[0]}" \
  "${contamination[1]}" \
  "$OUTPUT_ROOT/mask_deglitch_validation_receipt.json" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


paths = [pathlib.Path(value) for value in sys.argv[1:12]]
output = pathlib.Path(sys.argv[12])
commit = sys.argv[13]
pipeline = json.loads(paths[0].read_text(encoding="utf-8"))
expected_inputs = {
    "background": {"path": str(paths[8]), "sha256": digest(paths[8])},
    "clean_injections": {"path": str(paths[9]), "sha256": digest(paths[9])},
    "contaminated_injections": {
        "path": str(paths[10]),
        "sha256": digest(paths[10]),
    },
}
if (
    pipeline.get("status") != "validation_only_end_to_end_mask_search_pipeline"
    or pipeline.get("scientific_claim_allowed") is not False
    or pipeline.get("promotion_allowed") is not False
    or pipeline.get("test_rows_read") != 0
    or pipeline.get("test_evaluation") is not None
    or pipeline.get("code_commit") != commit
    or pipeline.get("input_manifests") != expected_inputs
    or digest(pipeline["checkpoint_path"]) != pipeline.get("checkpoint_sha256")
    or digest(pipeline["config_path"]) != pipeline.get("config_sha256")
    or digest(pipeline["comparison_path"]) != pipeline.get("comparison_sha256")
):
    raise SystemExit("mask-deglitch pipeline provenance replay failed")
identities = {
    name: {"path": str(path), "sha256": digest(path)}
    for name, path in zip(
        (
            "pipeline_report",
            "contamination_report",
            "five_seed_summary",
            "selected_model_report",
            "independent_validation_endpoint",
            "independent_pe_overlap_report",
            "independent_overlap_audit",
            "protocol_config",
        ),
        paths[:8],
    )
}
receipt = {
    "status": "completed_validation_only_mask_deglitch_gate",
    "execution_passed": True,
    "development_gates_passed": pipeline.get("development_gates_passed") is True,
    "scale_mask_conditioned_morphology_background": (
        pipeline.get("development_gates_passed") is True
    ),
    "coherent_background_scale_allowed": False,
    "scientific_claim_allowed": False,
    "locked_test_allowed": False,
    "test_rows_read": 0,
    "code_commit": commit,
    "model_selection_code_commit": json.loads(
        paths[2].read_text(encoding="utf-8")
    )["code_commit"],
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
    "artifacts": identities,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
