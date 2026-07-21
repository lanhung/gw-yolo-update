#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  MODEL_SELECTION_OVERLAP_MANIFEST
  MODEL_SELECTION_VALIDATION_MANIFEST
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  INDEPENDENT_PE_OVERLAP_REPORT
  INDEPENDENT_OVERLAP_AUDIT
  OVERLAP_MANIFEST
  INJECTION_MANIFEST
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
  "$FIVE_SEED_SUMMARY" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$MODEL_SELECTION_OVERLAP_MANIFEST" \
  "$MODEL_SELECTION_VALIDATION_MANIFEST" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$INDEPENDENT_PE_OVERLAP_REPORT" \
  "$INDEPENDENT_OVERLAP_AUDIT" \
  "$OVERLAP_MANIFEST" \
  "$INJECTION_MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "required promoted PE input is absent: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 2
fi

if ! selection_output=$(
  "$TASK_PYTHON" - \
    "$FIVE_SEED_SUMMARY" \
    "$UNIFORM_CONFIG" \
    "$FAMILY_BALANCED_CONFIG" \
    "$MODEL_SELECTION_OVERLAP_MANIFEST" \
    "$MODEL_SELECTION_VALIDATION_MANIFEST" <<'PY'
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


summary_path, uniform, balanced, selection_overlap, selection_validation = sys.argv[1:]
summary = json.loads(pathlib.Path(summary_path).read_text(encoding="utf-8"))
if (
    summary.get("status") != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("test_data_opened") is not False
):
    raise SystemExit("five-seed summary is not a validation-only promoted result")
arm = summary.get("promoted_arm")
if arm == "uniform":
    config = uniform
elif arm == "family_balanced":
    config = balanced
else:
    raise SystemExit("five-seed summary selected an unknown arm")
selected_checkpoint = str(summary["selected_checkpoint_path"])
if digest(selected_checkpoint) != str(summary["selected_checkpoint_sha256"]):
    raise SystemExit("selected checkpoint differs from five-seed summary")
matches = []
for identity in summary.get("finetune_reports", []):
    path = str(identity["path"])
    if digest(path) != str(identity["sha256"]):
        raise SystemExit("a five-seed finetune report hash changed")
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if str(report.get("checkpoint_path")) == selected_checkpoint:
        matches.append((path, report))
if len(matches) != 1:
    raise SystemExit("selected checkpoint does not resolve to exactly one finetune report")
report_path, report = matches[0]
expected = {
    "checkpoint_sha256": digest(selected_checkpoint),
    "config_file_sha256": digest(config),
    "overlap_validation_manifest_sha256": digest(selection_overlap),
    "clean_validation_manifest_sha256": digest(selection_validation),
}
if any(str(report.get(field)) != value for field, value in expected.items()):
    raise SystemExit("selected model report differs from paired PE inputs/configuration")
if report.get("status") != "validation_selected_real_glitch_overlap_finetune":
    raise SystemExit("selected model report has the wrong status")
print(report_path)
print(config)
PY
); then
  echo "promoted paired PE model selection failed" >&2
  exit 2
fi
readarray -t selection <<<"$selection_output"
if (( ${#selection[@]} != 2 )) || [[ -z "${selection[0]}" || -z "${selection[1]}" ]]; then
  echo "promoted paired PE selection returned an invalid result" >&2
  exit 2
fi

"$TASK_PYTHON" - \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$INDEPENDENT_PE_OVERLAP_REPORT" \
  "$INDEPENDENT_OVERLAP_AUDIT" \
  "$OVERLAP_MANIFEST" \
  "$INJECTION_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


endpoint_path, receipt_path, audit_path, overlap_path, injection_path = sys.argv[1:]
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
    endpoint.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or endpoint.get("passed") is not True
    or endpoint.get("test_rows_read") != 0
    or endpoint.get("test_evaluation") is not None
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or set(components) != expected_components
    or any(digest(item["path"]) != item["sha256"] for item in components.values())
    or pathlib.Path(endpoint["injection_arrival_manifest_path"]).resolve()
    != pathlib.Path(injection_path).resolve()
    or endpoint.get("injection_arrival_manifest_sha256") != digest(injection_path)
):
    raise SystemExit("paired PE inputs do not match the frozen independent endpoint")

receipt = json.loads(pathlib.Path(receipt_path).read_text(encoding="utf-8"))
if (
    receipt.get("status") != "verified_independent_validation_pe_overlap"
    or receipt.get("passed") is not True
    or receipt.get("scientific_claim_allowed") is not False
    or receipt.get("test_rows_read") != 0
    or receipt.get("test_evaluation") is not None
    or int(receipt.get("rows", 0)) <= 0
    or pathlib.Path(receipt["independent_validation_endpoint_report_path"]).resolve()
    != pathlib.Path(endpoint_path).resolve()
    or receipt.get("independent_validation_endpoint_report_sha256") != digest(endpoint_path)
    or pathlib.Path(receipt["injection_arrival_manifest_path"]).resolve()
    != pathlib.Path(injection_path).resolve()
    or receipt.get("injection_arrival_manifest_sha256") != digest(injection_path)
    or pathlib.Path(receipt["overlap_manifest_path"]).resolve()
    != pathlib.Path(overlap_path).resolve()
    or receipt.get("overlap_manifest_sha256") != digest(overlap_path)
    or pathlib.Path(receipt["joint_overlap_audit_path"]).resolve()
    != pathlib.Path(audit_path).resolve()
    or receipt.get("joint_overlap_audit_sha256") != digest(audit_path)
    or digest(receipt["overlap_report_path"]) != receipt["overlap_report_sha256"]
    or receipt.get("endpoint_component_reports") != components
    or any(
        digest(item["path"]) != item["sha256"]
        for item in receipt.get("endpoint_component_reports", {}).values()
    )
):
    raise SystemExit("independent validation PE overlap receipt failed hash replay")

audit = json.loads(pathlib.Path(audit_path).read_text(encoding="utf-8"))
cross = audit.get("cross_split_overlaps", {})
if (
    audit.get("status") != "passed_physical_overlap_group_audit"
    or audit.get("passed") is not True
    or set(audit.get("manifest_sha256_by_split", {})) != {"train", "val"}
    or audit["manifest_sha256_by_split"]["val"] != digest(overlap_path)
    or audit.get("rows_by_split", {}).get("val") != int(receipt["rows"])
    or not cross
    or any(values for pair in cross.values() for values in pair.values())
):
    raise SystemExit("independent validation overlap lacks a zero-leakage train/val audit")
PY

export GWYOLO_PYTHON="$TASK_PYTHON"
export GWYOLO_REPO="$TASK_CODE_DIR"
export GWYOLO_OUTPUT_ROOT="$OUTPUT_ROOT"
export GWYOLO_OVERLAP_MANIFEST="$OVERLAP_MANIFEST"
export GWYOLO_INJECTION_MANIFEST="$INJECTION_MANIFEST"
export GWYOLO_MODEL_REPORT="${selection[0]}"
export GWYOLO_MODEL_CONFIG="${selection[1]}"
export GWYOLO_MODEL_SELECTION_OVERLAP_MANIFEST="$MODEL_SELECTION_OVERLAP_MANIFEST"
export GWYOLO_MODEL_SELECTION_VALIDATION_MANIFEST="$MODEL_SELECTION_VALIDATION_MANIFEST"
export GWYOLO_INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT"
export GWYOLO_INDEPENDENT_PE_OVERLAP_REPORT="$INDEPENDENT_PE_OVERLAP_REPORT"
export GWYOLO_INDEPENDENT_OVERLAP_AUDIT="$INDEPENDENT_OVERLAP_AUDIT"
export GWYOLO_PE_SMOKE_LIMIT="${PE_SMOKE_LIMIT:-3}"
export GWYOLO_PE_SELECTION_SEED="${PE_SELECTION_SEED:-20260722}"
cd "$TASK_CODE_DIR"
bash scripts/run_paired_pe_smoke.sh
