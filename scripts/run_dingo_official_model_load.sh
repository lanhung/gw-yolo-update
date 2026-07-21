#!/usr/bin/env bash
set -euo pipefail

# Load both official DINGO posterior and GNPE time-initialization models only
# after replaying the immutable source-acquisition report. This is a backend
# compatibility gate, not a posterior or parameter-estimation result.

required=(
  TASK_PYTHON
  DINGO_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  MODEL_SOURCE_CONFIG
  MODEL_ACQUISITION_REPORT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

EXPECTED_DINGO_VERSION=${EXPECTED_DINGO_VERSION:-0.9.8}
DEVICE=${DEVICE:-cuda}
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout: $TASK_CODE_DIR" >&2
  exit 2
fi
for path in \
  "$TASK_PYTHON" \
  "$DINGO_PYTHON" \
  "$MODEL_SOURCE_CONFIG" \
  "$MODEL_ACQUISITION_REPORT" \
  "$TASK_CODE_DIR/scripts/run_pe_model_load_smoke.py"; do
  if [[ ! -s "$path" ]]; then
    echo "required DINGO model-load input is absent: $path" >&2
    exit 2
  fi
done

if ! model_output=$(
  "$TASK_PYTHON" - \
    "$MODEL_SOURCE_CONFIG" \
    "$MODEL_ACQUISITION_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys

import yaml


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


config_path, report_path = sys.argv[1:]
config = yaml.safe_load(pathlib.Path(config_path).read_text(encoding="utf-8"))
report = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
sources = config.get("sources", [])
if config.get("schema_version") != 1 or not sources:
    raise SystemExit("official model source configuration is invalid")
expected_roles = {
    "model_manifest",
    "training_settings",
    "posterior_model",
    "time_initialization_model",
}
by_role = {str(row.get("role")): row for row in sources}
files = {str(row.get("role")): row for row in report.get("files", [])}
recorded_config = pathlib.Path(report.get("config_path", ""))
if (
    set(by_role) != expected_roles
    or set(files) != expected_roles
    or report.get("status") != "verified"
    or report.get("download_enabled") is not True
    or not recorded_config.is_file()
    or digest(recorded_config) != report.get("config_sha256")
    or report.get("config_sha256") != digest(config_path)
):
    raise SystemExit("DINGO acquisition report does not bind the official source configuration")
for role in sorted(expected_roles):
    expected = by_role[role]
    observed = files[role]
    path = pathlib.Path(observed.get("path", ""))
    checksum = expected["checksum"]
    if (
        expected.get("backend") != "DINGO"
        or observed.get("backend") != "DINGO"
        or observed.get("valid") is not True
        or not path.is_file()
        or path.name != expected.get("filename")
        or int(observed.get("size_bytes", -1)) != int(expected.get("size_bytes", -2))
        or int(observed.get("expected_size_bytes", -1)) != int(expected.get("size_bytes", -2))
        or observed.get("checksum_algorithm") != checksum.get("algorithm")
        or observed.get("expected_checksum") != checksum.get("value")
        or digest(path, checksum["algorithm"]) != checksum["value"]
        or digest(path) != observed.get("sha256")
    ):
        raise SystemExit(f"official DINGO source failed integrity replay: {role}")

posterior = files["posterior_model"]
initialization = files["time_initialization_model"]
print(pathlib.Path(posterior["path"]).resolve())
print(posterior["sha256"])
print(pathlib.Path(initialization["path"]).resolve())
print(initialization["sha256"])
PY
); then
  echo "official DINGO source acquisition replay failed" >&2
  exit 2
fi
readarray -t model_identity <<<"$model_output"
if (( ${#model_identity[@]} != 4 )); then
  echo "official DINGO source replay returned an invalid model identity" >&2
  exit 2
fi
posterior_model=${model_identity[0]}
posterior_sha256=${model_identity[1]}
initialization_model=${model_identity[2]}
initialization_sha256=${model_identity[3]}

observed_version=$(
  "$DINGO_PYTHON" -c \
    'import importlib.metadata; print(importlib.metadata.version("dingo-gw"))'
)
if [[ "$observed_version" != "$EXPECTED_DINGO_VERSION" ]]; then
  echo "DINGO runtime version mismatch: $observed_version != $EXPECTED_DINGO_VERSION" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
load_report="$OUTPUT_ROOT/backend_model_load_report.json"
failure_receipt="$OUTPUT_ROOT/official_dingo_model_load_failure.json"
attempt_log="$OUTPUT_ROOT/backend_model_load_attempt.log"
if [[ -s "$failure_receipt" && ! -s "$load_report" ]]; then
  echo "an immutable DINGO model-load failure receipt already exists: $failure_receipt" >&2
  exit 1
fi
if [[ ! -s "$load_report" ]]; then
  if [[ "$DEVICE" == cuda* ]]; then
    while :; do
      gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | sed '/^[[:space:]]*$/d' || true)
      [[ -z "$gpu_pids" ]] && break
      sleep 30
    done
  fi
  if [[ -e "$attempt_log" || -e "$attempt_log.part" ]]; then
    echo "DINGO model-load attempt log already exists: $attempt_log" >&2
    exit 2
  fi
  if (
    cd "$TASK_CODE_DIR"
    "$DINGO_PYTHON" scripts/run_pe_model_load_smoke.py \
      --backend DINGO \
      --model "$posterior_model" \
      --expected-model-sha256 "$posterior_sha256" \
      --model-init "$initialization_model" \
      --expected-model-init-sha256 "$initialization_sha256" \
      --output "$load_report" \
      --device "$DEVICE"
  ) >"$attempt_log.part" 2>&1; then
    load_status=0
  else
    load_status=$?
  fi
  mv "$attempt_log.part" "$attempt_log"
  if (( load_status != 0 )); then
    "$TASK_PYTHON" - \
      "$MODEL_SOURCE_CONFIG" \
      "$MODEL_ACQUISITION_REPORT" \
      "$posterior_model" \
      "$posterior_sha256" \
      "$initialization_model" \
      "$initialization_sha256" \
      "$EXPECTED_DINGO_VERSION" \
      "$DEVICE" \
      "$DINGO_PYTHON" \
      "$GWYOLO_CODE_COMMIT" \
      "$attempt_log" \
      "$load_status" \
      "$failure_receipt" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


(
    config_path,
    acquisition_path,
    posterior_path,
    posterior_sha,
    initialization_path,
    initialization_sha,
    expected_version,
    device,
    dingo_python,
    code_commit,
    attempt_log_path,
    exit_code,
    receipt_path,
) = sys.argv[1:]
result = {
    "status": "official_dingo_dual_model_load_failed",
    "passed": False,
    "scientific_claim_allowed": False,
    "scientific_blocker": "official DINGO models failed the pinned runtime model-load gate",
    "fallback_allowed": False,
    "fallback_constraint": (
        "the attempt log must first be adjudicated as a version-compatibility failure; only "
        "then may the predeclared native DINGO 0.5.8 environment be tried; model substitution "
        "is forbidden"
    ),
    "failure_scope": "runtime_model_load",
    "test_rows_read": 0,
    "test_evaluation": None,
    "backend": "DINGO",
    "backend_version": expected_version,
    "device": device,
    "dingo_python": str(pathlib.Path(dingo_python).resolve()),
    "model_source_config_path": str(pathlib.Path(config_path).resolve()),
    "model_source_config_sha256": digest(config_path),
    "model_acquisition_report_path": str(pathlib.Path(acquisition_path).resolve()),
    "model_acquisition_report_sha256": digest(acquisition_path),
    "posterior_model_path": str(pathlib.Path(posterior_path).resolve()),
    "posterior_model_sha256": posterior_sha,
    "initialization_model_path": str(pathlib.Path(initialization_path).resolve()),
    "initialization_model_sha256": initialization_sha,
    "attempt_log_path": str(pathlib.Path(attempt_log_path).resolve()),
    "attempt_log_sha256": digest(attempt_log_path),
    "exit_code": int(exit_code),
    "code_commit": code_commit,
}
target = pathlib.Path(receipt_path)
if target.exists():
    raise SystemExit("official DINGO model-load failure receipt already exists")
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
    echo "official DINGO dual-model load failed; see $failure_receipt" >&2
    exit "$load_status"
  fi
fi

receipt="$OUTPUT_ROOT/official_dingo_model_load_receipt.json"
"$TASK_PYTHON" - \
  "$MODEL_SOURCE_CONFIG" \
  "$MODEL_ACQUISITION_REPORT" \
  "$load_report" \
  "$posterior_model" \
  "$posterior_sha256" \
  "$initialization_model" \
  "$initialization_sha256" \
  "$EXPECTED_DINGO_VERSION" \
  "$DEVICE" \
  "$GWYOLO_CODE_COMMIT" \
  "$receipt" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


(
    config_path,
    acquisition_path,
    load_path,
    posterior_path,
    posterior_sha,
    initialization_path,
    initialization_sha,
    expected_version,
    device,
    code_commit,
    receipt_path,
) = sys.argv[1:]
load = json.loads(pathlib.Path(load_path).read_text(encoding="utf-8"))
artifacts = load.get("artifacts", {})
if (
    load.get("status") != "real_pe_backend_model_load_smoke_complete"
    or load.get("scientific_claim_allowed") is not False
    or load.get("backend") != "DINGO"
    or load.get("backend_version") != expected_version
    or load.get("device") != device
    or pathlib.Path(artifacts.get("model", {}).get("path", "")).resolve()
    != pathlib.Path(posterior_path).resolve()
    or artifacts.get("model", {}).get("sha256") != posterior_sha
    or pathlib.Path(artifacts.get("model_init", {}).get("path", "")).resolve()
    != pathlib.Path(initialization_path).resolve()
    or artifacts.get("model_init", {}).get("sha256") != initialization_sha
    or digest(posterior_path) != posterior_sha
    or digest(initialization_path) != initialization_sha
    or int(load.get("observations", {}).get("model_parameter_count", 0)) <= 0
    or int(load.get("observations", {}).get("initialization_model_parameter_count", 0)) <= 0
):
    raise SystemExit("official DINGO posterior/time model load did not pass")
result = {
    "status": "verified_official_dingo_dual_model_load",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": "model compatibility only; common-prior paired posterior validation remains required",
    "test_rows_read": 0,
    "test_evaluation": None,
    "backend": "DINGO",
    "backend_version": expected_version,
    "device": device,
    "model_source_config_path": str(pathlib.Path(config_path).resolve()),
    "model_source_config_sha256": digest(config_path),
    "model_acquisition_report_path": str(pathlib.Path(acquisition_path).resolve()),
    "model_acquisition_report_sha256": digest(acquisition_path),
    "backend_model_load_report_path": str(pathlib.Path(load_path).resolve()),
    "backend_model_load_report_sha256": digest(load_path),
    "posterior_model_path": str(pathlib.Path(posterior_path).resolve()),
    "posterior_model_sha256": posterior_sha,
    "initialization_model_path": str(pathlib.Path(initialization_path).resolve()),
    "initialization_model_sha256": initialization_sha,
    "observations": load["observations"],
    "environment": load["environment"],
    "code_commit": code_commit,
}
target = pathlib.Path(receipt_path)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing official DINGO model-load receipt has a different identity")
else:
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
