#!/usr/bin/env bash
set -euo pipefail

# Retry the identical official model bytes under their predeclared native
# DINGO runtime only after a hash-bound compatibility adjudication authorizes it.

required=(
  TASK_PYTHON
  DINGO_NATIVE_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  MODEL_SOURCE_CONFIG
  MODEL_ACQUISITION_REPORT
  PRIMARY_FAILURE_RECEIPT
  COMPATIBILITY_ADJUDICATION_REPORT
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

EXPECTED_NATIVE_DINGO_VERSION=${EXPECTED_NATIVE_DINGO_VERSION:-0.5.8}
DEVICE=${DEVICE:-cuda}
for path in \
  "$TASK_PYTHON" \
  "$DINGO_NATIVE_PYTHON" \
  "$MODEL_SOURCE_CONFIG" \
  "$MODEL_ACQUISITION_REPORT" \
  "$PRIMARY_FAILURE_RECEIPT" \
  "$COMPATIBILITY_ADJUDICATION_REPORT" \
  "$TASK_CODE_DIR/scripts/run_dingo_official_model_load.sh"; do
  if [[ ! -s "$path" ]]; then
    echo "required native-fallback input is absent: $path" >&2
    exit 2
  fi
done

"$TASK_PYTHON" - \
  "$MODEL_SOURCE_CONFIG" \
  "$MODEL_ACQUISITION_REPORT" \
  "$PRIMARY_FAILURE_RECEIPT" \
  "$COMPATIBILITY_ADJUDICATION_REPORT" \
  "$EXPECTED_NATIVE_DINGO_VERSION" <<'PY'
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


config_path, acquisition_path, failure_path, adjudication_path, expected_version = sys.argv[1:]
failure = json.loads(pathlib.Path(failure_path).read_text(encoding="utf-8"))
adjudication = json.loads(pathlib.Path(adjudication_path).read_text(encoding="utf-8"))
if (
    adjudication.get("status") != "dingo_native_runtime_fallback_authorized"
    or adjudication.get("passed") is not True
    or adjudication.get("fallback_allowed") is not True
    or adjudication.get("model_substitution_allowed") is not False
    or adjudication.get("scientific_claim_allowed") is not False
    or adjudication.get("test_rows_read") != 0
    or adjudication.get("primary_runtime_version") != failure.get("backend_version")
    or adjudication.get("authorized_fallback_runtime_version") != expected_version
    or adjudication.get("failure_receipt_sha256") != digest(failure_path)
    or pathlib.Path(adjudication.get("failure_receipt_path", "")).resolve()
    != pathlib.Path(failure_path).resolve()
    or adjudication.get("model_acquisition_report_sha256") != digest(acquisition_path)
    or pathlib.Path(adjudication.get("model_acquisition_report_path", "")).resolve()
    != pathlib.Path(acquisition_path).resolve()
    or adjudication.get("model_source_config_sha256") != digest(config_path)
    or pathlib.Path(adjudication.get("model_source_config_path", "")).resolve()
    != pathlib.Path(config_path).resolve()
    or adjudication.get("posterior_model_sha256") != failure.get("posterior_model_sha256")
    or adjudication.get("initialization_model_sha256")
    != failure.get("initialization_model_sha256")
):
    raise SystemExit("native DINGO fallback is not authorized for these exact artifacts")
PY

observed_version=$(
  "$DINGO_NATIVE_PYTHON" -c \
    'import importlib.metadata; print(importlib.metadata.version("dingo-gw"))'
)
if [[ "$observed_version" != "$EXPECTED_NATIVE_DINGO_VERSION" ]]; then
  echo "native DINGO runtime version mismatch: $observed_version != $EXPECTED_NATIVE_DINGO_VERSION" >&2
  exit 2
fi

env \
  TASK_PYTHON="$TASK_PYTHON" \
  DINGO_PYTHON="$DINGO_NATIVE_PYTHON" \
  TASK_CODE_DIR="$TASK_CODE_DIR" \
  GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
  MODEL_SOURCE_CONFIG="$MODEL_SOURCE_CONFIG" \
  MODEL_ACQUISITION_REPORT="$MODEL_ACQUISITION_REPORT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  EXPECTED_DINGO_VERSION="$EXPECTED_NATIVE_DINGO_VERSION" \
  DEVICE="$DEVICE" \
  bash "$TASK_CODE_DIR/scripts/run_dingo_official_model_load.sh"

inner_receipt="$OUTPUT_ROOT/official_dingo_model_load_receipt.json"
fallback_receipt="$OUTPUT_ROOT/official_dingo_native_fallback_receipt.json"
"$TASK_PYTHON" - \
  "$PRIMARY_FAILURE_RECEIPT" \
  "$COMPATIBILITY_ADJUDICATION_REPORT" \
  "$inner_receipt" \
  "$EXPECTED_NATIVE_DINGO_VERSION" \
  "$GWYOLO_CODE_COMMIT" \
  "$fallback_receipt" <<'PY'
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


failure_path, adjudication_path, inner_path, version, code_commit, target_path = sys.argv[1:]
adjudication = json.loads(pathlib.Path(adjudication_path).read_text(encoding="utf-8"))
inner = json.loads(pathlib.Path(inner_path).read_text(encoding="utf-8"))
if (
    adjudication.get("status") != "dingo_native_runtime_fallback_authorized"
    or inner.get("status") != "verified_official_dingo_dual_model_load"
    or inner.get("passed") is not True
    or inner.get("backend_version") != version
    or inner.get("scientific_claim_allowed") is not False
    or inner.get("test_rows_read") != 0
):
    raise SystemExit("native DINGO dual-model load did not pass")
result = {
    "status": "verified_official_dingo_native_runtime_dual_model_load",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": "native runtime model loading is not posterior or calibration evidence",
    "test_rows_read": 0,
    "test_evaluation": None,
    "backend": "DINGO",
    "backend_version": version,
    "model_substitution_allowed": False,
    "primary_failure_receipt_path": str(pathlib.Path(failure_path).resolve()),
    "primary_failure_receipt_sha256": digest(failure_path),
    "compatibility_adjudication_report_path": str(pathlib.Path(adjudication_path).resolve()),
    "compatibility_adjudication_report_sha256": digest(adjudication_path),
    "native_model_load_receipt_path": str(pathlib.Path(inner_path).resolve()),
    "native_model_load_receipt_sha256": digest(inner_path),
    "posterior_model_sha256": inner["posterior_model_sha256"],
    "initialization_model_sha256": inner["initialization_model_sha256"],
    "observations": inner["observations"],
    "environment": inner["environment"],
    "code_commit": code_commit,
}
target = pathlib.Path(target_path)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing native DINGO fallback receipt has a different identity")
else:
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
