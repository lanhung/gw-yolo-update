#!/usr/bin/env bash
set -euo pipefail

# Install the exact DINGO version that trained the official O4a model while
# reusing the already provisioned dependency environment read-only. The DINGO
# package itself lives in a separate venv and shadows the dependency runtime.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  BASE_RUNTIME_PYTHON
  NATIVE_SOURCE_DIR
  NATIVE_VENV
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

EXPECTED_NATIVE_COMMIT=${EXPECTED_NATIVE_COMMIT:-04c8ab3ec694410ad85466ea6bfdc6aa2274ac14}
EXPECTED_NATIVE_TAG=${EXPECTED_NATIVE_TAG:-v0.5.8}
EXPECTED_NATIVE_VERSION=${EXPECTED_NATIVE_VERSION:-0.5.8}
EXPECTED_BASE_VERSION=${EXPECTED_BASE_VERSION:-0.9.8}
SETUPTOOLS_SCM_VERSION=${SETUPTOOLS_SCM_VERSION:-8.1.0}
INSTALL_LOCK=${INSTALL_LOCK:-/tmp/gwyolo-dingo-native-overlay-install.lock}
CONFIG_PATH=${CONFIG_PATH:-$TASK_CODE_DIR/configs/dingo_official_native_runtime.yaml}

for path in "$TASK_PYTHON" "$BASE_RUNTIME_PYTHON" "$CONFIG_PATH"; do
  if [[ ! -s "$path" ]]; then
    echo "required native-runtime setup input is absent: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" || ! -d "$NATIVE_SOURCE_DIR/.git" ]]; then
  echo "code or native DINGO source checkout is absent" >&2
  exit 2
fi
observed_code_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)
observed_source_commit=$(git -C "$NATIVE_SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)
observed_source_tag=$(git -C "$NATIVE_SOURCE_DIR" describe --tags --exact-match HEAD 2>/dev/null || true)
if [[ "$observed_code_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi
if [[ "$observed_source_commit" != "$EXPECTED_NATIVE_COMMIT" \
  || "$observed_source_tag" != "$EXPECTED_NATIVE_TAG" \
  || -n "$(git -C "$NATIVE_SOURCE_DIR" status --porcelain)" ]]; then
  echo "native DINGO source lock mismatch" >&2
  exit 3
fi

mkdir -p "$(dirname "$INSTALL_LOCK")"
exec {install_lock_fd}>"$INSTALL_LOCK"
if ! flock -n "$install_lock_fd"; then
  echo "another DINGO native-overlay installation owns the lock" >&2
  exit 4
fi
if ps -eo stat=,cmd= | awk '$1 !~ /^T/ && /[p]ip install|[c]onda install|[u]v pip/ {found=1} END {exit !found}'; then
  echo "an active package installation already exists" >&2
  exit 4
fi

mkdir -p "$OUTPUT_ROOT"
receipt="$OUTPUT_ROOT/dingo_native_overlay_receipt.json"
base_freeze="$OUTPUT_ROOT/base-runtime-pip-freeze.txt"
native_freeze="$OUTPUT_ROOT/native-overlay-pip-freeze.txt"

base_identity=$(
  "$BASE_RUNTIME_PYTHON" - "$EXPECTED_BASE_VERSION" <<'PY'
import importlib.metadata
import json
import pathlib
import site
import sys

expected = sys.argv[1]
observed = importlib.metadata.version("dingo-gw")
if observed != expected:
    raise SystemExit(f"base DINGO runtime mismatch: {observed} != {expected}")
site_packages = [str(pathlib.Path(path).resolve()) for path in site.getsitepackages()]
if not site_packages or any(not pathlib.Path(path).is_dir() for path in site_packages):
    raise SystemExit("cannot resolve base-runtime site-packages directories")
print(json.dumps({"version": observed, "site_packages": site_packages}))
PY
)
base_sites_output=$(
  "$TASK_PYTHON" -c \
    'import json,sys; print("\n".join(json.loads(sys.argv[1])["site_packages"]))' \
    "$base_identity"
)
readarray -t base_sites <<<"$base_sites_output"
if (( ${#base_sites[@]} == 0 )); then
  echo "base DINGO runtime exposes no dependency directories" >&2
  exit 5
fi
"$BASE_RUNTIME_PYTHON" -m pip freeze --all | LC_ALL=C sort >"$base_freeze.part"
mv "$base_freeze.part" "$base_freeze"

if [[ ! -x "$NATIVE_VENV/bin/python" ]]; then
  partial="${NATIVE_VENV}.part"
  if [[ ! -x "$partial/bin/python" ]]; then
    if [[ -e "$partial" ]]; then
      echo "incomplete native overlay exists without an executable Python: $partial" >&2
      exit 5
    fi
    "$BASE_RUNTIME_PYTHON" -m venv --without-pip "$partial"
  fi
  partial_site=$(
    "$partial/bin/python" -c \
      'import site; print(next(path for path in site.getsitepackages() if path.startswith(__import__("sys").prefix)))'
  )
  "$TASK_PYTHON" - "$partial_site/gwyolo-dingo-dependency-base.pth" \
    "${base_sites[@]}" <<'PY'
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".part")
paths = [str(pathlib.Path(path).resolve()) for path in sys.argv[2:]]
temporary.write_text("\n".join(paths) + "\n", encoding="utf-8")
temporary.replace(target)
PY
  "$partial/bin/python" -m pip install \
    --ignore-installed --no-deps --disable-pip-version-check \
    "setuptools-scm==$SETUPTOOLS_SCM_VERSION"
  SETUPTOOLS_SCM_PRETEND_VERSION="$EXPECTED_NATIVE_VERSION" \
  "$partial/bin/python" -m pip install \
    --ignore-installed --no-deps --no-build-isolation --disable-pip-version-check \
    "$NATIVE_SOURCE_DIR"
  "$partial/bin/python" - "$EXPECTED_NATIVE_VERSION" "$partial" <<'PY'
import importlib.metadata
import pathlib
import sys

import dingo
import torch

expected, prefix = sys.argv[1:]
if importlib.metadata.version("dingo-gw") != expected:
    raise SystemExit("native DINGO distribution version mismatch after installation")
if pathlib.Path(prefix).resolve() not in pathlib.Path(dingo.__file__).resolve().parents:
    raise SystemExit("native DINGO import did not resolve inside the overlay")
if not torch.cuda.is_available():
    raise SystemExit("native DINGO overlay cannot see CUDA through the dependency base")
PY
  if [[ -e "$NATIVE_VENV" ]]; then
    echo "native overlay target appeared during installation" >&2
    exit 5
  fi
  mv "$partial" "$NATIVE_VENV"
fi

"$NATIVE_VENV/bin/python" -m pip freeze --all | LC_ALL=C sort >"$native_freeze.part"
mv "$native_freeze.part" "$native_freeze"
"$TASK_PYTHON" - \
  "$CONFIG_PATH" "$base_freeze" "$native_freeze" "$NATIVE_SOURCE_DIR" \
  "$NATIVE_VENV" "$EXPECTED_NATIVE_COMMIT" "$EXPECTED_NATIVE_TAG" \
  "$EXPECTED_NATIVE_VERSION" "$EXPECTED_BASE_VERSION" "$GWYOLO_CODE_COMMIT" \
  "$SETUPTOOLS_SCM_VERSION" "$receipt" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


(
    config_path,
    base_freeze,
    native_freeze,
    source_dir,
    venv,
    expected_commit,
    expected_tag,
    expected_version,
    base_version,
    code_commit,
    setuptools_scm_version,
    receipt_path,
) = sys.argv[1:]
python = pathlib.Path(venv) / "bin/python"
probe = subprocess.run(
    [
        str(python),
        "-c",
        (
            "import dingo,importlib.metadata,json,pathlib,platform,sys,torch;"
            "print(json.dumps({'version':importlib.metadata.version('dingo-gw'),"
            "'module':str(pathlib.Path(dingo.__file__).resolve()),"
            "'prefix':str(pathlib.Path(sys.prefix).resolve()),'python':platform.python_version(),"
            "'torch':torch.__version__,'cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
        ),
    ],
    check=True,
    capture_output=True,
    text=True,
)
runtime = json.loads(probe.stdout)
if (
    runtime["version"] != expected_version
    or runtime["prefix"] != str(pathlib.Path(venv).resolve())
    or pathlib.Path(venv).resolve() not in pathlib.Path(runtime["module"]).parents
    or runtime["cuda_available"] is not True
):
    raise SystemExit("promoted native DINGO overlay failed runtime replay")
result = {
    "status": "verified_dingo_official_native_runtime_overlay",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": "runtime installation alone is not model-load or posterior evidence",
    "test_rows_read": 0,
    "test_evaluation": None,
    "backend": "DINGO",
    "backend_version": expected_version,
    "dependency_base_backend_version": base_version,
    "setuptools_scm_version": setuptools_scm_version,
    "source_path": str(pathlib.Path(source_dir).resolve()),
    "source_commit": expected_commit,
    "source_tag": expected_tag,
    "venv_path": str(pathlib.Path(venv).resolve()),
    "python_executable": str(python.resolve()),
    "runtime": runtime,
    "artifacts": {
        "config": {"path": str(pathlib.Path(config_path).resolve()), "sha256": digest(config_path)},
        "base_runtime_pip_freeze": {"path": str(pathlib.Path(base_freeze).resolve()), "sha256": digest(base_freeze)},
        "native_overlay_pip_freeze": {"path": str(pathlib.Path(native_freeze).resolve()), "sha256": digest(native_freeze)},
    },
    "code_commit": code_commit,
}
target = pathlib.Path(receipt_path)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing native DINGO overlay receipt has a different identity")
else:
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
