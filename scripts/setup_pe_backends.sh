#!/usr/bin/env bash
set -euo pipefail

required=(
  PE_BASE_PYTHON
  DINGO_SOURCE_DIR DINGO_EXPECTED_COMMIT DINGO_EXPECTED_TAG DINGO_VENV
  AMPLFI_SOURCE_DIR AMPLFI_EXPECTED_COMMIT AMPLFI_EXPECTED_TAG AMPLFI_VENV
  PE_ENVIRONMENT_REPORT_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: ${variable}" >&2
    exit 2
  fi
done

if [[ "$DINGO_VENV" == "$AMPLFI_VENV" ]]; then
  echo "DINGO_VENV and AMPLFI_VENV must be different" >&2
  exit 2
fi

PE_INSTALL_LOCK=${PE_INSTALL_LOCK:-/tmp/gwyolo-pe-backend-install.lock}
mkdir -p "$(dirname "$PE_INSTALL_LOCK")"
exec {pe_install_lock_fd}>"$PE_INSTALL_LOCK"
if ! flock -n "$pe_install_lock_fd"; then
  echo "another PE backend bootstrap owns the atomic installation lock" >&2
  exit 3
fi

if ps -eo stat=,cmd= | awk '$1 !~ /^T/ && /[p]ip install|[c]onda install|[u]v pip/ {found=1} END {exit !found}'; then
  echo "an active package installation already exists; refusing concurrent mutation" >&2
  exit 3
fi

verify_source() {
  local source=$1
  local expected_commit=$2
  local expected_tag=$3
  local observed_commit observed_tag
  observed_commit=$(git -C "$source" rev-parse HEAD)
  observed_tag=$(git -C "$source" describe --tags --exact-match HEAD)
  if [[ "$observed_commit" != "$expected_commit" || "$observed_tag" != "$expected_tag" ]]; then
    echo "source lock mismatch: $source $observed_commit $observed_tag" >&2
    exit 4
  fi
  if [[ -n "$(git -C "$source" status --porcelain)" ]]; then
    echo "source repository is dirty: $source" >&2
    exit 4
  fi
}

verify_source "$DINGO_SOURCE_DIR" "$DINGO_EXPECTED_COMMIT" "$DINGO_EXPECTED_TAG"
verify_source "$AMPLFI_SOURCE_DIR" "$AMPLFI_EXPECTED_COMMIT" "$AMPLFI_EXPECTED_TAG"

venv_flags=()
if [[ "${PE_USE_SYSTEM_SITE_PACKAGES:-0}" == "1" ]]; then
  venv_flags+=(--system-site-packages)
fi

install_backend() {
  local name=$1
  local source=$2
  local venv=$3
  local report_dir=$4
  if [[ ! -x "$venv/bin/python" ]]; then
    "$PE_BASE_PYTHON" -m venv "${venv_flags[@]}" "$venv"
  fi
  "$venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "$venv/bin/python" -m pip install "$source"
  mkdir -p "$report_dir"
  local temporary
  temporary=$(mktemp "$report_dir/.${name}.pip-freeze.XXXXXX")
  "$venv/bin/python" -m pip freeze --all | LC_ALL=C sort > "$temporary"
  mv "$temporary" "$report_dir/${name}.pip-freeze.txt"
  "$venv/bin/python" - <<'PY' > "$report_dir/${name}.runtime.json"
import importlib.metadata as metadata
import json
import platform
import sys

import torch

print(json.dumps({
    "python": platform.python_version(),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "torch": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "distributions": sorted(
        (item.metadata["Name"].lower(), item.version)
        for item in metadata.distributions()
        if item.metadata.get("Name")
    ),
}, indent=2, sort_keys=True))
PY
}

install_backend dingo "$DINGO_SOURCE_DIR" "$DINGO_VENV" "$PE_ENVIRONMENT_REPORT_DIR"
install_backend amplfi "$AMPLFI_SOURCE_DIR" "$AMPLFI_VENV" "$PE_ENVIRONMENT_REPORT_DIR"
