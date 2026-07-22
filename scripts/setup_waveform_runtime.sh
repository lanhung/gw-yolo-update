#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 BASE_PYTHON OUTPUT_DIR" >&2
  exit 2
fi

base_python=$(realpath "$1")
output_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/.." && pwd)
requirements="$code_dir/requirements-waveforms.txt"
venv_dir="$output_dir/venv"
runtime_python="$venv_dir/bin/python"
receipt="$output_dir/waveform_runtime_receipt.json"
install_log="$output_dir/install.log"

if [[ ! -x "$base_python" || ! -s "$requirements" ]]; then
  echo "waveform runtime inputs are absent" >&2
  exit 2
fi
mkdir -p "$output_dir"
if [[ ! -x "$runtime_python" ]]; then
  "$base_python" -m venv "$venv_dir"
fi
"$runtime_python" -m pip install --upgrade \
  pip==26.1.2 setuptools==80.9.0 wheel==0.47.0 >>"$install_log" 2>&1
"$runtime_python" -m pip install --requirement "$requirements" >>"$install_log" 2>&1
"$runtime_python" -c \
  'import lal, lalsimulation, pycbc; from pycbc.waveform import get_fd_waveform, get_td_waveform' \
  >>"$install_log" 2>&1

code_commit=$(git -C "$code_dir" rev-parse HEAD)
"$runtime_python" - "$requirements" "$receipt" "$code_commit" <<'PY'
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys

import lalsimulation
import pycbc

requirements = pathlib.Path(sys.argv[1]).resolve()
receipt = pathlib.Path(sys.argv[2]).resolve()
code_commit = sys.argv[3]
approximants = [
    "IMRPhenomXAS",
    "IMRPhenomXHM",
    "IMRPhenomXPHM",
    "IMRPhenomXAS_NRTidalv3",
    "IMRPhenomNSBH",
    "IMRPhenomD",
    "IMRPhenomHM",
    "IMRPhenomPv3HM",
    "IMRPhenomD_NRTidalv2",
]
resolved = {}
for name in approximants:
    approximant = int(lalsimulation.GetApproximantFromString(name))
    roundtrip = str(lalsimulation.GetStringFromApproximant(approximant))
    if roundtrip != name:
        raise RuntimeError(
            f"LALSimulation approximant did not round-trip: {name!r} -> "
            f"{approximant} -> {roundtrip!r}"
        )
    fd_implemented = bool(
        lalsimulation.SimInspiralImplementedFDApproximants(approximant)
    )
    td_implemented = bool(
        lalsimulation.SimInspiralImplementedTDApproximants(approximant)
    )
    if not (fd_implemented or td_implemented):
        raise RuntimeError(f"LALSimulation cannot generate approximant {name!r}")
    resolved[name] = {
        "enum": approximant,
        "roundtrip": roundtrip,
        "fd_implemented": fd_implemented,
        "td_implemented": td_implemented,
    }
freeze = subprocess.check_output(
    [sys.executable, "-m", "pip", "freeze", "--all"], text=True
)
payload = {
    "status": "verified_isolated_waveform_runtime",
    "passed": True,
    "code_commit": code_commit,
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "requirements_path": str(requirements),
    "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    "pycbc_version": pycbc.__version__,
    "lalsuite_version": importlib.metadata.version("lalsuite"),
    "approximants": resolved,
    "pip_freeze": freeze.splitlines(),
    "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
}
temporary = receipt.with_suffix(receipt.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(receipt)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf '%s\n' "$runtime_python"
