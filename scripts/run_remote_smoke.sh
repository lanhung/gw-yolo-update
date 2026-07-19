#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/gwyolo/bin/python}
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m pytest "$PROJECT_DIR/tests" -q
"$PYTHON_BIN" -m gwyolo.cli pipeline --config "$PROJECT_DIR/configs/legacy_smoke.yaml"
