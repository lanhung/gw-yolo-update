#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${1:-"$PROJECT_DIR/configs/legacy_remote.yaml"}
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m gwyolo.cli pipeline --config "$CONFIG_PATH"
