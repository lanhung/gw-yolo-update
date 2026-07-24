#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: resolve_promoted_overlap_model.sh SUMMARY UNIFORM FAMILY ADAPTER" >&2
  exit 2
fi
if [[ -z "${TASK_PYTHON:-}" ]]; then
  echo "TASK_PYTHON is required" >&2
  exit 2
fi

"$TASK_PYTHON" - "$@" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summary_path = pathlib.Path(sys.argv[1]).resolve()
configs = {
    "uniform": pathlib.Path(sys.argv[2]).resolve(),
    "family_balanced": pathlib.Path(sys.argv[3]).resolve(),
    "glitch_adapter": pathlib.Path(sys.argv[4]).resolve(),
}
if not summary_path.is_file():
    raise SystemExit("five-seed summary is absent")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
arm = summary.get("promoted_arm")
if (
    summary.get("status")
    != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("five_seed_stability", {}).get("status")
    != "five_seed_reproducibility_gate_v1"
    or summary.get("five_seed_stability", {}).get("passed") is not True
    or summary.get("test_data_opened") is not False
    or arm not in configs
):
    raise SystemExit("five-seed summary is not a supported promoted validation model")
checkpoint = pathlib.Path(str(summary.get("selected_checkpoint_path", ""))).resolve()
config = configs[arm]
if (
    not checkpoint.is_file()
    or digest(checkpoint) != summary.get("selected_checkpoint_sha256")
    or not config.is_file()
    or digest(config)
    != summary.get("common_artifact_hashes", {}).get("config_file_sha256")
):
    raise SystemExit("promoted checkpoint/config failed exact hash replay")
print(arm)
print(checkpoint)
print(config)
PY
