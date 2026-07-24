#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SUITE_CONFIG ACCESS_LOG OUTPUT_DIR" >&2
  exit 2
fi

suite_config=$(realpath "$1")
access_log=$(realpath -m "$2")
output_dir=$(realpath -m "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/.." && pwd)
task_python=${TASK_PYTHON:-python}

if [[ ! -s "$suite_config" ]]; then
  echo "locked suite config is absent: $suite_config" >&2
  exit 2
fi
if [[ -e "$access_log" ]]; then
  echo "locked access log already exists: $access_log" >&2
  exit 3
fi

code_commit=$(git -C "$code_dir" rev-parse HEAD)
(
  cd "$code_dir"
  export PYTHONPATH=src
  export GWYOLO_CODE_COMMIT="$code_commit"
  "$task_python" -m gwyolo.cli gwtc5-locked-availability-plan \
    --suite-config "$suite_config" \
    --access-log "$access_log" \
    --output-dir "$output_dir" \
    --sample-rate-khz 4
)

"$task_python" - "$output_dir" "$suite_config" "$access_log" "$code_commit" <<'PY'
import hashlib
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
suite_config = pathlib.Path(sys.argv[2])
access_log = pathlib.Path(sys.argv[3])
code_commit = sys.argv[4]
report_path = output_dir / "gwtc5_o4b_availability_report.json"
manifest_path = output_dir / "gwtc5_o4b_availability.jsonl"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    report.get("status") != "score_blind_gwtc5_o4b_availability_inventory"
    or report.get("passed") is not True
    or report.get("candidate_catalog_queried") is not False
    or report.get("candidate_scores_inspected") is not False
    or report.get("event_level_parameters_inspected") is not False
    or int(report.get("test_strain_files_downloaded", -1)) != 0
    or int(report.get("test_strain_bytes_read", -1)) != 0
    or int(report.get("test_strain_rows_read", -1)) != 0
    or int(report.get("availability_blocks", 0)) <= 0
    or int(report.get("unique_gps_blocks", 0)) != int(report.get("availability_blocks", -1))
    or report.get("required_detector_subsets_covered") is not True
    or report.get("manifest_sha256") != digest(manifest_path)
    or report.get("suite_config_sha256") != digest(suite_config)
    or report.get("code_commit") != code_commit
    or access_log.exists()
):
    raise SystemExit("GWTC-5 score-blind availability replay failed")
PY
