#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 AVAILABILITY_DIR SUITE_CONFIG POPULATION_CONFIG ACCESS_LOG OUTPUT_DIR" >&2
  exit 2
fi

availability_dir=$(realpath "$1")
suite_config=$(realpath "$2")
population_config=$(realpath "$3")
access_log=$(realpath -m "$4")
output_dir=$(realpath -m "$5")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/.." && pwd)
task_python=${TASK_PYTHON:-python}
waveform_python=${WAVEFORM_PYTHON:-$task_python}
waveform_runtime_receipt=${WAVEFORM_RUNTIME_RECEIPT:-}
availability_manifest="$availability_dir/gwtc5_o4b_availability.jsonl"
availability_report="$availability_dir/gwtc5_o4b_availability_report.json"
plan_dir="$output_dir/injection-plan"
inventory_manifest="$plan_dir/gwtc5_locked_injection_inventory.jsonl"
inventory_report="$plan_dir/gwtc5_locked_injection_inventory_report.json"
waveform_report="$output_dir/waveform_validation_report.json"
freeze_report="$output_dir/gwtc5_locked_corpus_unopened.json"
ledger_report="$output_dir/locked_corpus_gate_replay.json"

for required_path in \
  "$availability_manifest" \
  "$availability_report" \
  "$suite_config" \
  "$population_config"; do
  if [[ ! -s "$required_path" ]]; then
    echo "required locked-planning input is absent: $required_path" >&2
    exit 2
  fi
done
if [[ -e "$access_log" ]]; then
  echo "locked access log already exists: $access_log" >&2
  exit 3
fi
if [[ -z "$waveform_runtime_receipt" || ! -s "$waveform_runtime_receipt" ]]; then
  echo "WAVEFORM_RUNTIME_RECEIPT must name a verified isolated runtime receipt" >&2
  exit 2
fi
waveform_runtime_receipt=$(realpath "$waveform_runtime_receipt")

code_commit=$(git -C "$code_dir" rev-parse HEAD)
(
  cd "$code_dir"
  export PYTHONPATH=src
  export GWYOLO_CODE_COMMIT="$code_commit"
  "$task_python" -m gwyolo.cli gwtc5-locked-injection-plan \
    --availability-manifest "$availability_manifest" \
    --availability-report "$availability_report" \
    --suite-config "$suite_config" \
    --population-config "$population_config" \
    --access-log "$access_log" \
    --output-dir "$plan_dir"
  if [[ ! -s "$waveform_report" ]]; then
    "$waveform_python" -m gwyolo.cli waveform-validate \
      --recipes "$inventory_manifest" \
      --output "$waveform_report" \
      --sample-rate 2048 \
      --reference-duration 128 \
      --per-family 3 \
      --selection-mode family_approximant \
      --include-alternatives \
      --runtime-receipt "$waveform_runtime_receipt"
  fi
  "$task_python" -m gwyolo.cli gwtc5-locked-corpus-freeze \
    --manifest "$inventory_manifest" \
    --inventory-report "$inventory_report" \
    --waveform-validation-report "$waveform_report" \
    --suite-config "$suite_config" \
    --access-log "$access_log" \
    --output "$freeze_report"
  "$task_python" -m gwyolo.cli publication-evidence-audit \
    --config configs/publication_validation_evidence.yaml \
    --evidence "locked_corpus_unopened=$freeze_report" \
    --output "$ledger_report"
)

"$task_python" - "$output_dir" "$access_log" "$code_commit" <<'PY'
import hashlib
import json
import pathlib
import sys

output_dir = pathlib.Path(sys.argv[1])
access_log = pathlib.Path(sys.argv[2])
code_commit = sys.argv[3]
plan_dir = output_dir / "injection-plan"
manifest = plan_dir / "gwtc5_locked_injection_inventory.jsonl"
inventory_path = plan_dir / "gwtc5_locked_injection_inventory_report.json"
freeze_path = output_dir / "gwtc5_locked_corpus_unopened.json"
waveform_path = output_dir / "waveform_validation_report.json"
ledger_path = output_dir / "locked_corpus_gate_replay.json"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
waveform = json.loads(waveform_path.read_text(encoding="utf-8"))
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
gate = next(row for row in ledger["requirements"] if row["id"] == "locked_corpus_unopened")
if (
    inventory.get("status") != "score_blind_gwtc5_locked_injection_inventory"
    or inventory.get("passed") is not True
    or inventory.get("physical_stress_predicates_passed") is not True
    or int(inventory.get("rows", 0)) < 4000
    or int(inventory.get("minimum_usable_after_dq", 0)) < 3000
    or inventory.get("manifest_sha256") != digest(manifest)
    or inventory.get("candidate_scores_inspected") is not False
    or int(inventory.get("test_strain_rows_read", -1)) != 0
    or inventory.get("pre_access_vt_weights_assigned") is not False
    or inventory.get("post_access_dq_replacement_allowed") is not False
    or waveform.get("passed") is not True
    or waveform.get("selection_mode") != "family_approximant"
    or waveform.get("include_alternatives") is not True
    or int(waveform.get("selected_cases", 0)) < 3
    or any(int(value) < 3 for value in waveform.get("case_strata", {}).values())
    or freeze.get("status") != "locked_evaluation_corpus_unopened"
    or freeze.get("inventory_producer_bound") is not True
    or freeze.get("physical_stress_predicates_passed") is not True
    or freeze.get("waveform_runtime_validation_bound") is not True
    or freeze.get("code_commit") != code_commit
    or gate.get("state") != "passed"
    or waveform.get("runtime_receipt_bound") is not True
    or len(gate.get("artifact_replay", [])) != 8
    or access_log.exists()
):
    raise SystemExit("GWTC-5 physical locked-injection replay failed")
PY
