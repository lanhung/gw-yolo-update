#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIVE_SEED_SUMMARY
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  OVERLAP_MANIFEST
  INJECTION_MANIFEST
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$FIVE_SEED_SUMMARY" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$OVERLAP_MANIFEST" \
  "$INJECTION_MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "required promoted PE input is absent: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 2
fi

if ! selection_output=$(
  "$TASK_PYTHON" - \
    "$FIVE_SEED_SUMMARY" \
    "$UNIFORM_CONFIG" \
    "$FAMILY_BALANCED_CONFIG" \
    "$OVERLAP_MANIFEST" \
    "$INJECTION_MANIFEST" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


summary_path, uniform, balanced, overlap, injections = sys.argv[1:]
summary = json.loads(pathlib.Path(summary_path).read_text(encoding="utf-8"))
if (
    summary.get("status") != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("test_data_opened") is not False
):
    raise SystemExit("five-seed summary is not a validation-only promoted result")
arm = summary.get("promoted_arm")
if arm == "uniform":
    config = uniform
elif arm == "family_balanced":
    config = balanced
else:
    raise SystemExit("five-seed summary selected an unknown arm")
selected_checkpoint = str(summary["selected_checkpoint_path"])
if digest(selected_checkpoint) != str(summary["selected_checkpoint_sha256"]):
    raise SystemExit("selected checkpoint differs from five-seed summary")
matches = []
for identity in summary.get("finetune_reports", []):
    path = str(identity["path"])
    if digest(path) != str(identity["sha256"]):
        raise SystemExit("a five-seed finetune report hash changed")
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if str(report.get("checkpoint_path")) == selected_checkpoint:
        matches.append((path, report))
if len(matches) != 1:
    raise SystemExit("selected checkpoint does not resolve to exactly one finetune report")
report_path, report = matches[0]
expected = {
    "checkpoint_sha256": digest(selected_checkpoint),
    "config_file_sha256": digest(config),
    "overlap_validation_manifest_sha256": digest(overlap),
    "clean_validation_manifest_sha256": digest(injections),
}
if any(str(report.get(field)) != value for field, value in expected.items()):
    raise SystemExit("selected model report differs from paired PE inputs/configuration")
if report.get("status") != "validation_selected_real_glitch_overlap_finetune":
    raise SystemExit("selected model report has the wrong status")
print(report_path)
print(config)
PY
); then
  echo "promoted paired PE model selection failed" >&2
  exit 2
fi
readarray -t selection <<<"$selection_output"
if (( ${#selection[@]} != 2 )) || [[ -z "${selection[0]}" || -z "${selection[1]}" ]]; then
  echo "promoted paired PE selection returned an invalid result" >&2
  exit 2
fi

export GWYOLO_PYTHON="$TASK_PYTHON"
export GWYOLO_REPO="$TASK_CODE_DIR"
export GWYOLO_OUTPUT_ROOT="$OUTPUT_ROOT"
export GWYOLO_OVERLAP_MANIFEST="$OVERLAP_MANIFEST"
export GWYOLO_INJECTION_MANIFEST="$INJECTION_MANIFEST"
export GWYOLO_MODEL_REPORT="${selection[0]}"
export GWYOLO_MODEL_CONFIG="${selection[1]}"
export GWYOLO_PE_SMOKE_LIMIT="${PE_SMOKE_LIMIT:-3}"
export GWYOLO_PE_SELECTION_SEED="${PE_SELECTION_SEED:-20260722}"
cd "$TASK_CODE_DIR"
bash scripts/run_paired_pe_smoke.sh
