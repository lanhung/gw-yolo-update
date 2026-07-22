#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  NUMERIC_VALIDATION_MANIFEST
  AUDIT_OUTPUT_DIR
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required human-mask audit freeze variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]] \
  || [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
    != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "human-mask audit freeze requires its declared immutable checkout" >&2
  exit 2
fi
for path in "$TASK_PYTHON" "$NUMERIC_VALIDATION_MANIFEST"; do
  if [[ ! -s "$path" ]]; then
    echo "human-mask audit freeze input is absent: $path" >&2
    exit 2
  fi
done
if [[ -e "$AUDIT_OUTPUT_DIR" ]]; then
  echo "human-mask audit freeze output already exists: $AUDIT_OUTPUT_DIR" >&2
  exit 2
fi

(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  "$TASK_PYTHON" -m gwyolo.cli gravityspy-mask-audit-plan \
    --manifest "$NUMERIC_VALIDATION_MANIFEST" \
    --output-dir "$AUDIT_OUTPUT_DIR" \
    --per-label "${MASK_AUDIT_PER_LABEL:-5}" \
    --seed "${MASK_AUDIT_SEED:-20260720}"
)

"$TASK_PYTHON" - "$AUDIT_OUTPUT_DIR" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = pathlib.Path(sys.argv[1])
commit = sys.argv[2]
report = json.loads((root / "gravityspy_mask_audit_plan_report.json").read_text())
task_manifest = pathlib.Path(report.get("task_manifest_path", ""))
annotation_manifest = pathlib.Path(report.get("annotation_task_manifest_path", ""))
if (
    report.get("status") != "frozen_gravityspy_human_mask_audit_plan"
    or report.get("scientific_claim_allowed") is not False
    or report.get("mask_targets_exposed_to_annotators") is not False
    or int(report.get("tasks", 0)) < 90
    or int(report.get("unique_glitches", 0)) < 90
    or report.get("code_commit") != commit
    or not task_manifest.is_file()
    or not annotation_manifest.is_file()
    or report.get("task_manifest_sha256") != digest(task_manifest)
    or report.get("annotation_task_manifest_sha256") != digest(annotation_manifest)
):
    raise SystemExit("human-mask audit freeze failed replay")
print(json.dumps({
    "status": "human_annotation_required",
    "tasks": report["tasks"],
    "labels": report["label_counts"],
    "annotator_manifest": str(annotation_manifest),
}, sort_keys=True))
PY
