#!/usr/bin/env bash
set -euo pipefail

: "${TASK_PYTHON:?set TASK_PYTHON to the publication environment interpreter}"
: "${TASK_CODE_DIR:?set TASK_CODE_DIR to an immutable GW-YOLO checkout}"
: "${GWYOLO_CODE_COMMIT:?set the exact code commit}"
: "${TRAIN_MANIFEST:?set the source-component-safe aligned train manifest}"
: "${VALIDATION_MANIFEST:?set the source-component-safe aligned validation manifest}"
: "${GRAVITYSPY_CORPUS_AUDIT:?set the source-component-safe corpus audit}"
: "${OUTPUT_ROOT:?set a new resumable artifact directory}"

CONFIG="${CONFIG:-${TASK_CODE_DIR}/configs/glitch_ood_network_contrastive_energy.yaml}"
for path in \
  "$TASK_PYTHON" \
  "$TRAIN_MANIFEST" \
  "$VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "required network OOD artifact is absent: $path" >&2
    exit 3
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${TASK_CODE_DIR}"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
export CUDA_VISIBLE_DEVICES="${OOD_CUDA_VISIBLE_DEVICES:-0}"

"$TASK_PYTHON" - \
  "$TRAIN_MANIFEST" \
  "$VALIDATION_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


train, validation, audit_path = map(pathlib.Path, sys.argv[1:])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if (
    audit.get("status") != "verified_group_safe_gravityspy_aligned_network_corpus"
    or audit.get("passed") is not True
    or audit.get("train_manifest_sha256") != digest(train)
    or audit.get("validation_manifest_sha256") != digest(validation)
    or any(audit.get("split_audit", {}).get("cross_split_overlaps", {}).values())
):
    raise SystemExit("network OOD corpus audit replay failed")
PY

protocol="${OUTPUT_ROOT}/held_family_protocol.json"
"${TASK_PYTHON}" -m gwyolo.cli gravityspy-ood-family-freeze \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --output "${protocol}" \
  --exclude-family Blip \
  --exclude-family Tomte \
  --minimum-train-rows 20 \
  --minimum-validation-rows 20 \
  --minimum-validation-gps-blocks 5 \
  >"${OUTPUT_ROOT}/logs/family-freeze.log" 2>&1

held_family=$("${TASK_PYTHON}" - "${protocol}" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "frozen_score_blind_held_glitch_family_protocol":
    raise SystemExit("held-family protocol is not frozen")
if report.get("model_scores_used_for_selection") is not False:
    raise SystemExit("held-family protocol consumed model scores")
print(report["selected"]["glitch_family"])
PY
)

split_root="${OUTPUT_ROOT}/split"
"${TASK_PYTHON}" -m gwyolo.cli gravityspy-ood-split \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --held-out-family "${held_family}" \
  --output-dir "${split_root}" \
  --seed 20260722 \
  >"${OUTPUT_ROOT}/logs/split.log" 2>&1

while true; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  [[ -z "$gpu_pids" ]] && break
  sleep 30
done

"${TASK_PYTHON}" -m gwyolo.cli glitch-ood-train \
  --config "${CONFIG}" \
  --known-train-manifest "${split_root}/known_train.jsonl" \
  --known-calibration-manifest "${split_root}/known_calibration.jsonl" \
  --heldout-evaluation-manifest "${split_root}/heldout_evaluation.jsonl" \
  --output-dir "${OUTPUT_ROOT}/network_contrastive_energy" \
  --seed 20260722 \
  >"${OUTPUT_ROOT}/logs/train.log" 2>&1

"$TASK_PYTHON" - "$OUTPUT_ROOT" "$GRAVITYSPY_CORPUS_AUDIT" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

from gwyolo.io import atomic_write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = pathlib.Path(sys.argv[1]).resolve()
corpus_audit = pathlib.Path(sys.argv[2]).resolve()
code_commit = sys.argv[3]
paths = {
    "held_family_protocol": root / "held_family_protocol.json",
    "split_report": root / "split/leave_one_family_out_report.json",
    "embedding_report": root / "network_contrastive_energy/glitch_ood_embedding_report.json",
}
for label, path in paths.items():
    if not path.is_file():
        raise SystemExit(f"network OOD output is absent: {label}")
protocol = json.loads(paths["held_family_protocol"].read_text(encoding="utf-8"))
split = json.loads(paths["split_report"].read_text(encoding="utf-8"))
report = json.loads(paths["embedding_report"].read_text(encoding="utf-8"))
if (
    protocol.get("status") != "frozen_score_blind_held_glitch_family_protocol"
    or protocol.get("model_scores_used_for_selection") is not False
    or split.get("status") != "frozen_leave_one_glitch_family_out_split"
    or report.get("status") != "known_family_embedding_heldout_ood_validation"
    or report.get("architecture") != "detector_set"
    or report.get("ood_score_method") != "logit_energy"
    or report.get("device") != "cuda"
    or report.get("test_evaluation") is not None
    or report.get("ood_score_fit", {}).get(
        "heldout_scores_used_for_method_or_fit_selection"
    )
    is not False
    or report.get("ood_evaluation", {}).get("calibration", {}).get(
        "unknown_scores_used_for_selection"
    )
    is not False
):
    raise SystemExit("network OOD validation boundary failed")
artifacts = {
    label: {"path": str(path), "sha256": digest(path)} for label, path in paths.items()
}
for label in (
    "checkpoint",
    "known_calibration_scores",
    "heldout_evaluation_scores",
):
    path = pathlib.Path(report[f"{label}_path"])
    if not path.is_file() or digest(path) != report[f"{label}_sha256"]:
        raise SystemExit(f"network OOD artifact replay failed: {label}")
    artifacts[label] = {"path": str(path), "sha256": digest(path)}
artifacts["gravityspy_corpus_audit"] = {
    "path": str(corpus_audit),
    "sha256": digest(corpus_audit),
}
result = {
    "status": "completed_source_safe_detector_set_ood_validation",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "validation-only held-family auxiliary abstention; cannot veto a "
        "strain-coherent candidate or replace later-run OOD evaluation"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "code_commit": code_commit,
    "selected_held_family": protocol["selected"]["glitch_family"],
    "auxiliary_policy": report["auxiliary_policy"],
    "artifacts": artifacts,
}
target = root / "network_ood_validation_receipt.json"
atomic_write_json(target, result)
print(json.dumps(result, indent=2, sort_keys=True))
PY

endpoint="$OUTPUT_ROOT/network_ood_validation_endpoint.json"
if [[ ! -s "$endpoint" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli detector-set-ood-validation-bind \
    --source-receipt "$OUTPUT_ROOT/network_ood_validation_receipt.json" \
    --corpus-audit "$GRAVITYSPY_CORPUS_AUDIT" \
    --output "$endpoint"
fi
