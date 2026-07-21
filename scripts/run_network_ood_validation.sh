#!/usr/bin/env bash
set -euo pipefail

: "${TASK_PYTHON:?set TASK_PYTHON to the publication environment interpreter}"
: "${TASK_CODE_DIR:?set TASK_CODE_DIR to an immutable GW-YOLO checkout}"
: "${GWYOLO_CODE_COMMIT:?set the exact code commit}"
: "${TRAIN_MANIFEST:?set the source-component-safe aligned train manifest}"
: "${VALIDATION_MANIFEST:?set the source-component-safe aligned validation manifest}"
: "${OUTPUT_ROOT:?set a new resumable artifact directory}"

CONFIG="${CONFIG:-${TASK_CODE_DIR}/configs/glitch_ood_network_contrastive_energy.yaml}"
mkdir -p "${OUTPUT_ROOT}"
cd "${TASK_CODE_DIR}"
export PYTHONPATH=src

protocol="${OUTPUT_ROOT}/held_family_protocol.json"
"${TASK_PYTHON}" -m gwyolo.cli gravityspy-ood-family-freeze \
  --train-manifest "${TRAIN_MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --output "${protocol}" \
  --exclude-family Blip \
  --exclude-family Tomte \
  --minimum-train-rows 20 \
  --minimum-validation-rows 20 \
  --minimum-validation-gps-blocks 5

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
  --seed 20260722

"${TASK_PYTHON}" -m gwyolo.cli glitch-ood-train \
  --config "${CONFIG}" \
  --known-train-manifest "${split_root}/known_train.jsonl" \
  --known-calibration-manifest "${split_root}/known_calibration.jsonl" \
  --heldout-evaluation-manifest "${split_root}/heldout_evaluation.jsonl" \
  --output-dir "${OUTPUT_ROOT}/network_contrastive_energy" \
  --seed 20260722
