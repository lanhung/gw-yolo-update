#!/usr/bin/env bash
set -euo pipefail

# Bind a clean-only detector-set bootstrap to its fixed-channel teacher before
# any real-glitch overlap fine-tuning is allowed to use the checkpoint.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  FIXED_CHANNEL_REPORT
  FIXED_CHANNEL_CHECKPOINT
  DETECTOR_SET_REPORT
  DETECTOR_SET_CHECKPOINT
  OUTPUT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required clean-bootstrap variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$FIXED_CHANNEL_REPORT" \
  "$FIXED_CHANNEL_CHECKPOINT" \
  "$DETECTOR_SET_REPORT" \
  "$DETECTOR_SET_CHECKPOINT"; do
  if [[ ! -s "$path" ]]; then
    echo "clean-bootstrap input is absent: $path" >&2
    exit 3
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "clean-bootstrap checkout differs from its declared commit" >&2
  exit 3
fi

minimum_retention=${MINIMUM_CLEAN_CHIRP_IOU_RETENTION:-0.95}
"$TASK_PYTHON" - \
  "$FIXED_CHANNEL_REPORT" \
  "$FIXED_CHANNEL_CHECKPOINT" \
  "$DETECTOR_SET_REPORT" \
  "$DETECTOR_SET_CHECKPOINT" \
  "$minimum_retention" \
  "$OUTPUT" \
  "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import platform
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    fixed_report_path,
    fixed_checkpoint_path,
    detector_report_path,
    detector_checkpoint_path,
    minimum_retention,
    output_path,
    commit,
) = sys.argv[1:]
minimum_retention = float(minimum_retention)
if not 0 < minimum_retention <= 1:
    raise SystemExit("clean-bootstrap retention threshold is invalid")
fixed = json.loads(pathlib.Path(fixed_report_path).read_text(encoding="utf-8"))
detector = json.loads(
    pathlib.Path(detector_report_path).read_text(encoding="utf-8")
)
fixed_iou = float(fixed["calibrated_validation"]["chirp_iou"])
detector_iou = float(detector["calibrated_validation"]["chirp_iou"])
retention = detector_iou / max(fixed_iou, 1e-12)
dropout = detector.get("detector_set", {})
passed = bool(
    fixed.get("status") == "physical_real_noise_validation_only_finetune"
    and fixed.get("test_evaluation") is None
    and fixed.get("checkpoint_sha256") == digest(fixed_checkpoint_path)
    and detector.get("status") == "physical_real_noise_validation_only_finetune"
    and detector.get("architecture") == "detector_set"
    and detector.get("test_evaluation") is None
    and detector.get("checkpoint_sha256") == digest(detector_checkpoint_path)
    and detector.get("pretrained_checkpoint_sha256")
    == digest(fixed_checkpoint_path)
    and detector.get("validation_manifest_sha256")
    == fixed.get("validation_manifest_sha256")
    and dropout.get("enabled") is True
    and float(dropout.get("training_dropout_probability", -1)) == 0.5
    and int(dropout.get("minimum_available_detectors", -1)) == 2
    and math.isfinite(fixed_iou)
    and fixed_iou > 0
    and math.isfinite(detector_iou)
    and retention >= minimum_retention
)
result = {
    "status": "completed_clean_detector_set_bootstrap_gate",
    "passed": passed,
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "code_commit": commit,
    "minimum_clean_chirp_iou_retention": minimum_retention,
    "fixed_channel_validation_chirp_iou": fixed_iou,
    "detector_set_validation_chirp_iou": detector_iou,
    "clean_chirp_iou_retention": retention,
    "detector_dropout_probability": dropout.get(
        "training_dropout_probability"
    ),
    "minimum_available_detectors": dropout.get(
        "minimum_available_detectors"
    ),
    "fixed_channel": {
        "report_path": str(pathlib.Path(fixed_report_path).resolve()),
        "report_sha256": digest(fixed_report_path),
        "checkpoint_path": str(pathlib.Path(fixed_checkpoint_path).resolve()),
        "checkpoint_sha256": digest(fixed_checkpoint_path),
    },
    "detector_set": {
        "report_path": str(pathlib.Path(detector_report_path).resolve()),
        "report_sha256": digest(detector_report_path),
        "checkpoint_path": str(
            pathlib.Path(detector_checkpoint_path).resolve()
        ),
        "checkpoint_sha256": digest(detector_checkpoint_path),
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
    },
}
target = pathlib.Path(output_path)
target.parent.mkdir(parents=True, exist_ok=True)
part = target.with_suffix(target.suffix + ".part")
part.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
if not passed:
    raise SystemExit("clean detector-set bootstrap failed non-inferiority")
PY

