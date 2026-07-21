#!/usr/bin/env bash
set -euo pipefail

# Exercise the native DINGO 0.5.8 EventDataset -> GNPE posterior path with a
# deterministic synthetic zero-strain event. This is runtime evidence only.

required=(
  TASK_PYTHON
  DINGO_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  DINGO_MODEL
  DINGO_MODEL_SHA256
  DINGO_MODEL_INIT
  DINGO_MODEL_INIT_SHA256
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" "$DINGO_PYTHON" "$DINGO_MODEL" "$DINGO_MODEL_INIT" \
  "$TASK_CODE_DIR/scripts/run_dingo_common_event.py"; do
  if [[ ! -s "$path" ]]; then
    echo "required native-event smoke input is absent: $path" >&2
    exit 2
  fi
done
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi
if [[ "$($DINGO_PYTHON -c 'import importlib.metadata; print(importlib.metadata.version("dingo-gw"))')" \
  != "0.5.8" ]]; then
  echo "native-event smoke requires DINGO 0.5.8" >&2
  exit 3
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "native-event smoke refuses an existing output root" >&2
  exit 3
fi
mkdir -p "$OUTPUT_ROOT"
event="$OUTPUT_ROOT/synthetic_zero_event.hdf5"

event_sha=$(
  "$TASK_PYTHON" - "$event" <<'PY'
import ast
import hashlib
import os
import pathlib
import tempfile
import sys

import h5py
import numpy as np


target = pathlib.Path(sys.argv[1]).resolve()
sample_rate = 4096
duration = 16.0
post_trigger = 2.0
frequency_bins = int(1024.0 * duration) + 1
settings = {
    "time_event": 0.0,
    "time_buffer": post_trigger,
    "detectors": ["H1", "L1"],
    "f_s": sample_rate,
    "T": duration,
    "window_type": "tukey",
    "roll_off": 0.4,
    "minimum_frequency": {"H1": 20.0, "L1": 20.0},
    "maximum_frequency": {"H1": 1024.0, "L1": 1024.0},
    "gwyolo_condition": "synthetic_runtime_smoke_only",
    "gwyolo_injection_id": "synthetic-zero-not-physical",
}
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{target.name}.", suffix=".hdf5", dir=target.parent
)
os.close(descriptor)
temporary = pathlib.Path(temporary_name)
try:
    with h5py.File(temporary, "w") as handle:
        handle.attrs["dataset_type"] = "event_dataset"
        handle.attrs["settings"] = repr(settings)
        handle.create_dataset("version", data=np.bytes_("gwyolo-dingo-runtime-smoke-v1"))
        data = handle.create_group("data")
        waveforms = data.create_group("waveform")
        asds = data.create_group("asds")
        for ifo in ("H1", "L1"):
            waveforms.create_dataset(ifo, data=np.zeros(frequency_bins, dtype=np.complex128))
            asds.create_dataset(ifo, data=np.ones(frequency_bins, dtype=np.float64))
    with h5py.File(temporary, "r") as handle:
        if ast.literal_eval(handle.attrs["settings"])["detectors"] != ["H1", "L1"]:
            raise SystemExit("synthetic DINGO event settings failed replay")
    temporary.replace(target)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
print(hashlib.sha256(target.read_bytes()).hexdigest())
PY
)

"$DINGO_PYTHON" "$TASK_CODE_DIR/scripts/run_dingo_common_event.py" \
  --event "$event" \
  --model "$DINGO_MODEL" \
  --model-init "$DINGO_MODEL_INIT" \
  --posterior-output "$OUTPUT_ROOT/posterior.npz" \
  --result-output "$OUTPUT_ROOT/dingo_result.hdf5" \
  --report-output "$OUTPUT_ROOT/dingo_inference_report.json" \
  --expected-event-sha256 "$event_sha" \
  --expected-model-sha256 "$DINGO_MODEL_SHA256" \
  --expected-model-init-sha256 "$DINGO_MODEL_INIT_SHA256" \
  --num-samples 4 \
  --batch-size 2 \
  --num-gnpe-iterations 1 \
  --device cuda \
  --seed 20260721 \
  >"$OUTPUT_ROOT/inference.log" 2>&1

"$TASK_PYTHON" - "$OUTPUT_ROOT" "$event_sha" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys


root = pathlib.Path(sys.argv[1]).resolve()
event_sha, code_commit = sys.argv[2:]
report_path = root / "dingo_inference_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
if (
    report.get("status") != "real_dingo_gnpe_posterior_complete"
    or report.get("backend_version") != "0.5.8"
    or report.get("model_load_api") != "dingo.core.models.PosteriorModel"
    or report.get("event_sha256") != event_sha
    or report.get("posterior_samples") != 4
):
    raise SystemExit("native DINGO event smoke did not pass")
result = {
    "status": "dingo_official_native_synthetic_event_runtime_smoke_complete",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": "synthetic zero-strain event; not an injection or detector-data result",
    "test_rows_read": 0,
    "test_evaluation": None,
    "code_commit": code_commit,
    "event_sha256": event_sha,
    "inference_report_path": str(report_path),
    "inference_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
}
target = root / "dingo_native_event_smoke_summary.json"
temporary = target.with_suffix(".json.part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
