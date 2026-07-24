#!/usr/bin/env bash
set -euo pipefail

# Wait for the validation-only hard-endpoint decision, then replay the exact
# detector-compatible physical-source capacity gate. This queue never opens a
# test partition and never launches training.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  HARD_ENDPOINT_REPORT
  CURRENT_OVERLAP_MANIFEST
  CANDIDATE_GLITCH_MANIFEST
  CANDIDATE_INJECTION_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  OUTPUT_REPORT
  QUEUE_RECEIPT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout" >&2
  exit 3
fi
observed_commit=$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)
if [[ "$observed_commit" != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "TASK_CODE_DIR commit differs from GWYOLO_CODE_COMMIT" >&2
  exit 3
fi

write_incomplete_receipt() {
  mkdir -p "$(dirname "$QUEUE_RECEIPT")"
  "$TASK_PYTHON" - "$QUEUE_RECEIPT" "$GWYOLO_CODE_COMMIT" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
result = {
    "status": "physical_overlap_expansion_capacity_queue_upstream_incomplete",
    "passed": False,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "upstream ended without the frozen validation hard-endpoint report; "
        "next-scale training remains unauthorized"
    ),
    "test_rows_read": 0,
    "test_evaluation": None,
    "code_commit": sys.argv[2],
}
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
}

while [[ ! -s "$HARD_ENDPOINT_REPORT" ]]; do
  if [[ -n "${UPSTREAM_PID:-}" ]] && ! kill -0 "$UPSTREAM_PID" 2>/dev/null; then
    write_incomplete_receipt
    exit 0
  fi
  sleep "${QUEUE_POLL_SECONDS:-30}"
done

export PYTHONPATH="$TASK_CODE_DIR/src"
export GWYOLO_CODE_COMMIT
candidate_injection_audit_args=()
if [[ -n "${CANDIDATE_INJECTION_AUDIT:-}" ]]; then
  if [[ ! -s "$CANDIDATE_INJECTION_AUDIT" ]]; then
    echo "candidate injection audit is absent" >&2
    exit 2
  fi
  candidate_injection_audit_args+=(
    --candidate-injection-audit "$CANDIDATE_INJECTION_AUDIT"
  )
fi
"$TASK_PYTHON" -m gwyolo.cli physical-overlap-expansion-capacity \
  --hard-endpoint-report "$HARD_ENDPOINT_REPORT" \
  --current-overlap-manifest "$CURRENT_OVERLAP_MANIFEST" \
  --candidate-glitch-manifest "$CANDIDATE_GLITCH_MANIFEST" \
  --candidate-injection-manifest "$CANDIDATE_INJECTION_MANIFEST" \
  "${candidate_injection_audit_args[@]}" \
  --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT" \
  --output "$OUTPUT_REPORT" \
  --seed "${OVERLAP_PAIRING_SEED:-20260728}"

mkdir -p "$(dirname "$QUEUE_RECEIPT")"
"$TASK_PYTHON" - "$QUEUE_RECEIPT" "$OUTPUT_REPORT" \
  "$HARD_ENDPOINT_REPORT" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
capacity = pathlib.Path(sys.argv[2]).resolve()
hard_endpoint = pathlib.Path(sys.argv[3]).resolve()
report = json.loads(capacity.read_text(encoding="utf-8"))
if (
    report.get("status") != "audited_physical_overlap_expansion_capacity"
    or report.get("passed") is not True
    or report.get("test_rows_read") != 0
    or report.get("test_evaluation") is not None
):
    raise SystemExit("physical-overlap expansion capacity crossed its validation boundary")
result = {
    "status": "physical_overlap_expansion_capacity_queue_completed",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": report.get("scientific_blocker"),
    "test_rows_read": 0,
    "test_evaluation": None,
    "next_scale_training_authorized": report.get(
        "next_scale_training_authorized"
    ),
    "expansion_mode": report.get("expansion_mode"),
    "artifacts": {
        "capacity_report": {
            "path": str(capacity),
            "sha256": hashlib.sha256(capacity.read_bytes()).hexdigest(),
        },
        "hard_endpoint_report": {
            "path": str(hard_endpoint),
            "sha256": hashlib.sha256(hard_endpoint.read_bytes()).hexdigest(),
        },
    },
    "code_commit": sys.argv[4],
}
temporary = target.with_suffix(target.suffix + ".part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, target)
PY
