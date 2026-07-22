#!/usr/bin/env bash
set -euo pipefail

# Materialize validation-only clean/contaminated/mask-conditioned posterior
# inputs after a detector-set overlap model has completed. The historical
# "smoke" filename/status is retained for backwards-compatible replay; the
# report's evaluation_tier distinguishes bounded smoke from the predeclared
# publication-validation batch.
# Every machine path is supplied explicitly through the environment.

required=(
  GWYOLO_PYTHON
  GWYOLO_REPO
  GWYOLO_OUTPUT_ROOT
  GWYOLO_OVERLAP_MANIFEST
  GWYOLO_INJECTION_MANIFEST
  GWYOLO_MODEL_REPORT
  GWYOLO_MODEL_CONFIG
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

if [[ -n "${GWYOLO_WAIT_FOR_PID:-}" ]]; then
  while kill -0 "$GWYOLO_WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
fi

cd "$GWYOLO_REPO"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT="${GWYOLO_CODE_COMMIT:-$(git rev-parse --short=7 HEAD)}"

for path in \
  "$GWYOLO_OVERLAP_MANIFEST" \
  "$GWYOLO_INJECTION_MANIFEST" \
  "$GWYOLO_MODEL_REPORT" \
  "$GWYOLO_MODEL_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "required paired-PE source is absent: $path" >&2
    exit 3
  fi
done

mkdir -p "$GWYOLO_OUTPUT_ROOT"
checkpoint=$(
  "$GWYOLO_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' \
    "$GWYOLO_MODEL_REPORT"
)
if [[ ! -s "$checkpoint" ]]; then
  echo "validation-selected overlap checkpoint is absent: $checkpoint" >&2
  exit 4
fi
expected_config_sha256=$(
  "$GWYOLO_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["config_file_sha256"])' \
    "$GWYOLO_MODEL_REPORT"
)
observed_config_sha256=$(sha256sum "$GWYOLO_MODEL_CONFIG" | awk '{print $1}')
if [[ "$observed_config_sha256" != "$expected_config_sha256" ]]; then
  echo "selected overlap model/config hash mismatch" >&2
  exit 4
fi

contamination="$GWYOLO_OUTPUT_ROOT/contamination"
scores="$GWYOLO_OUTPUT_ROOT/contaminated-scores"
masked="$GWYOLO_OUTPUT_ROOT/mask-conditioned"
common="$GWYOLO_OUTPUT_ROOT/common-sources"
dingo="$GWYOLO_OUTPUT_ROOT/dingo-native"
amplfi="$GWYOLO_OUTPUT_ROOT/amplfi-native"

if [[ ! -s "$contamination/contaminated_injection_report.json" ]]; then
  "$GWYOLO_PYTHON" -m gwyolo.cli physical-overlap-contamination \
    --overlap-manifest "$GWYOLO_OVERLAP_MANIFEST" \
    --injection-manifest "$GWYOLO_INJECTION_MANIFEST" \
    --output-dir "$contamination" \
    --required-split val
fi

scores_complete=false
if [[ -s "$scores/injection_score_report.json" ]]; then
  scores_complete=$(
    "$GWYOLO_PYTHON" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print(str(d.get("failed_injections", 1) == 0 and d.get("scored_injections") == d.get("input_injections")).lower())' \
      "$scores/injection_score_report.json"
  )
fi
if [[ "$scores_complete" != true ]]; then
  while true; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  "$GWYOLO_PYTHON" -m gwyolo.cli injection-score \
    --manifest "$contamination/contaminated_injection_val.jsonl" \
    --checkpoint "$checkpoint" \
    --config "$GWYOLO_MODEL_CONFIG" \
    --output-dir "$scores" \
    --model-ifos H1 L1 V1 \
    --q-values 4 8 16 \
    --target-sample-rate 1024 \
    --save-probabilities \
    --required-split val
fi

if [[ ! -s "$masked/learned_deglitch_report.json" ]]; then
  "$GWYOLO_PYTHON" -m gwyolo.cli learned-deglitch \
    --materialized-manifest "$contamination/contaminated_injection_val.jsonl" \
    --scored-manifest "$scores/injection_triggers.jsonl" \
    --output-dir "$masked" \
    --strength 0.9
fi

if [[ ! -s "$common/common_pe_inputs_report.json" ]]; then
  "$GWYOLO_PYTHON" -m gwyolo.cli pe-input-materialize \
    --clean-manifest "$contamination/paired_clean_injection_val.jsonl" \
    --contaminated-manifest "$contamination/contaminated_injection_val.jsonl" \
    --mask-conditioned-manifest "$masked/learned_deglitch.jsonl" \
    --common-prior configs/pe_common_bbh_analysis_prior.yaml \
    --mask-model "$checkpoint" \
    --mask-policy configs/pe_mask_conditioning_policy.yaml \
    --output-dir "$common" \
    --required-split val \
    --required-ifos H1 L1 \
    --source-sample-rate-hz 4096 \
    --source-duration-seconds 16 \
    --source-post-trigger-seconds 2 \
    --analysis-high-frequency-hz 1024 \
    --limit "${GWYOLO_PE_SMOKE_LIMIT:-3}" \
    --selection-seed "${GWYOLO_PE_SELECTION_SEED:-20260721}"
fi

if [[ ! -s "$dingo/native_conditioning_report.json" ]]; then
  "$GWYOLO_PYTHON" -m gwyolo.cli pe-native-condition \
    --source-manifest "$common/common_pe_inputs.jsonl" \
    --config configs/dingo_o4a_native_conditioning.yaml \
    --output-dir "$dingo" \
    --required-split val
fi

if [[ ! -s "$amplfi/native_conditioning_report.json" ]]; then
  "$GWYOLO_PYTHON" -m gwyolo.cli pe-native-condition \
    --source-manifest "$common/common_pe_inputs.jsonl" \
    --config configs/amplfi_common_native_conditioning.yaml \
    --output-dir "$amplfi" \
    --required-split val
fi

"$GWYOLO_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["GWYOLO_OUTPUT_ROOT"]).resolve()
reports = {
    "contamination": root / "contamination/contaminated_injection_report.json",
    "scores": root / "contaminated-scores/injection_score_report.json",
    "mask_conditioned": root / "mask-conditioned/learned_deglitch_report.json",
    "common_sources": root / "common-sources/common_pe_inputs_report.json",
    "dingo_native": root / "dingo-native/native_conditioning_report.json",
    "amplfi_native": root / "amplfi-native/native_conditioning_report.json",
}
missing = [str(path) for path in reports.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(f"paired PE smoke reports are missing: {missing}")
receipt_variables = {
    "model_selection_overlap_manifest": "GWYOLO_MODEL_SELECTION_OVERLAP_MANIFEST",
    "model_selection_validation_manifest": "GWYOLO_MODEL_SELECTION_VALIDATION_MANIFEST",
    "independent_validation_endpoint": "GWYOLO_INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
    "independent_pe_overlap": "GWYOLO_INDEPENDENT_PE_OVERLAP_REPORT",
    "independent_overlap_audit": "GWYOLO_INDEPENDENT_OVERLAP_AUDIT",
}
receipt_paths = {
    label: os.environ.get(variable)
    for label, variable in receipt_variables.items()
}
if any(receipt_paths.values()) and not all(receipt_paths.values()):
    raise RuntimeError("promoted paired PE receipt environment is only partially populated")
source_receipts = {}
for label, value in receipt_paths.items():
    if value is None:
        continue
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"paired PE source receipt is absent: {path}")
    source_receipts[label] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
loaded_reports = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in reports.items()
}
paired_injections = int(loaded_reports["common_sources"]["paired_injections"])
minimum_publication_validation_injections = 100
evaluation_tier = (
    "publication_validation"
    if paired_injections >= minimum_publication_validation_injections
    else "bounded_smoke"
)
if evaluation_tier == "publication_validation":
    blocker = (
        "validation-only paired inputs meet the predeclared event-count floor, but "
        "backend posteriors, 10000-replicate paired uncertainty, promotion, and locked "
        "test evaluation remain required"
    )
else:
    blocker = (
        "bounded validation smoke is below the predeclared 100-injection publication "
        "floor; backend posteriors, promotion, and locked test evaluation remain required"
    )
summary = {
    "status": "paired_pe_native_inputs_smoke_complete",
    "scientific_claim_allowed": False,
    "scientific_blocker": blocker,
    "evaluation_tier": evaluation_tier,
    "paired_injections": paired_injections,
    "minimum_publication_validation_injections": (
        minimum_publication_validation_injections
    ),
    "test_rows_read": 0 if source_receipts else None,
    "source_receipts": source_receipts,
    "reports": loaded_reports,
}
target = root / "paired_pe_smoke_summary.json"
temporary = target.with_suffix(".json.part")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
