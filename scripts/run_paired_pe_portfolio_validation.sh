#!/usr/bin/env bash
set -euo pipefail

# Aggregate strict within-backend DINGO and AMPLFI validation evidence on the
# same events. This path never permits an absolute cross-backend ranking.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  DINGO_WITHIN_SUMMARY
  AMPLFI_WITHIN_SUMMARY
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in "$TASK_PYTHON" "$DINGO_WITHIN_SUMMARY" "$AMPLFI_WITHIN_SUMMARY"; do
  if [[ ! -s "$path" ]]; then
    echo "required paired PE portfolio artifact is absent: $path" >&2
    exit 3
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

if ! resolved_output=$(
  "$TASK_PYTHON" - "$DINGO_WITHIN_SUMMARY" "$AMPLFI_WITHIN_SUMMARY" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


specifications = (
    (
        "DINGO",
        pathlib.Path(sys.argv[1]),
        "validation_only_dingo_official_native_paired_smoke_complete",
    ),
    (
        "AMPLFI",
        pathlib.Path(sys.argv[2]),
        "validation_only_amplfi_within_backend_paired_smoke_complete",
    ),
)
for backend, summary_path, status in specifications:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts", {})
    if (
        summary.get("status") != status
        or summary.get("scientific_claim_allowed") is not False
        or summary.get("cross_backend_absolute_comparison_allowed") is not False
        or set(("posterior_batch", "robustness")) - set(artifacts)
    ):
        raise SystemExit(f"{backend} within-backend summary boundary failed")
    for label in ("posterior_batch", "robustness"):
        identity = artifacts[label]
        path = pathlib.Path(identity.get("path", "")).resolve()
        if not path.is_file() or digest(path) != identity.get("sha256"):
            raise SystemExit(f"{backend} within-backend artifact changed: {label}")
        print(path)
PY
); then
  echo "paired PE portfolio input resolution failed" >&2
  exit 4
fi
readarray -t resolved <<<"$resolved_output"
if (( ${#resolved[@]} != 4 )) || [[ -z "${resolved[0]}" || -z "${resolved[3]}" ]]; then
  echo "paired PE portfolio input resolution failed" >&2
  exit 4
fi

cd "$TASK_CODE_DIR"
export PYTHONPATH=src
export GWYOLO_CODE_COMMIT
mkdir -p "$OUTPUT_ROOT/logs"

"$TASK_PYTHON" -m gwyolo.cli pe-robustness-portfolio-evaluate \
  --dingo-batch-report "${resolved[0]}" \
  --dingo-robustness-report "${resolved[1]}" \
  --amplfi-batch-report "${resolved[2]}" \
  --amplfi-robustness-report "${resolved[3]}" \
  --manifest-output "$OUTPUT_ROOT/paired_pe_portfolio.jsonl" \
  --output "$OUTPUT_ROOT/paired_pe_portfolio.json" \
  --credible-level "${PE_CREDIBLE_LEVEL:-0.9}" \
  --bootstrap-replicates "${PE_BOOTSTRAP_REPLICATES:-10000}" \
  --bootstrap-seed "${PE_BOOTSTRAP_SEED:-20260721}" \
  --required-split val \
  >"$OUTPUT_ROOT/logs/portfolio-evaluation.log" 2>&1

"$TASK_PYTHON" -m gwyolo.cli pe-robustness-promote \
  --joint-report "$OUTPUT_ROOT/paired_pe_portfolio.json" \
  --config configs/pe_robustness_promotion.yaml \
  --output "$OUTPUT_ROOT/paired_pe_portfolio_promotion.json" \
  >"$OUTPUT_ROOT/logs/portfolio-promotion.log" 2>&1

"$TASK_PYTHON" - "$OUTPUT_ROOT" "$DINGO_WITHIN_SUMMARY" \
  "$AMPLFI_WITHIN_SUMMARY" "$GWYOLO_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


root = pathlib.Path(sys.argv[1]).resolve()
sources = {
    "dingo_within_summary": pathlib.Path(sys.argv[2]).resolve(),
    "amplfi_within_summary": pathlib.Path(sys.argv[3]).resolve(),
    "portfolio": root / "paired_pe_portfolio.json",
    "promotion": root / "paired_pe_portfolio_promotion.json",
}
commit = sys.argv[4]
reports = {
    label: json.loads(path.read_text(encoding="utf-8"))
    for label, path in sources.items()
}
portfolio = reports["portfolio"]
promotion = reports["promotion"]
if (
    portfolio.get("status")
    != "paired_dingo_amplfi_within_backend_portfolio_complete"
    or portfolio.get("comparison_scope")
    != "matched_event_within_backend_deltas_only"
    or portfolio.get("absolute_cross_backend_comparison_allowed") is not False
    or portfolio.get("matched_event_gate") is not True
    or portfolio.get("within_backend_provenance_gate") is not True
    or portfolio.get("required_split") != "val"
    or portfolio.get("test_rows_read") != 0
    or promotion.get("status") != "pe_robustness_validation_promotion_decision"
    or promotion.get("evidence_mode")
    != "matched_event_within_backend_portfolio"
    or promotion.get("absolute_cross_backend_comparison_allowed") is not False
):
    raise SystemExit("paired PE portfolio violated its comparison boundary")
artifacts = {
    label: {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for label, path in sources.items()
}
result = {
    "status": "validation_only_paired_pe_portfolio_complete",
    "passed": promotion.get("passed") is True,
    "promote_to_locked_test": promotion.get("promote_to_locked_test") is True,
    "scientific_claim_allowed": False,
    "scientific_blocker": (
        "this is validation-only matched-event within-backend evidence; locked evaluation "
        "is still required and absolute DINGO/AMPLFI ranking remains forbidden"
    ),
    "comparison_scope": "matched_event_within_backend_deltas_only",
    "absolute_cross_backend_comparison_allowed": False,
    "matched_event_gate": True,
    "within_backend_provenance_gate": True,
    "paired_injections": portfolio["common_injection_count"],
    "bootstrap_replicates": portfolio["bootstrap_replicates"],
    "test_rows_read": 0,
    "code_commit": commit,
    "artifacts": artifacts,
}
target = root / "paired_pe_portfolio_summary.json"
temporary = target.with_suffix(".json.part")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
