#!/usr/bin/env bash
set -euo pipefail

# Continue the publication candidate-search path only after the source-safe
# five-seed gate has produced an immutable positive receipt. A negative
# promotion is retained as a completed null outcome and never triggers the
# expensive continuous-background acquisition.

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SOURCE_SAFE_CHAIN_RECEIPT
  FIVE_SEED_SUMMARY
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  BASELINE_CHECKPOINT
  BASELINE_CONFIG
  UNIFORM_CONFIG
  FAMILY_BALANCED_CONFIG
  COHERENCE_CONFIG
  PROMOTION_CONFIG
  PARENT_PLAN
  VALIDATION_PURPOSE_AUDIT
  CAPACITY_FORECAST
  EVENT_EXCLUSIONS
  NETWORK_CONFIG
  CACHE_ROOT
  OUTPUT_ROOT
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required post-five-seed candidate variable is unset: $variable" >&2
    exit 2
  fi
done

if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "post-five-seed candidate chain requires its exact checkout" >&2
  exit 2
fi
for input in \
  "$TASK_PYTHON" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$BASELINE_CHECKPOINT" \
  "$BASELINE_CONFIG" \
  "$UNIFORM_CONFIG" \
  "$FAMILY_BALANCED_CONFIG" \
  "$COHERENCE_CONFIG" \
  "$PROMOTION_CONFIG" \
  "$PARENT_PLAN" \
  "$VALIDATION_PURPOSE_AUDIT" \
  "$CAPACITY_FORECAST" \
  "$EVENT_EXCLUSIONS" \
  "$NETWORK_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "post-five-seed candidate input is absent: $input" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_ROOT"
final_receipt="$OUTPUT_ROOT/post_five_seed_candidate_publication_receipt.json"
if [[ -s "$final_receipt" ]]; then
  "$TASK_PYTHON" - "$final_receipt" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    receipt.get("status")
    not in {
        "completed_post_five_seed_candidate_publication_chain",
        "completed_post_five_seed_candidate_negative_promotion",
    }
    or receipt.get("test_rows_read") != 0
    or receipt.get("scientific_claim_allowed") is not False
):
    raise SystemExit("existing post-five-seed candidate receipt failed replay")
PY
  exit 0
fi

upstream_pid=${UPSTREAM_PID:-}
while [[ ! -s "$SOURCE_SAFE_CHAIN_RECEIPT" ]]; do
  if [[ -n "$upstream_pid" ]] && ! kill -0 "$upstream_pid" 2>/dev/null; then
    echo "source-safe overlap chain ended without its immutable receipt" >&2
    exit 1
  fi
  sleep 30
done

selection=$(
  "$TASK_PYTHON" - "$SOURCE_SAFE_CHAIN_RECEIPT" "$FIVE_SEED_SUMMARY" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("execution_passed") is not True
    or receipt.get("scientific_claim_allowed") is not False
    or receipt.get("search_claim_allowed") is not False
    or int(receipt.get("test_rows_read", -1)) != 0
):
    raise SystemExit("source-safe overlap receipt failed replay")
if receipt.get("five_seed_promoted") is not True:
    if receipt.get("status") not in {
        "completed_source_safe_overlap_negative_five_seed",
        "completed_source_safe_overlap_negative_promotion",
    }:
        raise SystemExit("negative source-safe outcome has the wrong status")
    print("SKIP")
    raise SystemExit(0)
entry = receipt.get("five_seed_summary")
if (
    receipt.get("status") != "completed_source_safe_overlap_five_seed_chain"
    or not isinstance(entry, dict)
    or pathlib.Path(str(entry.get("path", ""))).resolve()
    != summary_path.resolve()
    or not summary_path.is_file()
    or entry.get("sha256")
    != hashlib.sha256(summary_path.read_bytes()).hexdigest()
):
    raise SystemExit("positive source-safe outcome does not bind the five-seed summary")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if (
    summary.get("status")
    != "completed_five_seed_source_safe_overlap_validation"
    or summary.get("passed") is not True
    or summary.get("five_seed_stability", {}).get("passed") is not True
    or summary.get("test_data_opened") is not False
):
    raise SystemExit("five-seed summary did not pass its frozen gate")
print("PASS")
PY
)

write_negative_receipt() {
  local reason=$1
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT
    "$TASK_PYTHON" - "$final_receipt" "$SOURCE_SAFE_CHAIN_RECEIPT" \
      "$reason" "${2:--}" <<'PY'
import json
import pathlib
import sys

from gwyolo.io import atomic_write_json, file_sha256
from gwyolo.runtime import execution_provenance

output = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
reason = sys.argv[3]
comparison_arg = sys.argv[4]
comparison = pathlib.Path(comparison_arg) if comparison_arg != "-" else None
result = {
    "status": "completed_post_five_seed_candidate_negative_promotion",
    "execution_passed": True,
    "continuous_background_started": False,
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "retained_negative_reason": reason,
    "source_safe_chain_receipt": {
        "path": str(source.resolve()),
        "sha256": file_sha256(source),
    },
    "candidate_comparison": (
        {
            "path": str(comparison.resolve()),
            "sha256": file_sha256(comparison),
        }
        if comparison is not None
        else None
    ),
    **execution_provenance(),
}
atomic_write_json(output, result)
PY
  )
}

if [[ "$selection" == "SKIP" ]]; then
  write_negative_receipt "source_safe_overlap_gate_did_not_promote_five_seeds"
  exit 0
elif [[ "$selection" != "PASS" ]]; then
  echo "source-safe overlap selector returned an invalid decision" >&2
  exit 2
fi

endpoint_values=$(
  "$TASK_PYTHON" - "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
endpoint = json.loads(path.read_text(encoding="utf-8"))
background = pathlib.Path(
    str(endpoint.get("candidate_calibration_background_manifest_path", ""))
).resolve()
injections = pathlib.Path(
    str(endpoint.get("injection_arrival_manifest_path", ""))
).resolve()
digest = lambda value: hashlib.sha256(value.read_bytes()).hexdigest()
if (
    endpoint.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or endpoint.get("passed") is not True
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or int(endpoint.get("test_rows_read", -1)) != 0
    or endpoint.get("test_evaluation") is not None
    or not background.is_file()
    or not injections.is_file()
    or endpoint.get("candidate_calibration_background_manifest_sha256")
    != digest(background)
    or endpoint.get("injection_arrival_manifest_sha256") != digest(injections)
):
    raise SystemExit("independent validation endpoint failed replay")
print(background)
print(injections)
PY
)
readarray -t endpoint_paths <<<"$endpoint_values"
if (( ${#endpoint_paths[@]} != 2 )); then
  echo "independent endpoint did not resolve background and injection manifests" >&2
  exit 2
fi
validation_background=${endpoint_paths[0]}
validation_injections=${endpoint_paths[1]}

promoted_root="$OUTPUT_ROOT/promoted"
promoted_pipeline="$promoted_root/pipeline/candidate_validation_pipeline_report.json"
(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    FIVE_SEED_SUMMARY="$FIVE_SEED_SUMMARY" \
    INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    BACKGROUND_MANIFEST="$validation_background" \
    INJECTION_MANIFEST="$validation_injections" \
    UNIFORM_CONFIG="$UNIFORM_CONFIG" \
    FAMILY_BALANCED_CONFIG="$FAMILY_BALANCED_CONFIG" \
    COHERENCE_CONFIG="$COHERENCE_CONFIG" \
    OUTPUT_ROOT="$promoted_root" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    bash scripts/run_promoted_candidate_validation.sh
)

baseline_root="$OUTPUT_ROOT/baseline/pipeline"
comparison="$OUTPUT_ROOT/candidate_validation_promotion.json"
(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    SCORING_CODE_DIR="$TASK_CODE_DIR" \
    SCORING_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    BACKGROUND_VAL_MANIFEST="$validation_background" \
    INJECTION_ARRIVAL_MANIFEST="$validation_injections" \
    BASELINE_CHECKPOINT="$BASELINE_CHECKPOINT" \
    BASELINE_CONFIG="$BASELINE_CONFIG" \
    COHERENCE_CONFIG="$COHERENCE_CONFIG" \
    PROMOTED_PIPELINE_REPORT="$promoted_pipeline" \
    BASELINE_OUTPUT_ROOT="$baseline_root" \
    PROMOTION_CONFIG="$PROMOTION_CONFIG" \
    COMPARISON_OUTPUT="$comparison" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    bash scripts/run_candidate_validation_comparison.sh
)

promotion_decision=$(
  "$TASK_PYTHON" - "$comparison" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "paired_validation_candidate_search_promotion"
    or report.get("scientific_claim_allowed") is not False
    or report.get("test_data_opened") is not False
):
    raise SystemExit("candidate validation comparison failed replay")
print("PASS" if report.get("passed") and report.get("scale_continuous_background") else "SKIP")
PY
)
if [[ "$promotion_decision" == "SKIP" ]]; then
  write_negative_receipt "candidate_validation_gate_did_not_authorize_background_scaling" \
    "$comparison"
  exit 0
elif [[ "$promotion_decision" != "PASS" ]]; then
  echo "candidate validation comparison returned an invalid decision" >&2
  exit 2
fi

continuous_root="$OUTPUT_ROOT/continuous-background"
timing_report="$promoted_root/pipeline/candidate_timing_calibration.json"
injection_ranking="$promoted_root/pipeline/injection_rankings/val_injection_candidate_ranking_report.json"
(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    SCORING_CODE_DIR="$TASK_CODE_DIR" \
    SCORING_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    PROMOTION_REPORT="$comparison" \
    PROMOTED_PIPELINE_REPORT="$promoted_pipeline" \
    INDEPENDENT_VALIDATION_ENDPOINT_REPORT="$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    PARENT_PLAN="$PARENT_PLAN" \
    VALIDATION_PURPOSE_AUDIT="$VALIDATION_PURPOSE_AUDIT" \
    CAPACITY_FORECAST="$CAPACITY_FORECAST" \
    EVENT_EXCLUSIONS="$EVENT_EXCLUSIONS" \
    COHERENCE_CONFIG="$COHERENCE_CONFIG" \
    TIMING_CALIBRATION_REPORT="$timing_report" \
    VALIDATION_INJECTION_RANKING_REPORT="$injection_ranking" \
    CACHE_ROOT="$CACHE_ROOT" \
    OUTPUT_ROOT="$continuous_root" \
    SHARD_STOP_EXCLUSIVE="${SHARD_STOP_EXCLUSIVE:-220}" \
    PAIRS_PER_SHARD="${PAIRS_PER_SHARD:-4}" \
    MINIMUM_FREE_KB="${MINIMUM_FREE_KB:-8388608}" \
    DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-2}" \
    FIVE_SEED_SUMMARY="$FIVE_SEED_SUMMARY" \
    UNIFORM_CONFIG="$UNIFORM_CONFIG" \
    FAMILY_BALANCED_CONFIG="$FAMILY_BALANCED_CONFIG" \
    bash scripts/run_candidate_background_range.sh
)

merge_report="$continuous_root/merged/streamed_background_merge_report.json"
background_manifest="$continuous_root/merged/background_windows.jsonl"
authorization="$continuous_root/publication_background_plan_authorization.json"
variable_root="$OUTPUT_ROOT/variable-detector"
(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  env \
    TASK_PYTHON="$TASK_PYTHON" \
    TASK_CODE_DIR="$TASK_CODE_DIR" \
    GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT" \
    SOURCE_PIPELINE_REPORT="$promoted_pipeline" \
    BACKGROUND_MANIFEST="$background_manifest" \
    EXPANDED_BACKGROUND_MERGE_REPORT="$merge_report" \
    BACKGROUND_PLAN_AUTHORIZATION="$authorization" \
    NETWORK_CONFIG="$NETWORK_CONFIG" \
    OUTPUT_ROOT="$variable_root" \
    bash scripts/run_candidate_validation_detector_set_successor.sh
)

(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT
  "$TASK_PYTHON" - "$final_receipt" \
    "$SOURCE_SAFE_CHAIN_RECEIPT" "$FIVE_SEED_SUMMARY" "$promoted_pipeline" \
    "$comparison" "$merge_report" "$authorization" \
    "$variable_root/candidate_validation_detector_set_block_pipeline_report.json" \
    "$variable_root/frozen_validation_candidate_search_calibration_endpoint_bound.json" <<'PY'
import json
import pathlib
import sys

from gwyolo.io import atomic_write_json, file_sha256
from gwyolo.runtime import execution_provenance

(
    output_arg,
    source_arg,
    summary_arg,
    promoted_arg,
    comparison_arg,
    merge_arg,
    authorization_arg,
    variable_arg,
    binding_arg,
) = sys.argv[1:]
paths = {
    name: pathlib.Path(value)
    for name, value in {
        "source_safe_chain_receipt": source_arg,
        "five_seed_summary": summary_arg,
        "promoted_pipeline": promoted_arg,
        "candidate_comparison": comparison_arg,
        "continuous_background_merge": merge_arg,
        "background_authorization": authorization_arg,
        "variable_detector_pipeline": variable_arg,
        "endpoint_binding": binding_arg,
    }.items()
}
if any(not path.is_file() for path in paths.values()):
    raise SystemExit("post-five-seed candidate chain omitted a final artifact")
values = {
    name: json.loads(path.read_text(encoding="utf-8"))
    for name, path in paths.items()
}
if (
    values["candidate_comparison"].get("passed") is not True
    or values["candidate_comparison"].get("scale_continuous_background") is not True
    or values["continuous_background_merge"].get("status")
    != "verified_merged_streamed_candidate_background"
    or values["continuous_background_merge"].get("complete_parent_plan") is not True
    or int(values["continuous_background_merge"].get("split_counts", {}).get("test", -1))
    != 0
    or values["variable_detector_pipeline"].get("status")
    != "validation_only_clustered_candidate_search_pipeline"
    or values["variable_detector_pipeline"].get("frozen_search", {}).get(
        "publication_calibration_eligible"
    )
    is not True
    or values["endpoint_binding"].get("status")
    != "frozen_validation_candidate_search_calibration_endpoint_bound"
    or values["endpoint_binding"].get("passed") is not True
):
    raise SystemExit("post-five-seed candidate final gate failed")
result = {
    "status": "completed_post_five_seed_candidate_publication_chain",
    "execution_passed": True,
    "continuous_background_started": True,
    "continuous_background_completed": True,
    "variable_detector_calibration_frozen": True,
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "artifacts": {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for name, path in paths.items()
    },
    **execution_provenance(),
}
atomic_write_json(output_arg, result)
PY
)
