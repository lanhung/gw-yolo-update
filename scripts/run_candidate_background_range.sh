#!/usr/bin/env bash
set -euo pipefail

required_variables=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  SCORING_CODE_DIR
  SCORING_CODE_COMMIT
  PROMOTION_REPORT
  PROMOTED_PIPELINE_REPORT
  PARENT_PLAN
  EVENT_EXCLUSIONS
  COHERENCE_CONFIG
  TIMING_CALIBRATION_REPORT
  VALIDATION_INJECTION_RANKING_REPORT
  CACHE_ROOT
  OUTPUT_ROOT
  SHARD_STOP_EXCLUSIVE
)
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

SHARD_START=${SHARD_START:-0}
BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT:-}
CAPACITY_EXTENSION_DECISION=${CAPACITY_EXTENSION_DECISION:-}
PAIRS_PER_SHARD=${PAIRS_PER_SHARD:-4}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.2}
TEST_FRACTION=${TEST_FRACTION:-0}
BACKGROUND_SEED=${BACKGROUND_SEED:-20260719}
MODEL_IFOS=${MODEL_IFOS:-"H1 L1 V1"}
Q_VALUES=${Q_VALUES:-"4 8 16"}
TARGET_SAMPLE_RATE=${TARGET_SAMPLE_RATE:-1024}
CONTEXT_DURATION=${CONTEXT_DURATION:-64}
CHIRP_THRESHOLD=${CHIRP_THRESHOLD:-0.3}
MINIMUM_BINS=${MINIMUM_BINS:-1}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-2}
MINIMUM_FREE_KB=${MINIMUM_FREE_KB:-8388608}
TARGET_FAR_PER_YEAR=${TARGET_FAR_PER_YEAR:-0.1}
ZERO_COUNT_CONFIDENCE=${ZERO_COUNT_CONFIDENCE:-0.9}
REFERENCE_IFO=${REFERENCE_IFO:-H1}
SHIFTED_IFO=${SHIFTED_IFO:-L1}
CHECKPOINT=${CHECKPOINT:-}
CONFIG=${CONFIG:-}

if ! [[ "$SHARD_START" =~ ^[0-9]+$ ]] \
  || ! [[ "$SHARD_STOP_EXCLUSIVE" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$PAIRS_PER_SHARD" =~ ^[1-9][0-9]*$ ]] \
  || (( SHARD_STOP_EXCLUSIVE <= SHARD_START )); then
  echo "publication candidate background requires a valid bounded shard range" >&2
  exit 2
fi
if (( SHARD_START > 0 )) && [[ -z "$BASE_OUTPUT_ROOT" ]]; then
  echo "a nonzero extension range requires BASE_OUTPUT_ROOT" >&2
  exit 2
fi
if (( SHARD_START > 0 )) && [[ "$BASE_OUTPUT_ROOT" == "$OUTPUT_ROOT" ]]; then
  echo "extension output must be separate from the immutable base output" >&2
  exit 2
fi
if (( SHARD_START > 0 )) && [[ -z "$CAPACITY_EXTENSION_DECISION" ]]; then
  echo "a nonzero extension range requires CAPACITY_EXTENSION_DECISION" >&2
  exit 2
fi
if [[ "$TEST_FRACTION" != "0" && "$TEST_FRACTION" != "0.0" ]]; then
  echo "validation-scale background must keep test_fraction=0" >&2
  exit 2
fi

for code_dir in "$TASK_CODE_DIR" "$SCORING_CODE_DIR"; do
  if [[ ! -d "$code_dir/src/gwyolo" ]]; then
    echo "code directory is invalid: $code_dir" >&2
    exit 2
  fi
done
if [[ -z "$CHECKPOINT" || -z "$CONFIG" ]]; then
  for variable in FIVE_SEED_SUMMARY UNIFORM_CONFIG FAMILY_BALANCED_CONFIG; do
    if [[ -z "${!variable:-}" ]]; then
      echo "checkpoint/config selection variable is unset: $variable" >&2
      exit 2
    fi
  done
  readarray -t selection < <("$TASK_PYTHON" -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "completed_five_seed_source_safe_overlap_validation":
    raise SystemExit("five-seed summary has the wrong status")
print(report["promoted_arm"])
print(report["selected_checkpoint_path"])
' "$FIVE_SEED_SUMMARY")
  arm=${selection[0]}
  CHECKPOINT=${selection[1]}
  if [[ "$arm" == uniform ]]; then
    CONFIG=$UNIFORM_CONFIG
  elif [[ "$arm" == family_balanced ]]; then
    CONFIG=$FAMILY_BALANCED_CONFIG
  else
    echo "five-seed summary selected an unknown arm: $arm" >&2
    exit 2
  fi
fi
for input in \
  "$TASK_PYTHON" \
  "$PROMOTION_REPORT" \
  "$PROMOTED_PIPELINE_REPORT" \
  "$PARENT_PLAN" \
  "$EVENT_EXCLUSIONS" \
  "$CHECKPOINT" \
  "$CONFIG" \
  "$COHERENCE_CONFIG" \
  "$TIMING_CALIBRATION_REPORT" \
  "$VALIDATION_INJECTION_RANKING_REPORT"; do
  if [[ ! -f "$input" ]]; then
    echo "required input is absent: $input" >&2
    exit 2
  fi
done
if (( SHARD_START > 0 )); then
  if [[ ! -f "$CAPACITY_EXTENSION_DECISION" ]]; then
    echo "capacity extension decision is absent: $CAPACITY_EXTENSION_DECISION" >&2
    exit 2
  fi
  "$TASK_PYTHON" - \
    "$CAPACITY_EXTENSION_DECISION" "$PARENT_PLAN" \
    "$SHARD_START" "$SHARD_STOP_EXCLUSIVE" "$PAIRS_PER_SHARD" <<'PY'
import hashlib
import json
import math
import pathlib
import sys

decision_path, plan_path, start, stop, per_shard = sys.argv[1:]
decision = json.loads(pathlib.Path(decision_path).read_text(encoding="utf-8"))
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
plan_hash = hashlib.sha256(pathlib.Path(plan_path).read_bytes()).hexdigest()
base_pairs = int(start) * int(per_shard)
extended_pairs = int(plan.get("selected_pairs", -1))
if (
    decision.get("status")
    != "frozen_score_blind_background_capacity_extension_decision"
    or decision.get("candidate_scores_inspected") is not False
    or decision.get("test_data_opened") is not False
    or decision.get("extended_plan_sha256") != plan_hash
    or int(decision.get("base_source_pairs", -1)) != base_pairs
    or int(decision.get("extended_source_pairs", -1)) != extended_pairs
    or math.ceil(extended_pairs / int(per_shard)) != int(stop)
):
    raise SystemExit("capacity extension decision does not authorize this shard range")
PY
fi

preflight=$(
  "$TASK_PYTHON" - \
    "$PROMOTION_REPORT" \
    "$PROMOTED_PIPELINE_REPORT" \
    "$CHECKPOINT" \
    "$CONFIG" \
    "$COHERENCE_CONFIG" \
    "$TIMING_CALIBRATION_REPORT" \
    "$VALIDATION_INJECTION_RANKING_REPORT" \
    "$SCORING_CODE_COMMIT" \
    "$REFERENCE_IFO" \
    "$SHIFTED_IFO" \
    "$MODEL_IFOS" \
    "$Q_VALUES" \
    "$TARGET_SAMPLE_RATE" \
    "$CONTEXT_DURATION" \
    "$CHIRP_THRESHOLD" \
    "$MINIMUM_BINS" <<'PY'
import hashlib
import json
import math
import pathlib
import sys


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


(
    promotion_path,
    pipeline_path,
    checkpoint,
    config,
    coherence,
    timing,
    injection_ranking,
    scoring_commit,
    reference_ifo,
    shifted_ifo,
    model_ifos,
    q_values,
    target_sample_rate,
    context_duration,
    chirp_threshold,
    minimum_bins,
) = sys.argv[1:]
promotion = json.loads(pathlib.Path(promotion_path).read_text(encoding="utf-8"))
if (
    promotion.get("status") != "paired_validation_candidate_search_promotion"
    or promotion.get("test_data_opened") is not False
):
    raise SystemExit("candidate promotion report has the wrong contract")
if not promotion.get("passed") or not promotion.get("scale_continuous_background"):
    print("SKIP")
    raise SystemExit(0)
pipeline = json.loads(pathlib.Path(pipeline_path).read_text(encoding="utf-8"))
if (
    promotion.get("input_report_hashes", {}).get("promoted") != digest(pipeline_path)
    or pipeline.get("status") != "validation_only_clustered_candidate_search_pipeline"
    or pipeline.get("test_evaluation") is not None
    or pipeline.get("model_selection") is None
):
    raise SystemExit("promoted pipeline is not the report authorized by the gate")
identity = pipeline["run_identity"]
expected_hashes = {
    "checkpoint_sha256": digest(checkpoint),
    "config_sha256": digest(config),
    "coherence_config_sha256": digest(coherence),
}
if any(identity.get(field) != value for field, value in expected_hashes.items()):
    raise SystemExit("candidate streaming artifacts differ from the promoted scorer")
if identity.get("code_commit") != scoring_commit:
    raise SystemExit("candidate streaming code commit differs from validation scoring")
if identity.get("reference_ifo") != reference_ifo or identity.get("second_ifo") != shifted_ifo:
    raise SystemExit("candidate detector pair differs from validation scoring")
if (
    identity.get("model_ifos") != model_ifos.split()
    or [float(value) for value in identity.get("q_values", [])]
    != [float(value) for value in q_values.split()]
    or int(identity.get("target_sample_rate", -1)) != int(target_sample_rate)
    or not math.isclose(
        float(identity.get("context_duration", -1)), float(context_duration), abs_tol=1e-12
    )
    or not math.isclose(
        float(identity.get("chirp_threshold", -1)), float(chirp_threshold), abs_tol=1e-12
    )
    or int(identity.get("minimum_bins", -1)) != int(minimum_bins)
):
    raise SystemExit("candidate streaming representation differs from validation scoring")
if pipeline.get("timing_calibration_report_sha256") != digest(timing):
    raise SystemExit("timing calibration differs from promoted validation")
if pipeline.get("injection_ranking_report_sha256") != digest(injection_ranking):
    raise SystemExit("validation injection rankings differ from promoted validation")
physical = float(pipeline["physical_delay_limit_seconds"])
uncertainty = float(pipeline["empirical_timing_uncertainty_seconds"])
coincidence = float(pipeline["coincidence_window_seconds"])
cluster = float(identity["cluster_window_seconds"])
if (
    physical <= 0
    or uncertainty < 0
    or cluster <= 0
    or not math.isclose(coincidence, physical + 2 * uncertainty, abs_tol=1e-12)
):
    raise SystemExit("promoted timing/coincidence contract is invalid")
print("PASS", physical, uncertainty, coincidence, cluster, sep="\t")
PY
)
if [[ "$preflight" == "SKIP" ]]; then
  echo "paired validation gate did not authorize continuous-background scaling"
  exit 0
fi
IFS=$'\t' read -r decision physical_delay timing_uncertainty coincidence_window cluster_window \
  <<<"$preflight"
if [[ "$decision" != "PASS" ]]; then
  echo "candidate background preflight returned an invalid decision" >&2
  exit 2
fi

read -r -a model_ifos <<<"$MODEL_IFOS"
read -r -a q_values <<<"$Q_VALUES"
mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
reports=()
for ((shard = 0; shard < SHARD_START; shard++)); do
  report="$BASE_OUTPUT_ROOT/shard-$shard/streamed_background_shard_report.json"
  if [[ ! -s "$report" ]]; then
    echo "base streaming shard report is absent: $shard" >&2
    exit 1
  fi
  reports+=(--shard-report "$report")
done
for ((shard = SHARD_START; shard < SHARD_STOP_EXCLUSIVE; shard++)); do
  available_kb=$(df -Pk "$CACHE_ROOT" | awk 'NR == 2 {print $4}')
  if (( available_kb < MINIMUM_FREE_KB )); then
    echo "insufficient cache filesystem space before shard $shard" >&2
    exit 1
  fi
  while :; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  shard_output="$OUTPUT_ROOT/shard-$shard"
  (
    cd "$SCORING_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$SCORING_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli background-stream-shard \
      --parent-plan "$PARENT_PLAN" \
      --event-exclusions "$EVENT_EXCLUSIONS" \
      --timing-calibration-report "$TIMING_CALIBRATION_REPORT" \
      --checkpoint "$CHECKPOINT" \
      --config "$CONFIG" \
      --coherence-config "$COHERENCE_CONFIG" \
      --cache-root "$CACHE_ROOT" \
      --output-dir "$shard_output" \
      --shard-index "$shard" \
      --pairs-per-shard "$PAIRS_PER_SHARD" \
      --validation-fraction "$VALIDATION_FRACTION" \
      --test-fraction "$TEST_FRACTION" \
      --seed "$BACKGROUND_SEED" \
      --model-ifos "${model_ifos[@]}" \
      --q-values "${q_values[@]}" \
      --target-sample-rate "$TARGET_SAMPLE_RATE" \
      --context-duration "$CONTEXT_DURATION" \
      --chirp-threshold "$CHIRP_THRESHOLD" \
      --minimum-bins "$MINIMUM_BINS" \
      --download-workers "$DOWNLOAD_WORKERS"
  )
  report="$shard_output/streamed_background_shard_report.json"
  if [[ ! -s "$report" ]]; then
    echo "streaming shard completed without its immutable report: $shard" >&2
    exit 1
  fi
  reports+=(--shard-report "$report")
done

merge_dir="$OUTPUT_ROOT/merged"
merge_report="$merge_dir/streamed_background_merge_report.json"
if [[ ! -s "$merge_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli background-stream-merge \
      "${reports[@]}" \
      --parent-plan "$PARENT_PLAN" \
      --output-dir "$merge_dir"
  )
fi

background_manifest="$merge_dir/background_windows.jsonl"
candidate_manifest="$merge_dir/val_calibrated_candidates.jsonl"
schedule="$OUTPUT_ROOT/val_candidate_block_permutation_schedule.json"
block_dir="$OUTPUT_ROOT/val_candidate_block_background"
block_report="$block_dir/val_candidate_time_slide_report.json"
calibration="$OUTPUT_ROOT/frozen_validation_candidate_search_calibration.json"

"$TASK_PYTHON" - "$merge_report" "$SHARD_STOP_EXCLUSIVE" "$PARENT_PLAN" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
digest = hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
if (
    report.get("status") != "verified_merged_streamed_candidate_background"
    or not report.get("complete_parent_plan")
    or int(report.get("shard_count_merged", -1)) != int(sys.argv[2])
    or int(report.get("split_counts", {}).get("test", -1)) != 0
    or report.get("common_run_identity", {}).get("parent_plan_sha256") != digest
):
    raise SystemExit(
        "merged validation background is incomplete, has another parent, or exposes test data"
    )
PY

if [[ ! -s "$schedule" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli candidate-block-permutation-schedule-freeze \
      --background-manifest "$background_manifest" \
      --output "$schedule" \
      --split val \
      --reference-ifo "$REFERENCE_IFO" \
      --shifted-ifo "$SHIFTED_IFO" \
      --target-far-per-year "$TARGET_FAR_PER_YEAR" \
      --zero-count-confidence "$ZERO_COUNT_CONFIDENCE"
  )
fi

(
  cd "$TASK_CODE_DIR"
  export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
  "$TASK_PYTHON" -m gwyolo.cli candidate-block-permutations \
    --candidates "$candidate_manifest" \
    --background-manifest "$background_manifest" \
    --schedule "$schedule" \
    --output-dir "$block_dir" \
    --split val \
    --reference-ifo "$REFERENCE_IFO" \
    --shifted-ifo "$SHIFTED_IFO" \
    --coincidence-window-seconds "$coincidence_window" \
    --cluster-window-seconds "$cluster_window" \
    --physical-delay-limit-seconds "$physical_delay" \
    --empirical-timing-uncertainty-seconds "$timing_uncertainty"
)

if [[ ! -s "$calibration" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli candidate-search-calibrate \
      --validation-time-slide-report "$block_report" \
      --validation-injection-ranking-report "$VALIDATION_INJECTION_RANKING_REPORT" \
      --target-far-per-year "$TARGET_FAR_PER_YEAR" \
      --output "$calibration" \
      --bootstrap-replicates 10000 \
      --seed 20260720
  )
fi

"$TASK_PYTHON" - "$calibration" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "frozen_validation_candidate_search_calibration"
    or not report.get("publication_calibration_eligible")
    or not report.get("slide_schedule_audit", {}).get("passed")
):
    raise SystemExit("continuous validation background did not reach the publication FAR gate")
PY
