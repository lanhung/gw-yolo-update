#!/usr/bin/env bash
set -euo pipefail

required=(
  TASK_PYTHON AMPLFI_PYTHON AMPLFI_CLI TASK_CODE_DIR GWYOLO_CODE_COMMIT
  BACKGROUND_RECEIPT BACKGROUND_DATA_DIR BACKGROUND_BANK_REPORT OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done
for path in \
  "$TASK_PYTHON" \
  "$AMPLFI_PYTHON" \
  "$AMPLFI_CLI" \
  "$BACKGROUND_RECEIPT" \
  "$BACKGROUND_BANK_REPORT"; do
  if [[ ! -s "$path" ]]; then
    echo "required AMPLFI stage-1 input is absent: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" || ! -d "$BACKGROUND_DATA_DIR/train/background" || ! -d "$BACKGROUND_DATA_DIR/validation/background" ]]; then
  echo "AMPLFI code or group-safe background directories are absent" >&2
  exit 2
fi

EXPECTED_AMPLFI_VERSION=${EXPECTED_AMPLFI_VERSION:-0.6.0}
observed_version=$(
  "$AMPLFI_PYTHON" -c 'import importlib.metadata; print(importlib.metadata.version("amplfi"))'
)
if [[ "$observed_version" != "$EXPECTED_AMPLFI_VERSION" ]]; then
  echo "AMPLFI runtime version mismatch: $observed_version != $EXPECTED_AMPLFI_VERSION" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
cd "$TASK_CODE_DIR"
export PYTHONPATH="$TASK_CODE_DIR/src"
export AMPLFI_DATADIR="$BACKGROUND_DATA_DIR"
export AMPLFI_OUTDIR="$OUTPUT_ROOT/training"
export GWYOLO_AMPLFI_TRAINING_PRIOR="$TASK_CODE_DIR/configs/amplfi_common_bbh_training_prior.yaml"

"$TASK_PYTHON" - \
  "$BACKGROUND_RECEIPT" "$BACKGROUND_DATA_DIR" "$BACKGROUND_BANK_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


receipt_path, bank_value, bank_report_value = sys.argv[1:]
bank = pathlib.Path(bank_value).resolve()
receipt = json.loads(pathlib.Path(receipt_path).read_text(encoding="utf-8"))
merge_path = pathlib.Path(receipt.get("stream_merge_report_path", ""))
capacity_path = pathlib.Path(receipt.get("capacity_report_path", ""))
receipt_status = receipt.get("status")
status_to_merge = {
    "verified_capacity_ready_amplfi_training_background": (
        "verified_streamed_amplfi_background_bank"
    ),
    "verified_capacity_ready_amplfi_background_extension": (
        "verified_extended_streamed_amplfi_background_bank"
    ),
}
if (
    receipt_status not in status_to_merge
    or receipt.get("passed") is not True
    or receipt.get("scientific_claim_allowed") is not False
    or receipt.get("test_rows_read") != 0
    or not merge_path.is_file()
    or digest(merge_path) != receipt.get("stream_merge_report_sha256")
    or not capacity_path.is_file()
    or digest(capacity_path) != receipt.get("capacity_report_sha256")
):
    raise SystemExit("AMPLFI training background receipt failed replay")
merge = json.loads(merge_path.read_text(encoding="utf-8"))
capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
if (
    merge.get("status") != status_to_merge[receipt_status]
    or merge.get("passed") is not True
    or merge.get("test_rows_exported") != 0
    or capacity.get("status") != "amplfi_background_capacity_ready"
    or capacity.get("passed") is not True
    or capacity.get("test_strain_rows_read") != 0
    or capacity.get("manifest_sha256") != merge.get("background_manifest_sha256")
):
    raise SystemExit("AMPLFI background merge/capacity identity failed")
bank_report_path = pathlib.Path(bank_report_value).resolve()
bank_report = json.loads(bank_report_path.read_text(encoding="utf-8"))
if (
    bank_report.get("status") != "frozen_hash_bound_amplfi_training_bank"
    or bank_report.get("passed") is not True
    or bank_report.get("test_rows_read") != 0
    or bank_report.get("test_files_linked") != 0
    or pathlib.Path(bank_report.get("background_receipt_path", "")).resolve()
    != pathlib.Path(receipt_path).resolve()
    or bank_report.get("background_receipt_sha256") != digest(receipt_path)
    or pathlib.Path(bank_report.get("bank_root", "")).resolve() != bank
    or bank_report.get("background_manifest_sha256")
    != merge.get("background_manifest_sha256")
):
    raise SystemExit("AMPLFI frozen training bank failed receipt replay")
files = bank_report.get("files", [])
if not files:
    raise SystemExit("AMPLFI frozen training bank is empty")
for row in files:
    relative = pathlib.Path(row["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("AMPLFI frozen bank contains an unsafe relative path")
    path = bank / relative
    resolved = path.resolve()
    if (
        row.get("split") not in {"train", "val"}
        or not path.is_file()
        or digest(resolved) != row["source_sha256"]
    ):
        raise SystemExit("AMPLFI frozen training file changed")
PY

resolved_config="$OUTPUT_ROOT/amplfi_publication_stage1.yaml"
stage_report="$OUTPUT_ROOT/amplfi_publication_stage1_config.json"
if [[ ! -s "$stage_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli amplfi-training-stage-freeze \
    --base-config configs/amplfi_common_bbh_publication.yaml \
    --stage-policy configs/amplfi_training_stage_policy.yaml \
    --stage publication_stage_1 \
    --output-config "$resolved_config" \
    --output-report "$stage_report"
fi

prior_report="$OUTPUT_ROOT/amplfi_prior_projection.json"
if [[ ! -s "$prior_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli amplfi-common-prior-audit \
    --canonical-prior configs/pe_common_bbh_analysis_prior.yaml \
    --amplfi-prior configs/amplfi_common_bbh_training_prior.yaml \
    --training-config "$resolved_config" \
    --output "$prior_report"
fi

training_marker="$OUTPUT_ROOT/training_complete.json"
if [[ ! -s "$training_marker" ]]; then
  while :; do
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | sed '/^[[:space:]]*$/d' || true)
    [[ -z "$gpu_pids" ]] && break
    sleep 30
  done
  resume_args=()
  mapfile -t last_checkpoints < <(find "$AMPLFI_OUTDIR" -type f -name last.ckpt 2>/dev/null | sort)
  if (( ${#last_checkpoints[@]} > 1 )); then
    echo "multiple AMPLFI last.ckpt files make resumption ambiguous" >&2
    exit 2
  elif (( ${#last_checkpoints[@]} == 1 )); then
    resume_args+=(--ckpt_path "${last_checkpoints[0]}")
  fi
  "$AMPLFI_CLI" fit --config "$resolved_config" "${resume_args[@]}"
  "$TASK_PYTHON" - "$resolved_config" "$BACKGROUND_RECEIPT" \
    "$BACKGROUND_BANK_REPORT" "$GWYOLO_CODE_COMMIT" "$training_marker" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


config, background, bank_report, commit, target_value = sys.argv[1:]
result = {
    "status": "amplfi_publication_stage1_training_process_complete",
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "resolved_config_sha256": hashlib.sha256(pathlib.Path(config).read_bytes()).hexdigest(),
    "background_receipt_sha256": hashlib.sha256(pathlib.Path(background).read_bytes()).hexdigest(),
    "background_bank_report_sha256": hashlib.sha256(
        pathlib.Path(bank_report).read_bytes()
    ).hexdigest(),
    "code_commit": commit,
}
target = pathlib.Path(target_value)
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(part, target)
PY
fi

mapfile -t metrics_files < <(find "$AMPLFI_OUTDIR" -type f -name metrics.csv | sort)
if (( ${#metrics_files[@]} != 1 )); then
  echo "AMPLFI stage-1 requires exactly one metrics.csv" >&2
  exit 2
fi
checkpoint_index="$OUTPUT_ROOT/amplfi_checkpoint_index.json"
if [[ ! -s "$checkpoint_index" ]]; then
  "$AMPLFI_PYTHON" scripts/index_lightning_checkpoints.py \
    --checkpoint-dir "$AMPLFI_OUTDIR" \
    --output "$checkpoint_index"
fi
selection_report="$OUTPUT_ROOT/amplfi_validation_selection.json"
if [[ ! -s "$selection_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli pe-lightning-checkpoint-select \
    --training-config "$resolved_config" \
    --training-data-manifest "$BACKGROUND_BANK_REPORT" \
    --metrics-csv "${metrics_files[0]}" \
    --checkpoint-index "$checkpoint_index" \
    --output "$selection_report" \
    --selection-metric valid_loss \
    --selection-metric-mode min \
    --minimum-publication-epochs 100 \
    --minimum-validation-points 50
fi
checkpoint=$(
  "$TASK_PYTHON" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["publication_eligible"] is True; print(d["selected_checkpoint_path"])' \
    "$selection_report"
)
checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
config_sha256=$(sha256sum "$resolved_config" | awk '{print $1}')
load_report="$OUTPUT_ROOT/amplfi_model_load_report.json"
if [[ ! -s "$load_report" ]]; then
  "$AMPLFI_PYTHON" scripts/run_pe_model_load_smoke.py \
    --backend AMPLFI \
    --model "$checkpoint" \
    --expected-model-sha256 "$checkpoint_sha256" \
    --model-config "$resolved_config" \
    --expected-model-config-sha256 "$config_sha256" \
    --output "$load_report" \
    --device cuda
fi

metadata="$OUTPUT_ROOT/amplfi_model_metadata.json"
if [[ ! -s "$metadata" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli pe-backend-model-freeze \
    --backend AMPLFI \
    --model "$checkpoint" \
    --training-config "$resolved_config" \
    --training-data-manifest "$BACKGROUND_BANK_REPORT" \
    --analysis-prior configs/pe_common_bbh_analysis_prior.yaml \
    --native-prior configs/amplfi_common_bbh_training_prior.yaml \
    --prior-projection-report "$prior_report" \
    --selection-report "$selection_report" \
    --native-conditioning-config configs/amplfi_common_native_conditioning.yaml \
    --source-sample-rate-hz 4096 \
    --source-duration-seconds 16 \
    --source-post-trigger-seconds 2 \
    --analysis-waveform-approximant IMRPhenomXPHM \
    --native-model-waveform-approximant ml4gw.waveforms.IMRPhenomPv2 \
    --model-training-backend-version 0.6.0 \
    --native-inference-parameters chirp_mass mass_ratio distance phic inclination dec psi phi \
    --reported-parameter-mapping chirp_mass=chirp_mass mass_ratio=mass_ratio \
      luminosity_distance=distance theta_jn=inclination ra=phi dec=dec psi=psi \
    --output "$metadata"
fi

"$TASK_PYTHON" - "$stage_report" "$BACKGROUND_RECEIPT" "$BACKGROUND_BANK_REPORT" \
  "$selection_report" \
  "$load_report" "$metadata" "$GWYOLO_CODE_COMMIT" \
  "$OUTPUT_ROOT/amplfi_publication_stage1_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    stage_path,
    background_path,
    bank_report_path,
    selection_path,
    load_path,
    metadata_path,
    commit,
    target_value,
) = sys.argv[1:]
stage = json.loads(pathlib.Path(stage_path).read_text(encoding="utf-8"))
selection = json.loads(pathlib.Path(selection_path).read_text(encoding="utf-8"))
load = json.loads(pathlib.Path(load_path).read_text(encoding="utf-8"))
metadata = json.loads(pathlib.Path(metadata_path).read_text(encoding="utf-8"))
if (
    stage.get("stage") != "publication_stage_1"
    or stage.get("compute_budget", {}).get("updates") != 20000
    or stage.get("compute_budget", {}).get("online_waveform_examples") != 5120000
    or selection.get("publication_eligible") is not True
    or selection.get("selection_split") != "validation"
    or load.get("status") != "real_pe_backend_model_load_smoke_complete"
    or load.get("backend") != "AMPLFI"
    or metadata.get("backend") != "AMPLFI"
    or metadata.get("selection_split") != "validation"
):
    raise SystemExit("AMPLFI stage-1 training did not pass selection/load/metadata gates")
result = {
    "status": "verified_amplfi_publication_stage1_model",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": "paired posterior robustness validation remains required",
    "test_rows_read": 0,
    "stage_config_report_sha256": digest(stage_path),
    "background_receipt_sha256": digest(background_path),
    "background_bank_report_path": str(pathlib.Path(bank_report_path).resolve()),
    "background_bank_report_sha256": digest(bank_report_path),
    "selection_report_sha256": digest(selection_path),
    "model_load_report_sha256": digest(load_path),
    "model_metadata_path": str(pathlib.Path(metadata_path).resolve()),
    "model_metadata_sha256": digest(metadata_path),
    "selected_checkpoint_path": selection["selected_checkpoint_path"],
    "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
    "code_commit": commit,
}
target = pathlib.Path(target_value)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing AMPLFI stage-1 receipt has another identity")
else:
    part = target.with_suffix(target.suffix + ".part")
    part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
