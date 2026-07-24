#!/usr/bin/env bash
set -euo pipefail

# Exercise the real group-safe AMPLFI bank and model-loading path on one
# explicitly assigned GPU. This stage is an engineering gate, never paper evidence.

required=(
  TASK_PYTHON AMPLFI_PYTHON AMPLFI_CLI TASK_CODE_DIR GWYOLO_CODE_COMMIT
  BACKGROUND_RECEIPT BACKGROUND_DATA_DIR BACKGROUND_BANK_REPORT OUTPUT_ROOT
  CUDA_VISIBLE_DEVICES
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required AMPLFI engineering-smoke variable is unset: $variable" >&2
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
    echo "required AMPLFI engineering-smoke input is absent: $path" >&2
    exit 2
  fi
done
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "AMPLFI engineering smoke requires exactly one visible GPU" >&2
  exit 2
fi
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" \
  || ! -d "$BACKGROUND_DATA_DIR/train/background" \
  || ! -d "$BACKGROUND_DATA_DIR/validation/background" ]]; then
  echo "AMPLFI engineering-smoke code or bank directories are absent" >&2
  exit 2
fi
if [[ "$(git -C "$TASK_CODE_DIR" rev-parse HEAD 2>/dev/null || true)" \
  != "$GWYOLO_CODE_COMMIT" ]]; then
  echo "AMPLFI engineering smoke requires its exact checkout" >&2
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
"$AMPLFI_PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("AMPLFI engineering smoke requires exactly one usable CUDA device")
print(torch.cuda.get_device_name(0))
PY

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
status_to_merge = {
    "verified_capacity_ready_amplfi_training_background": (
        "verified_streamed_amplfi_background_bank"
    ),
    "verified_capacity_ready_amplfi_background_extension": (
        "verified_extended_streamed_amplfi_background_bank"
    ),
}
if (
    receipt.get("status") not in status_to_merge
    or receipt.get("passed") is not True
    or receipt.get("scientific_claim_allowed") is not False
    or receipt.get("test_rows_read") != 0
    or not merge_path.is_file()
    or digest(merge_path) != receipt.get("stream_merge_report_sha256")
    or not capacity_path.is_file()
    or digest(capacity_path) != receipt.get("capacity_report_sha256")
):
    raise SystemExit("AMPLFI engineering-smoke background receipt failed replay")
merge = json.loads(merge_path.read_text(encoding="utf-8"))
capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
if (
    merge.get("status") != status_to_merge[receipt["status"]]
    or merge.get("passed") is not True
    or merge.get("test_rows_exported") != 0
    or capacity.get("status") != "amplfi_background_capacity_ready"
    or capacity.get("passed") is not True
    or capacity.get("test_strain_rows_read") != 0
    or capacity.get("manifest_sha256") != merge.get("background_manifest_sha256")
):
    raise SystemExit("AMPLFI engineering-smoke bank identity failed")
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
    raise SystemExit("AMPLFI engineering-smoke frozen bank failed replay")
files = bank_report.get("files", [])
if not files:
    raise SystemExit("AMPLFI engineering-smoke bank is empty")
for row in files:
    relative = pathlib.Path(row["relative_path"])
    path = bank / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or row.get("split") not in {"train", "val"}
        or not path.is_file()
        or digest(path.resolve()) != row["source_sha256"]
    ):
        raise SystemExit("AMPLFI engineering-smoke bank file changed")
PY

resolved_config="$OUTPUT_ROOT/amplfi_engineering_smoke.yaml"
stage_report="$OUTPUT_ROOT/amplfi_engineering_smoke_config.json"
if [[ ! -s "$stage_report" ]]; then
  "$TASK_PYTHON" -m gwyolo.cli amplfi-training-stage-freeze \
    --base-config configs/amplfi_common_bbh_publication.yaml \
    --stage-policy configs/amplfi_training_stage_policy.yaml \
    --stage engineering_smoke \
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
  resume_args=()
  mapfile -t last_checkpoints < <(
    find "$AMPLFI_OUTDIR" -type f -name last.ckpt 2>/dev/null | sort
  )
  if (( ${#last_checkpoints[@]} > 1 )); then
    echo "multiple AMPLFI smoke last.ckpt files make resumption ambiguous" >&2
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
    "status": "amplfi_engineering_smoke_training_process_complete",
    "scientific_claim_allowed": False,
    "test_rows_read": 0,
    "resolved_config_sha256": hashlib.sha256(
        pathlib.Path(config).read_bytes()
    ).hexdigest(),
    "background_receipt_sha256": hashlib.sha256(
        pathlib.Path(background).read_bytes()
    ).hexdigest(),
    "background_bank_report_sha256": hashlib.sha256(
        pathlib.Path(bank_report).read_bytes()
    ).hexdigest(),
    "code_commit": commit,
}
target = pathlib.Path(target_value)
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, target)
PY
fi

mapfile -t metrics_files < <(find "$AMPLFI_OUTDIR" -type f -name metrics.csv | sort)
mapfile -t last_checkpoints < <(
  find "$AMPLFI_OUTDIR" -type f -name last.ckpt | sort
)
if (( ${#metrics_files[@]} != 1 || ${#last_checkpoints[@]} != 1 )); then
  echo "AMPLFI engineering smoke requires one metrics file and one last checkpoint" >&2
  exit 2
fi
checkpoint="${last_checkpoints[0]}"
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

"$TASK_PYTHON" - \
  "$stage_report" "$resolved_config" "$BACKGROUND_RECEIPT" "$BACKGROUND_BANK_REPORT" \
  "${metrics_files[0]}" "$checkpoint" "$load_report" \
  "$GWYOLO_CODE_COMMIT" "$OUTPUT_ROOT/amplfi_engineering_smoke_receipt.json" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

import yaml


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


(
    stage_path,
    resolved_config_path,
    background_path,
    bank_report_path,
    metrics_path,
    checkpoint_path,
    load_path,
    commit,
    target_value,
) = sys.argv[1:]
stage = json.loads(pathlib.Path(stage_path).read_text(encoding="utf-8"))
resolved_config = yaml.safe_load(
    pathlib.Path(resolved_config_path).read_text(encoding="utf-8")
)
load = json.loads(pathlib.Path(load_path).read_text(encoding="utf-8"))
if (
    stage.get("stage") != "engineering_smoke"
    or stage.get("publication_candidate") is not False
    or stage.get("compute_budget", {}).get("epochs") != 5
    or stage.get("compute_budget", {}).get("updates") != 250
    or stage.get("compute_budget", {}).get("online_waveform_examples") != 32000
    or resolved_config.get("trainer", {}).get("deterministic") != "warn"
    or load.get("status") != "real_pe_backend_model_load_smoke_complete"
    or load.get("backend") != "AMPLFI"
):
    raise SystemExit("AMPLFI engineering smoke did not pass its frozen gates")
result = {
    "status": "verified_amplfi_engineering_smoke",
    "passed": True,
    "publication_candidate": False,
    "deterministic_policy": "seeded_warn_on_unsupported_cuda_operations",
    "scientific_claim_allowed": False,
    "search_claim_allowed": False,
    "test_rows_read": 0,
    "stage_config_report_sha256": digest(stage_path),
    "background_receipt_sha256": digest(background_path),
    "background_bank_report_sha256": digest(bank_report_path),
    "metrics_path": str(pathlib.Path(metrics_path).resolve()),
    "metrics_sha256": digest(metrics_path),
    "last_checkpoint_path": str(pathlib.Path(checkpoint_path).resolve()),
    "last_checkpoint_sha256": digest(checkpoint_path),
    "model_load_report_sha256": digest(load_path),
    "code_commit": commit,
}
target = pathlib.Path(target_value)
part = target.with_suffix(target.suffix + ".part")
part.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(part, target)
PY
