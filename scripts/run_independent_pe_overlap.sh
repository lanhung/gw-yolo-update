#!/usr/bin/env bash
set -euo pipefail

# Build a validation-only chirp+real-glitch overlap corpus against the frozen,
# GPS- and purpose-disjoint injection endpoint.  The historical overlap
# validation manifest remains model-selection evidence; this output is a new
# paired-PE evaluation input and is never used to select the checkpoint.

required=(
  TASK_PYTHON
  TASK_CODE_DIR
  GWYOLO_CODE_COMMIT
  INDEPENDENT_VALIDATION_ENDPOINT_REPORT
  VALIDATION_GLITCH_MANIFEST
  GRAVITYSPY_CORPUS_AUDIT
  TRAIN_OVERLAP_MANIFEST
  MATERIALIZATION_CONFIG
  OUTPUT_ROOT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required environment variable is unset: $variable" >&2
    exit 2
  fi
done

SEED=${SEED:-20260726}
MINIMUM_OVERLAP_ROWS=${MINIMUM_OVERLAP_ROWS:-100}
if ! [[ "$SEED" =~ ^[0-9]+$ ]] || ! [[ "$MINIMUM_OVERLAP_ROWS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED and MINIMUM_OVERLAP_ROWS must be non-negative/positive integers" >&2
  exit 2
fi
if [[ ! -d "$TASK_CODE_DIR/src/gwyolo" ]]; then
  echo "TASK_CODE_DIR is not a GW-YOLO checkout: $TASK_CODE_DIR" >&2
  exit 2
fi
for path in \
  "$TASK_PYTHON" \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$VALIDATION_GLITCH_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$TRAIN_OVERLAP_MANIFEST" \
  "$MATERIALIZATION_CONFIG"; do
  if [[ ! -s "$path" ]]; then
    echo "required independent PE overlap input is absent: $path" >&2
    exit 2
  fi
done

if ! preflight_output=$(
  "$TASK_PYTHON" - \
    "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
    "$VALIDATION_GLITCH_MANIFEST" \
    "$GRAVITYSPY_CORPUS_AUDIT" \
    "$TRAIN_OVERLAP_MANIFEST" \
    "$MINIMUM_OVERLAP_ROWS" <<'PY'
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"manifest is empty: {path}")
    return rows


endpoint_path, glitch_path, corpus_audit_path, train_overlap_path, minimum = sys.argv[1:]
endpoint = json.loads(pathlib.Path(endpoint_path).read_text(encoding="utf-8"))
components = endpoint.get("component_reports", {})
expected_components = {
    "purpose_partition",
    "injection_plan",
    "waveform_validation",
    "materialization",
    "snr_annotation",
    "arrival_annotation",
}
if (
    endpoint.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or endpoint.get("passed") is not True
    or endpoint.get("test_rows_read") != 0
    or endpoint.get("test_evaluation") is not None
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or set(components) != expected_components
    or any(digest(item["path"]) != item["sha256"] for item in components.values())
):
    raise SystemExit("independent validation endpoint failed complete hash replay")

injection_path = pathlib.Path(endpoint["injection_arrival_manifest_path"])
if (
    not injection_path.is_file()
    or digest(injection_path) != endpoint.get("injection_arrival_manifest_sha256")
):
    raise SystemExit("endpoint injection-arrival manifest is absent or hash-invalid")
injections = read_jsonl(injection_path)
if (
    len(injections) != int(endpoint.get("rows", -1))
    or any(row.get("split") != "val" for row in injections)
    or len({str(row["injection_id"]) for row in injections}) != len(injections)
    or len({str(row["waveform_id"]) for row in injections}) != len(injections)
):
    raise SystemExit("endpoint injections are not a unique validation-only population")

corpus = json.loads(pathlib.Path(corpus_audit_path).read_text(encoding="utf-8"))
cross_split = corpus.get("split_audit", {}).get("cross_split_overlaps", {})
if (
    corpus.get("status") != "verified_group_safe_gravityspy_aligned_network_corpus"
    or corpus.get("passed") is not True
    or corpus.get("validation_manifest_sha256") != digest(glitch_path)
    or not cross_split
    or any(cross_split.values())
):
    raise SystemExit("Gravity Spy validation manifest lacks a zero-overlap corpus certificate")

glitches = read_jsonl(glitch_path)
if (
    any(row.get("split") != "val" for row in glitches)
    or len({str(row["glitch_id"]) for row in glitches}) != len(glitches)
):
    raise SystemExit("Gravity Spy input is not a unique validation-only glitch population")

train_overlaps = read_jsonl(train_overlap_path)
corpus_sha = digest(corpus_audit_path)
if (
    any(row.get("split") != "train" for row in train_overlaps)
    or any(row.get("gravityspy_corpus_audit_sha256") != corpus_sha for row in train_overlaps)
):
    raise SystemExit("training overlap manifest is not bound to the same source-safe corpus audit")

supported_sets = []
for row in injections:
    ifos = row.get("ifos") or list((row.get("optimal_snr_by_ifo") or {}).keys())
    if not ifos:
        raise SystemExit("endpoint injection does not declare detector availability")
    supported_sets.append(frozenset(str(ifo) for ifo in ifos))

pairable = []
excluded = []
for row in glitches:
    required = frozenset(str(ifo) for ifo in row.get("available_ifos", [row["ifo"]]))
    if any(required <= supported for supported in supported_sets):
        pairable.append(row)
    else:
        excluded.append(row)
pair_count = min(len(pairable), len(injections))
if pair_count < int(minimum):
    raise SystemExit(
        f"only {pair_count} detector-compatible validation overlaps; minimum is {minimum}"
    )

train_fields = {
    field: {str(row[field]) for row in train_overlaps}
    for field in (
        "injection_id",
        "waveform_id",
        "glitch_id",
        "injection_gps_block",
        "network_gps_block",
    )
}
endpoint_fields = {
    "injection_id": {str(row["injection_id"]) for row in injections},
    "waveform_id": {str(row["waveform_id"]) for row in injections},
    "injection_gps_block": {str(row["gps_block"]) for row in injections},
}
for field, values in endpoint_fields.items():
    if train_fields[field] & values:
        raise SystemExit(f"independent PE endpoint overlaps training field {field}")
if train_fields["glitch_id"] & {str(row["glitch_id"]) for row in glitches}:
    raise SystemExit("validation glitch IDs overlap the overlap-training corpus")
if train_fields["network_gps_block"] & {
    str(row["network_gps_block"]) for row in glitches
}:
    raise SystemExit("validation glitch GPS blocks overlap the overlap-training corpus")

print(injection_path.resolve())
print(pair_count)
print(len(excluded))
PY
); then
  echo "independent PE overlap preflight failed" >&2
  exit 2
fi
readarray -t preflight <<<"$preflight_output"
if (( ${#preflight[@]} != 3 )); then
  echo "independent PE overlap preflight returned an invalid result" >&2
  exit 2
fi
injection_manifest=${preflight[0]}
pairable_rows=${preflight[1]}
excluded_rows=${preflight[2]}

mkdir -p "$OUTPUT_ROOT"
overlap_dir="$OUTPUT_ROOT/validation-overlap"
overlap_report="$overlap_dir/physical_overlap_report.json"
overlap_manifest="$overlap_dir/physical_overlap_val_manifest.jsonl"
if [[ ! -s "$overlap_report" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli physical-overlap-materialize \
      --gravityspy-manifest "$VALIDATION_GLITCH_MANIFEST" \
      --injection-manifest "$injection_manifest" \
      --config "$MATERIALIZATION_CONFIG" \
      --output-dir "$overlap_dir" \
      --split val \
      --seed "$SEED" \
      --limit "$pairable_rows" \
      --gravityspy-corpus-audit "$GRAVITYSPY_CORPUS_AUDIT"
  )
fi

joint_audit="$OUTPUT_ROOT/physical_overlap_audit.json"
if [[ ! -s "$joint_audit" ]]; then
  (
    cd "$TASK_CODE_DIR"
    export PYTHONPATH=src GWYOLO_CODE_COMMIT="$GWYOLO_CODE_COMMIT"
    "$TASK_PYTHON" -m gwyolo.cli physical-overlap-audit \
      --manifest "$TRAIN_OVERLAP_MANIFEST" \
      --manifest "$overlap_manifest" \
      --output "$joint_audit"
  )
fi

final_report="$OUTPUT_ROOT/independent_pe_overlap_report.json"
"$TASK_PYTHON" - \
  "$INDEPENDENT_VALIDATION_ENDPOINT_REPORT" \
  "$VALIDATION_GLITCH_MANIFEST" \
  "$GRAVITYSPY_CORPUS_AUDIT" \
  "$TRAIN_OVERLAP_MANIFEST" \
  "$MATERIALIZATION_CONFIG" \
  "$overlap_report" \
  "$overlap_manifest" \
  "$joint_audit" \
  "$MINIMUM_OVERLAP_ROWS" \
  "$pairable_rows" \
  "$excluded_rows" \
  "$SEED" \
  "$GWYOLO_CODE_COMMIT" \
  "$final_report" <<'PY'
import collections
import hashlib
import json
import pathlib
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


(
    endpoint_path,
    glitch_path,
    corpus_audit_path,
    train_overlap_path,
    config_path,
    overlap_report_path,
    overlap_manifest_path,
    joint_audit_path,
    minimum,
    pairable,
    excluded,
    seed,
    code_commit,
    final_path,
) = sys.argv[1:]

endpoint = json.loads(pathlib.Path(endpoint_path).read_text(encoding="utf-8"))
components = endpoint.get("component_reports", {})
expected_components = {
    "purpose_partition",
    "injection_plan",
    "waveform_validation",
    "materialization",
    "snr_annotation",
    "arrival_annotation",
}
if (
    endpoint.get("status") != "frozen_gps_and_purpose_disjoint_validation_endpoint"
    or endpoint.get("passed") is not True
    or endpoint.get("test_rows_read") != 0
    or endpoint.get("test_evaluation") is not None
    or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
    or set(components) != expected_components
    or any(digest(item["path"]) != item["sha256"] for item in components.values())
):
    raise SystemExit("independent endpoint changed after overlap materialization")
injection_path = endpoint["injection_arrival_manifest_path"]
injections = {row["injection_id"]: row for row in read_jsonl(injection_path)}
rows = read_jsonl(overlap_manifest_path)
report = json.loads(pathlib.Path(overlap_report_path).read_text(encoding="utf-8"))
corpus_sha = digest(corpus_audit_path)
unique_counts = report.get("unique_physical_counts", {})
if (
    report.get("status") != "verified_real_glitch_physical_overlap_training_data"
    or report.get("scientific_claim_allowed") is not False
    or report.get("search_claim_allowed") is not False
    or report.get("split") != "val"
    or int(report.get("rows", -1)) != int(pairable)
    or int(report.get("rows", -1)) < int(minimum)
    or report.get("manifest_sha256") != digest(overlap_manifest_path)
    or pathlib.Path(report.get("manifest_path", "")).resolve()
    != pathlib.Path(overlap_manifest_path).resolve()
    or report.get("gravityspy_manifest_sha256") != digest(glitch_path)
    or report.get("injection_manifest_sha256") != digest(injection_path)
    or report.get("config_sha256") != digest(config_path)
    or report.get("gravityspy_corpus_audit_sha256") != corpus_sha
    or int(report.get("rendered_image_count", -1)) != 0
    or any(
        int(unique_counts.get(field, -1)) != int(pairable)
        for field in ("mixtures", "injections", "waveforms", "glitches")
    )
    or int(report.get("aligned_network_rows", -1))
    + int(report.get("single_ifo_rows", -1))
    != int(pairable)
    or int(report.get("weak_masks", -1)) + int(report.get("human_pixel_masks", -1))
    != int(pairable)
    or report.get("code_commit") != code_commit
):
    raise SystemExit("materialized independent PE overlap report failed its identity gate")
if (
    len(rows) != int(pairable)
    or any(row.get("split") != "val" for row in rows)
    or len({row["mixture_id"] for row in rows}) != len(rows)
    or len({row["injection_id"] for row in rows}) != len(rows)
    or len({row["waveform_id"] for row in rows}) != len(rows)
    or len({row["glitch_id"] for row in rows}) != len(rows)
    or any(row.get("gravityspy_corpus_audit_sha256") != corpus_sha for row in rows)
):
    raise SystemExit("independent PE overlap manifest is not unique and validation-only")
for row in rows:
    source = injections.get(row["injection_id"])
    if source is None:
        raise SystemExit("overlap injection is absent from the frozen endpoint")
    if (
        row.get("waveform_id") != source.get("waveform_id")
        or row.get("injection_materialized_sha256")
        != source.get("materialized_sha256")
        or row.get("injection_gps_block") != source.get("gps_block")
    ):
        raise SystemExit("overlap row differs from its frozen endpoint injection")

audit = json.loads(pathlib.Path(joint_audit_path).read_text(encoding="utf-8"))
cross = audit.get("cross_split_overlaps", {})
if (
    audit.get("status") != "passed_physical_overlap_group_audit"
    or audit.get("passed") is not True
    or set(audit.get("manifest_sha256_by_split", {})) != {"train", "val"}
    or audit["manifest_sha256_by_split"]["train"] != digest(train_overlap_path)
    or audit["manifest_sha256_by_split"]["val"] != digest(overlap_manifest_path)
    or audit.get("rows_by_split", {}).get("val") != len(rows)
    or not cross
    or any(values for pair in cross.values() for values in pair.values())
):
    raise SystemExit("joint train/independent-validation overlap audit did not pass")

detector_subsets = collections.Counter("".join(row["available_ifos"]) for row in rows)
result = {
    "status": "verified_independent_validation_pe_overlap",
    "passed": True,
    "scientific_claim_allowed": False,
    "scientific_blocker": "validation-only PE smoke input; formal paired posteriors and locked test remain required",
    "test_rows_read": 0,
    "test_evaluation": None,
    "rows": len(rows),
    "minimum_overlap_rows": int(minimum),
    "excluded_detector_incompatible_glitch_rows": int(excluded),
    "detector_subset_counts": dict(sorted(detector_subsets.items())),
    "independent_validation_endpoint_report_path": str(pathlib.Path(endpoint_path).resolve()),
    "independent_validation_endpoint_report_sha256": digest(endpoint_path),
    "endpoint_component_reports": components,
    "injection_arrival_manifest_path": str(pathlib.Path(injection_path).resolve()),
    "injection_arrival_manifest_sha256": digest(injection_path),
    "validation_glitch_manifest_path": str(pathlib.Path(glitch_path).resolve()),
    "validation_glitch_manifest_sha256": digest(glitch_path),
    "gravityspy_corpus_audit_path": str(pathlib.Path(corpus_audit_path).resolve()),
    "gravityspy_corpus_audit_sha256": corpus_sha,
    "training_overlap_manifest_path": str(pathlib.Path(train_overlap_path).resolve()),
    "training_overlap_manifest_sha256": digest(train_overlap_path),
    "materialization_config_path": str(pathlib.Path(config_path).resolve()),
    "materialization_config_sha256": digest(config_path),
    "overlap_report_path": str(pathlib.Path(overlap_report_path).resolve()),
    "overlap_report_sha256": digest(overlap_report_path),
    "overlap_manifest_path": str(pathlib.Path(overlap_manifest_path).resolve()),
    "overlap_manifest_sha256": digest(overlap_manifest_path),
    "joint_overlap_audit_path": str(pathlib.Path(joint_audit_path).resolve()),
    "joint_overlap_audit_sha256": digest(joint_audit_path),
    "seed": int(seed),
    "code_commit": code_commit,
}
target = pathlib.Path(final_path)
if target.exists():
    existing = json.loads(target.read_text(encoding="utf-8"))
    if existing != result:
        raise SystemExit("existing independent PE overlap receipt has a different identity")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
print(json.dumps(result, indent=2, sort_keys=True))
PY
