#!/usr/bin/env bash
set -euo pipefail

required=(
  UPSTREAM_PID
  TASK_PYTHON
  AUDIT_SCRIPT
  AUDIT_ROOT
  SOURCE_MANIFEST
  BASE_DIR
  TRANSFER_ROOT_A
  TRANSFER_ROOT_B
  EXPECTED_FILES
  EXPECTED_SOURCE_MANIFEST_SHA256
  GWYOLO_CODE_COMMIT
)
for variable in "${required[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "required variable is unset: $variable" >&2
    exit 2
  fi
done
if ! [[ "$UPSTREAM_PID" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "$EXPECTED_FILES" =~ ^[1-9][0-9]*$ ]]; then
  echo "upstream PID and expected file count must be positive integers" >&2
  exit 2
fi

while kill -0 "$UPSTREAM_PID" 2>/dev/null; do
  sleep 30
done

actual_source_manifest_sha256=$(sha256sum "$SOURCE_MANIFEST" | awk '{print $1}')
if [[ "$actual_source_manifest_sha256" != "$EXPECTED_SOURCE_MANIFEST_SHA256" ]]; then
  echo "source transfer manifest hash changed" >&2
  exit 1
fi

mkdir -p "$AUDIT_ROOT"
destination_manifest="$AUDIT_ROOT/destination.sha256"
temporary_manifest="$AUDIT_ROOT/destination.sha256.tmp.$$"
(
  cd "$BASE_DIR"
  LC_ALL=C find "$TRANSFER_ROOT_A" "$TRANSFER_ROOT_B" -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "$temporary_manifest"
mv "$temporary_manifest" "$destination_manifest"

actual_files=$(wc -l < "$destination_manifest")
if [[ "$actual_files" != "$EXPECTED_FILES" ]]; then
  echo "destination transfer tree has $actual_files files, expected $EXPECTED_FILES" >&2
  exit 1
fi
if ! cmp -s "$SOURCE_MANIFEST" "$destination_manifest"; then
  echo "destination transfer tree differs from the frozen source manifest" >&2
  exit 1
fi
(
  cd "$BASE_DIR"
  sha256sum -c --quiet "$SOURCE_MANIFEST"
)

destination_manifest_sha256=$(sha256sum "$destination_manifest" | awk '{print $1}')
audit_script_sha256=$(sha256sum "$AUDIT_SCRIPT" | awk '{print $1}')
total_bytes=$(
  cd "$BASE_DIR"
  find "$TRANSFER_ROOT_A" "$TRANSFER_ROOT_B" -type f -printf '%s\n' \
    | awk '{value += $1} END {printf "%.0f\n", value}'
)
export destination_manifest destination_manifest_sha256 audit_script_sha256 total_bytes
"$TASK_PYTHON" - <<'PY'
import json
import os
import pathlib
import tempfile

target = pathlib.Path(os.environ["AUDIT_ROOT"]) / "transfer_tree_audit.json"
result = {
    "status": "verified_remote_tree_transfer",
    "passed": True,
    "source_manifest_path": os.environ["SOURCE_MANIFEST"],
    "source_manifest_sha256": os.environ["EXPECTED_SOURCE_MANIFEST_SHA256"],
    "destination_manifest_path": os.environ["destination_manifest"],
    "destination_manifest_sha256": os.environ["destination_manifest_sha256"],
    "files": int(os.environ["EXPECTED_FILES"]),
    "bytes": int(os.environ["total_bytes"]),
    "base_dir": os.environ["BASE_DIR"],
    "transfer_roots": [
        os.environ["TRANSFER_ROOT_A"],
        os.environ["TRANSFER_ROOT_B"],
    ],
    "audit_script_path": os.environ["AUDIT_SCRIPT"],
    "audit_script_sha256": os.environ["audit_script_sha256"],
    "code_commit": os.environ["GWYOLO_CODE_COMMIT"],
}
target.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(json.dumps(result, indent=2, sort_keys=True))
PY
