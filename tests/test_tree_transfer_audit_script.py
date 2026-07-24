from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "queue_tree_transfer_audit.sh"


def test_tree_transfer_audit_is_waited_atomic_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'while kill -0 "$UPSTREAM_PID"' in source
    assert "source transfer manifest hash changed" in source
    assert 'cmp -s "$SOURCE_MANIFEST" "$destination_manifest"' in source
    assert 'sha256sum -c --quiet "$SOURCE_MANIFEST"' in source
    assert "os.replace(temporary, target)" in source
    assert '"status": "verified_remote_tree_transfer"' in source


def test_tree_transfer_audit_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 1
    compile(snippets[0], f"{SCRIPT.name}:heredoc", "exec")


def test_tree_transfer_audit_rejects_changed_source_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "source.sha256"
    manifest.write_text("", encoding="utf-8")
    environment = {
        **os.environ,
        "UPSTREAM_PID": "99999999",
        "TASK_PYTHON": sys.executable,
        "AUDIT_SCRIPT": str(SCRIPT),
        "AUDIT_ROOT": str(tmp_path / "audit"),
        "SOURCE_MANIFEST": str(manifest),
        "BASE_DIR": str(tmp_path),
        "TRANSFER_ROOT_A": "a",
        "TRANSFER_ROOT_B": "b",
        "EXPECTED_FILES": "1",
        "EXPECTED_SOURCE_MANIFEST_SHA256": "0" * 64,
        "GWYOLO_CODE_COMMIT": "test",
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "source transfer manifest hash changed" in completed.stderr
    assert not (tmp_path / "audit/transfer_tree_audit.json").exists()


def test_tree_transfer_audit_accepts_only_exact_file_set(tmp_path: Path) -> None:
    first = tmp_path / "tree-a"
    second = tmp_path / "tree-b"
    first.mkdir()
    second.mkdir()
    (first / "one.bin").write_bytes(b"one")
    (second / "two.bin").write_bytes(b"two")
    relative_paths = [Path("tree-a/one.bin"), Path("tree-b/two.bin")]
    manifest = tmp_path / "source.sha256"
    manifest.write_text(
        "".join(
            f"{hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()}  {path}\n"
            for path in relative_paths
        ),
        encoding="utf-8",
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    audit = tmp_path / "audit"
    environment = {
        **os.environ,
        "UPSTREAM_PID": "99999999",
        "TASK_PYTHON": sys.executable,
        "AUDIT_SCRIPT": str(SCRIPT),
        "AUDIT_ROOT": str(audit),
        "SOURCE_MANIFEST": str(manifest),
        "BASE_DIR": str(tmp_path),
        "TRANSFER_ROOT_A": "tree-a",
        "TRANSFER_ROOT_B": "tree-b",
        "EXPECTED_FILES": "2",
        "EXPECTED_SOURCE_MANIFEST_SHA256": manifest_hash,
        "GWYOLO_CODE_COMMIT": "test-commit",
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((audit / "transfer_tree_audit.json").read_text())
    assert receipt["passed"] is True
    assert receipt["files"] == 2
    assert receipt["bytes"] == 6
    assert receipt["source_manifest_sha256"] == manifest_hash
    assert receipt["destination_manifest_sha256"] == manifest_hash
