from __future__ import annotations

import os
import subprocess
import fcntl
from pathlib import Path


def test_pe_backend_setup_fails_before_mutation_when_inputs_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/setup_pe_backends.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "PE_BASE_PYTHON" in completed.stderr


def test_pe_backend_setup_holds_an_atomic_installation_lock() -> None:
    script = Path(__file__).parents[1] / "scripts/setup_pe_backends.sh"
    source = script.read_text(encoding="utf-8")
    assert "PE_INSTALL_LOCK" in source
    assert 'flock -n "$pe_install_lock_fd"' in source
    assert "active package installation already exists" in source
    assert source.index('flock -n "$pe_install_lock_fd"') < source.index(
        "active package installation already exists"
    )


def test_pe_backend_setup_refuses_a_second_lock_owner(tmp_path) -> None:
    script = Path(__file__).parents[1] / "scripts/setup_pe_backends.sh"
    lock_path = tmp_path / "backend.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = {
            "PATH": os.environ["PATH"],
            "PE_BASE_PYTHON": "/missing/python",
            "DINGO_SOURCE_DIR": "/missing/dingo",
            "DINGO_EXPECTED_COMMIT": "0" * 40,
            "DINGO_EXPECTED_TAG": "v0",
            "DINGO_VENV": "/missing/dingo-venv",
            "AMPLFI_SOURCE_DIR": "/missing/amplfi",
            "AMPLFI_EXPECTED_COMMIT": "1" * 40,
            "AMPLFI_EXPECTED_TAG": "v0",
            "AMPLFI_VENV": "/missing/amplfi-venv",
            "PE_ENVIRONMENT_REPORT_DIR": "/missing/report",
            "PE_INSTALL_LOCK": str(lock_path),
        }
        completed = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    assert completed.returncode == 3
    assert "atomic installation lock" in completed.stderr
