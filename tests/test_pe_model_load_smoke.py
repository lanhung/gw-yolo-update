from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


def test_model_load_smoke_rejects_hash_before_backend_import(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/run_pe_model_load_smoke.py"
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a real backend model")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--backend",
            "DINGO",
            "--model",
            str(model),
            "--expected-model-sha256",
            hashlib.sha256(b"another model").hexdigest(),
            "--output",
            str(tmp_path / "report.json"),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "SHA256 mismatch" in completed.stderr
    assert not (tmp_path / "report.json").exists()
