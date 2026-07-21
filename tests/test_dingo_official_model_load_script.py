from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_dingo_official_model_load.sh"


def _digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    value.update(path.read_bytes())
    return value.hexdigest()


def test_dingo_official_load_runner_is_dual_model_and_test_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "MODEL_ACQUISITION_REPORT",
        "MODEL_SOURCE_CONFIG",
        "posterior_model",
        "time_initialization_model",
        "run_pe_model_load_smoke.py",
        "expected-model-init-sha256",
        "verified_official_dingo_dual_model_load",
        "model_acquisition_report_sha256",
        "official_dingo_dual_model_load_failed",
        "fallback_constraint",
        "attempt_log_sha256",
        "test_rows_read",
    ):
        assert token in source
    assert '--backend DINGO' in source


def test_dingo_official_load_embedded_python_compiles() -> None:
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    assert len(snippets) == 3
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_dingo_official_load_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_dingo_official_load_freezes_acquisition_and_dual_model_receipt(
    tmp_path: Path,
) -> None:
    roles = {
        "model_manifest": "manifest.md",
        "training_settings": "settings.yaml",
        "posterior_model": "posterior.pt",
        "time_initialization_model": "time.pt",
    }
    sources = []
    acquired = []
    paths = {}
    for index, (role, filename) in enumerate(roles.items()):
        path = tmp_path / filename
        path.write_bytes(f"official-{role}-{index}".encode())
        paths[role] = path
        checksum = _digest(path, "md5")
        sources.append(
            {
                "backend": "DINGO",
                "role": role,
                "filename": filename,
                "url": f"https://example.invalid/{filename}",
                "size_bytes": path.stat().st_size,
                "checksum": {"algorithm": "md5", "value": checksum},
            }
        )
        acquired.append(
            {
                "backend": "DINGO",
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "expected_size_bytes": path.stat().st_size,
                "checksum_algorithm": "md5",
                "checksum": checksum,
                "expected_checksum": checksum,
                "sha256": _digest(path),
                "valid": True,
            }
        )
    config = tmp_path / "sources.yaml"
    config.write_text(
        yaml.safe_dump({"schema_version": 1, "sources": sources}), encoding="utf-8"
    )
    recorded_config = tmp_path / "acquisition-sources.yaml"
    recorded_config.write_bytes(config.read_bytes())
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "status": "verified",
                "download_enabled": True,
                "config_path": str(recorded_config.resolve()),
                "config_sha256": _digest(config),
                "files": acquired,
            }
        ),
        encoding="utf-8",
    )
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    preflight = subprocess.run(
        [sys.executable, "-c", snippets[0], str(config), str(acquisition)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert preflight.stdout.splitlines() == [
        str(paths["posterior_model"].resolve()),
        _digest(paths["posterior_model"]),
        str(paths["time_initialization_model"].resolve()),
        _digest(paths["time_initialization_model"]),
    ]

    load = tmp_path / "load.json"
    load.write_text(
        json.dumps(
            {
                "status": "real_pe_backend_model_load_smoke_complete",
                "scientific_claim_allowed": False,
                "backend": "DINGO",
                "backend_version": "0.9.8",
                "device": "cuda",
                "artifacts": {
                    "model": {
                        "path": str(paths["posterior_model"].resolve()),
                        "sha256": _digest(paths["posterior_model"]),
                    },
                    "model_init": {
                        "path": str(paths["time_initialization_model"].resolve()),
                        "sha256": _digest(paths["time_initialization_model"]),
                    },
                },
                "observations": {
                    "model_parameter_count": 10,
                    "initialization_model_parameter_count": 5,
                },
                "environment": {"gpu": "test"},
            }
        ),
        encoding="utf-8",
    )
    receipt = tmp_path / "receipt.json"
    finalized = subprocess.run(
        [
            sys.executable,
            "-c",
            snippets[2],
            str(config),
            str(acquisition),
            str(load),
            str(paths["posterior_model"]),
            _digest(paths["posterior_model"]),
            str(paths["time_initialization_model"]),
            _digest(paths["time_initialization_model"]),
            "0.9.8",
            "cuda",
            "test-commit",
            str(receipt),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finalized.returncode == 0, finalized.stderr
    frozen = json.loads(receipt.read_text(encoding="utf-8"))
    assert frozen["status"] == "verified_official_dingo_dual_model_load"
    assert frozen["test_rows_read"] == 0
    assert frozen["posterior_model_sha256"] == _digest(paths["posterior_model"])
    assert frozen["initialization_model_sha256"] == _digest(
        paths["time_initialization_model"]
    )


def test_dingo_official_load_freezes_machine_readable_failure(tmp_path: Path) -> None:
    config = tmp_path / "sources.yaml"
    acquisition = tmp_path / "acquisition.json"
    posterior = tmp_path / "posterior.pt"
    initialization = tmp_path / "time.pt"
    attempt_log = tmp_path / "attempt.log"
    for path, content in (
        (config, "schema_version: 1\n"),
        (acquisition, "{}\n"),
        (posterior, "posterior"),
        (initialization, "initialization"),
        (attempt_log, "compatibility error\n"),
    ):
        path.write_text(content, encoding="utf-8")
    receipt = tmp_path / "failure.json"
    snippets = re.findall(
        r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            snippets[1],
            str(config),
            str(acquisition),
            str(posterior),
            _digest(posterior),
            str(initialization),
            _digest(initialization),
            "0.9.8",
            "cuda",
            sys.executable,
            "test-commit",
            str(attempt_log),
            "17",
            str(receipt),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    frozen = json.loads(receipt.read_text(encoding="utf-8"))
    assert frozen["status"] == "official_dingo_dual_model_load_failed"
    assert frozen["passed"] is False
    assert frozen["fallback_allowed"] is False
    assert frozen["exit_code"] == 17
    assert frozen["attempt_log_sha256"] == _digest(attempt_log)
    assert frozen["test_rows_read"] == 0
