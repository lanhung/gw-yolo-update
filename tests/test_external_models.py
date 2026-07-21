from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import gwyolo.external_models as external_models
from gwyolo.external_models import (
    acquire_external_model_sources,
    run_external_model_source_acquisition,
)


def _config(tmp_path: Path, payload: bytes) -> Path:
    import hashlib

    config = {
        "schema_version": 1,
        "sources": [
            {
                "backend": "DINGO",
                "role": "posterior_model",
                "filename": "model.pt",
                "url": "https://example.invalid/model.pt",
                "size_bytes": len(payload),
                "checksum": {
                    "algorithm": "md5",
                    "value": hashlib.md5(payload).hexdigest(),
                },
            }
        ],
    }
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_external_model_source_verifies_hand_calculated_file(tmp_path: Path) -> None:
    payload = b"publication model fixture"
    config = _config(tmp_path, payload)
    output = tmp_path / "models"
    output.mkdir()
    (output / "model.pt").write_bytes(payload)
    report = acquire_external_model_sources(config, output)
    assert report["status"] == "verified"
    assert report["files"][0]["size_bytes"] == len(payload)
    assert report["files"][0]["valid"] is True


def test_external_model_source_refuses_missing_or_corrupt_file_atomically(
    tmp_path: Path,
) -> None:
    payload = b"expected"
    config = _config(tmp_path, payload)
    output = tmp_path / "models"
    report_path = tmp_path / "report.json"
    with pytest.raises(FileNotFoundError, match="verify-only"):
        run_external_model_source_acquisition(config, output, report_path)
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"

    output.mkdir(exist_ok=True)
    (output / "model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity lock"):
        acquire_external_model_sources(config, output, download=True)
    assert (output / "model.pt").read_bytes() == b"corrupt"


def test_external_model_source_resumes_transient_partial_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"resumable-publication-model"
    config = _config(tmp_path, payload)
    output = tmp_path / "models"
    calls = []

    def fake_curl(command: list[str], *, check: bool) -> None:
        assert check is True
        calls.append(command)
        partial = Path(command[command.index("--output") + 1])
        partial.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            partial.write_bytes(payload[:8])
            raise external_models.subprocess.CalledProcessError(18, command)
        assert partial.read_bytes() == payload[:8]
        partial.write_bytes(payload)

    monkeypatch.setattr(external_models.subprocess, "run", fake_curl)
    report = acquire_external_model_sources(
        config,
        output,
        download=True,
        transfer_attempts=3,
        retry_delay_seconds=0,
        maximum_stalled_attempts=2,
    )
    assert len(calls) == 2
    assert all("--continue-at" in command for command in calls)
    assert all("--retry-connrefused" in command for command in calls)
    assert report["files"][0]["download_attempts"] == 2
    assert report["files"][0]["initial_partial_bytes"] == 0
    assert report["files"][0]["downloaded_bytes_this_run"] == len(payload)
    assert (output / "model.pt").read_bytes() == payload
    assert not (output / "model.pt.part").exists()


def test_external_model_source_does_not_retry_permanent_curl_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"permanent-error-model"
    config = _config(tmp_path, payload)
    calls = []

    def fake_curl(command: list[str], *, check: bool) -> None:
        calls.append(command)
        raise external_models.subprocess.CalledProcessError(22, command)

    monkeypatch.setattr(external_models.subprocess, "run", fake_curl)
    with pytest.raises(external_models.subprocess.CalledProcessError):
        acquire_external_model_sources(
            config,
            tmp_path / "models",
            download=True,
            transfer_attempts=100,
            retry_delay_seconds=0,
        )
    assert len(calls) == 1


def test_external_model_source_stops_after_bounded_no_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, b"stalled-model")
    calls = []

    def stalled_curl(command: list[str], *, check: bool) -> None:
        calls.append(command)
        raise external_models.subprocess.CalledProcessError(18, command)

    monkeypatch.setattr(external_models.subprocess, "run", stalled_curl)
    with pytest.raises(RuntimeError, match="no byte progress for 2"):
        acquire_external_model_sources(
            config,
            tmp_path / "models",
            download=True,
            transfer_attempts=100,
            retry_delay_seconds=0,
            maximum_stalled_attempts=2,
        )
    assert len(calls) == 2


def test_external_model_source_promotes_complete_partial_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"already-complete-partial"
    config = _config(tmp_path, payload)
    output = tmp_path / "models"
    output.mkdir()
    (output / "model.pt.part").write_bytes(payload)

    def unexpected_curl(command: list[str], *, check: bool) -> None:
        raise AssertionError(f"network should not be used: {command}, {check}")

    monkeypatch.setattr(external_models.subprocess, "run", unexpected_curl)
    report = acquire_external_model_sources(config, output, download=True)
    assert report["files"][0]["download_attempts"] == 0
    assert report["files"][0]["initial_partial_bytes"] == len(payload)
    assert (output / "model.pt").read_bytes() == payload


def test_external_model_source_rejects_invalid_retry_controls(tmp_path: Path) -> None:
    config = _config(tmp_path, b"model")
    with pytest.raises(ValueError, match="transfer_attempts"):
        acquire_external_model_sources(config, tmp_path / "models", transfer_attempts=0)
    with pytest.raises(ValueError, match="retry_delay_seconds"):
        acquire_external_model_sources(
            config, tmp_path / "models", retry_delay_seconds=-1
        )
    with pytest.raises(ValueError, match="maximum_stalled_attempts"):
        acquire_external_model_sources(
            config, tmp_path / "models", maximum_stalled_attempts=0
        )
