from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

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
