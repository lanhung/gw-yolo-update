from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gwyolo.io import file_sha256
from gwyolo.pe_compatibility import (
    adjudicate_dingo_runtime_failure,
    run_dingo_runtime_failure_adjudication,
)


def _fixture(tmp_path: Path, log_text: str) -> tuple[Path, Path]:
    config = tmp_path / "sources.yaml"
    config.write_text("schema_version: 1\n", encoding="utf-8")
    posterior = tmp_path / "posterior.pt"
    initialization = tmp_path / "time.pt"
    posterior.write_bytes(b"official-posterior")
    initialization.write_bytes(b"official-time-initialization")
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(
        json.dumps(
            {
                "status": "verified",
                "download_enabled": True,
                "config_sha256": file_sha256(config),
                "files": [
                    {
                        "role": "posterior_model",
                        "path": str(posterior.resolve()),
                        "sha256": file_sha256(posterior),
                        "valid": True,
                    },
                    {
                        "role": "time_initialization_model",
                        "path": str(initialization.resolve()),
                        "sha256": file_sha256(initialization),
                        "valid": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    attempt_log = tmp_path / "attempt.log"
    attempt_log.write_text(log_text, encoding="utf-8")
    failure = tmp_path / "failure.json"
    failure.write_text(
        json.dumps(
            {
                "status": "official_dingo_dual_model_load_failed",
                "passed": False,
                "fallback_allowed": False,
                "scientific_claim_allowed": False,
                "test_rows_read": 0,
                "backend": "DINGO",
                "backend_version": "0.9.8",
                "model_source_config_path": str(config.resolve()),
                "model_source_config_sha256": file_sha256(config),
                "model_acquisition_report_path": str(acquisition.resolve()),
                "model_acquisition_report_sha256": file_sha256(acquisition),
                "posterior_model_path": str(posterior.resolve()),
                "posterior_model_sha256": file_sha256(posterior),
                "initialization_model_path": str(initialization.resolve()),
                "initialization_model_sha256": file_sha256(initialization),
                "attempt_log_path": str(attempt_log.resolve()),
                "attempt_log_sha256": file_sha256(attempt_log),
            }
        ),
        encoding="utf-8",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "primary_runtime_version": "0.9.8",
                "fallback_runtime_version": "0.5.8",
                "compatibility_allow_patterns": [
                    r"AttributeError: (Can't|Cannot) get attribute"
                ],
                "infrastructure_deny_patterns": [r"out of memory", r"CUDA.*unavailable"],
            }
        ),
        encoding="utf-8",
    )
    return failure, policy


def test_dingo_compatibility_adjudication_authorizes_exact_native_fallback(
    tmp_path: Path,
) -> None:
    failure, policy = _fixture(
        tmp_path,
        "Traceback\nAttributeError: Can't get attribute 'LegacyEmbedding'\n",
    )
    report = adjudicate_dingo_runtime_failure(failure, policy)
    assert report["status"] == "dingo_native_runtime_fallback_authorized"
    assert report["fallback_allowed"] is True
    assert report["authorized_fallback_runtime_version"] == "0.5.8"
    assert report["model_substitution_allowed"] is False
    assert report["test_rows_read"] == 0


def test_dingo_compatibility_adjudication_infrastructure_deny_overrides_match(
    tmp_path: Path,
) -> None:
    failure, policy = _fixture(
        tmp_path,
        "AttributeError: Can't get attribute 'LegacyEmbedding'\nCUDA out of memory\n",
    )
    output = tmp_path / "adjudication.json"
    with pytest.raises(RuntimeError, match="not eligible"):
        run_dingo_runtime_failure_adjudication(failure, policy, output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "dingo_native_runtime_fallback_rejected"
    assert report["fallback_allowed"] is False
    assert report["matched_compatibility_patterns"]
    assert report["matched_infrastructure_patterns"]


def test_dingo_compatibility_adjudication_rejects_unknown_failure(tmp_path: Path) -> None:
    failure, policy = _fixture(tmp_path, "RuntimeError: unknown model-load failure\n")
    report = adjudicate_dingo_runtime_failure(failure, policy)
    assert report["passed"] is False
    assert report["matched_compatibility_patterns"] == []


def test_dingo_compatibility_adjudication_replays_log_hash(tmp_path: Path) -> None:
    failure, policy = _fixture(
        tmp_path, "AttributeError: Can't get attribute 'LegacyEmbedding'\n"
    )
    failure_value = json.loads(failure.read_text(encoding="utf-8"))
    Path(failure_value["attempt_log_path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attempt log SHA256 mismatch"):
        adjudicate_dingo_runtime_failure(failure, policy)
