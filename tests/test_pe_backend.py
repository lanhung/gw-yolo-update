from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gwyolo.io import file_sha256
from gwyolo.pe_backend import audit_pe_backend_lock, run_pe_backend_lock_audit


def _write_executable(path: Path, version: str) -> None:
    path.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        + json.dumps(
            {
                "python": "3.10.14",
                "distribution": version,
                "torch": "2.5.1",
                "cuda_available": True,
                "cuda_version": "12.4",
                "gpu": "test-gpu",
                "prefix": f"/isolated/{version}",
                "base_prefix": "/base",
                "packages": [["backend", version]],
                "environment_packages_sha256": f"environment-{version}",
            }
        )
        + "\nEOF\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _git(tmp_path: Path, name: str, tag: str) -> tuple[Path, str]:
    import subprocess

    source = tmp_path / name
    source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "README").write_text(name, encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
    subprocess.run(["git", "tag", tag], cwd=source, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()
    return source, commit


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    backends = {}
    for backend, version in (("DINGO", "0.9.8"), ("AMPLFI", "0.6.0")):
        source, commit = _git(tmp_path, backend.lower(), f"v{version}")
        python = tmp_path / f"{backend.lower()}-python"
        _write_executable(python, version)
        model = tmp_path / f"{backend.lower()}.pt"
        metadata = tmp_path / f"{backend.lower()}.yaml"
        model.write_bytes(backend.encode())
        metadata.write_text(f"backend: {backend}\n", encoding="utf-8")
        for suffix, value in (
            ("SOURCE", source),
            ("PYTHON", python),
            ("MODEL", model),
            ("METADATA", metadata),
        ):
            monkeypatch.setenv(f"TEST_{backend}_{suffix}", str(value))
        backends[backend] = {
            "source_path_env": f"TEST_{backend}_SOURCE",
            "expected_git_tag": f"v{version}",
            "expected_git_commit": commit,
            "python_executable_env": f"TEST_{backend}_PYTHON",
            "python_requires": ">=3.10,<3.13",
            "distribution": "dingo-gw" if backend == "DINGO" else "amplfi",
            "expected_distribution_version": version,
            "environment_packages_sha256": f"environment-{version}",
            "model_path_env": f"TEST_{backend}_MODEL",
            "model_sha256": file_sha256(model),
            "model_metadata_path_env": f"TEST_{backend}_METADATA",
            "model_metadata_sha256": file_sha256(metadata),
        }
    config = {
        "schema_version": 1,
        "comparison_contract": {
            "population": "BBH",
            "source_ifos": ["H1", "L1"],
            "source_sample_rate_hz": 2048,
            "source_duration_seconds": 8,
            "conditions": ["clean", "contaminated", "mask_conditioned"],
            "identical_source_bytes_across_backends": True,
        },
        "backends": backends,
    }
    path = tmp_path / "lock.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_pe_backend_lock_accepts_two_isolated_pinned_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = audit_pe_backend_lock(_config(tmp_path, monkeypatch))
    assert report["publication_ready"] is True
    assert report["status"] == "ready"
    assert report["backends"]["DINGO"]["source"]["tag"] == "v0.9.8"


def test_pe_backend_lock_rejects_unresolved_model_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path, monkeypatch)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["backends"]["AMPLFI"]["model_sha256"] = "UNRESOLVED"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = audit_pe_backend_lock(path)
    assert report["publication_ready"] is False
    assert "AMPLFI: model SHA256 is unresolved" in report["failures"]

    output = tmp_path / "report.json"
    with pytest.raises(RuntimeError, match="lock is incomplete"):
        run_pe_backend_lock_audit(path, output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "incomplete"
