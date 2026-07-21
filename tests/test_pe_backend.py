from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from gwyolo.io import file_sha256
from gwyolo.pe_backend import (
    audit_pe_backend_lock,
    freeze_pe_backend_model_metadata,
    run_pe_backend_lock_audit,
)


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
    analysis_prior = tmp_path / "analysis-prior.yaml"
    analysis_prior.write_text("prior: common\n", encoding="utf-8")
    for backend, version in (("DINGO", "0.9.8"), ("AMPLFI", "0.6.0")):
        source, commit = _git(tmp_path, backend.lower(), f"v{version}")
        python = tmp_path / f"{backend.lower()}-python"
        _write_executable(python, version)
        model = tmp_path / f"{backend.lower()}.pt"
        metadata = tmp_path / f"{backend.lower()}.yaml"
        model.write_bytes(backend.encode())
        artifacts = {
            "training_config": tmp_path / f"{backend.lower()}-training.yaml",
            "training_data_manifest": tmp_path / f"{backend.lower()}-training.jsonl",
            "analysis_prior": analysis_prior,
            "selection_report": tmp_path / f"{backend.lower()}-selection.json",
            "native_conditioning_config": tmp_path / f"{backend.lower()}-conditioning.yaml",
        }
        for label, artifact in artifacts.items():
            if label == "selection_report":
                artifact.write_text(
                    json.dumps(
                        {
                            "status": "validation_selected_checkpoint",
                            "publication_eligible": True,
                            "selection_split": "validation",
                            "selection_metric": "validation_loss",
                            "selected_checkpoint_sha256": file_sha256(model),
                        }
                    ),
                    encoding="utf-8",
                )
            elif artifact != analysis_prior:
                artifact.write_text(f"backend: {backend}\nartifact: {label}\n", encoding="utf-8")
        if backend == "AMPLFI":
            native_prior = tmp_path / "amplfi-native-prior.yaml"
            native_prior.write_text("prior: native-amplfi\n", encoding="utf-8")
            projection = tmp_path / "amplfi-prior-projection.json"
            projection.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "publication_ready": True,
                        "canonical_prior_sha256": file_sha256(analysis_prior),
                        "amplfi_prior_sha256": file_sha256(native_prior),
                        "amplfi_training_config_sha256": file_sha256(
                            artifacts["training_config"]
                        ),
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            artifacts.update(
                {
                    "native_prior": native_prior,
                    "prior_projection_report": projection,
                }
            )
        else:
            native_prior = tmp_path / "dingo-native-prior.yaml"
            native_prior.write_text("prior: native-dingo\n", encoding="utf-8")
            projection = tmp_path / "dingo-prior-projection.json"
            projection.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "publication_ready": True,
                        "canonical_prior_sha256": file_sha256(analysis_prior),
                        "dingo_prior_config_sha256": file_sha256(native_prior),
                        "dingo_training_config_sha256": file_sha256(
                            artifacts["training_config"]
                        ),
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            initialization_model = tmp_path / "dingo-initialization.pt"
            initialization_model.write_bytes(b"dingo-initialization")
            artifacts.update(
                {
                    "native_prior": native_prior,
                    "prior_projection_report": projection,
                    "initialization_model": initialization_model,
                }
            )
        metadata.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "backend": backend,
                    "model_sha256": file_sha256(model),
                    "population": "BBH",
                    "source_input": {
                        "ifos": ["H1", "L1"],
                        "sample_rate_hz": 4096,
                        "duration_seconds": 16,
                        "post_trigger_seconds": 2,
                        "common_asd_required": True,
                    },
                    "analysis_waveform_approximant": "IMRPhenomXPHM",
                    "native_model_waveform_approximant": "IMRPhenomPv2",
                    "model_training_backend_version": version,
                    "native_inference_parameters": [
                        "chirp_mass",
                        "mass_ratio",
                        "distance" if backend == "AMPLFI" else "luminosity_distance",
                    ],
                    "reported_parameter_mapping": {
                        "chirp_mass": "chirp_mass",
                        "mass_ratio": "mass_ratio",
                        "luminosity_distance": (
                            "distance" if backend == "AMPLFI" else "luminosity_distance"
                        ),
                    },
                    "selection_split": "validation",
                    "selection_metric": "validation_loss",
                    "artifacts": {
                        label: {"path": str(artifact), "sha256": file_sha256(artifact)}
                        for label, artifact in artifacts.items()
                    },
                }
            ),
            encoding="utf-8",
        )
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
            "source_sample_rate_hz": 4096,
            "source_duration_seconds": 16,
            "source_post_trigger_seconds": 2,
            "common_asd_required": True,
            "common_asd_condition_invariant": True,
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


def test_pe_backend_lock_rejects_abbreviated_or_malformed_source_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path, monkeypatch)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["backends"]["DINGO"]["expected_git_commit"] = "abc123"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = audit_pe_backend_lock(path)
    assert report["publication_ready"] is False
    assert any("full 40-character" in failure for failure in report["failures"])


def test_pe_backend_model_freeze_requires_validation_selected_checkpoint(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"weights")
    artifacts = {}
    for label in (
        "training_config",
        "training_data_manifest",
        "analysis_prior",
        "native_conditioning_config",
    ):
        path = tmp_path / f"{label}.yaml"
        path.write_text(f"artifact: {label}\n", encoding="utf-8")
        artifacts[label] = path
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "validation_selected_checkpoint",
                "publication_eligible": True,
                "selection_split": "validation",
                "selection_metric": "validation_loss",
                "selected_checkpoint_sha256": file_sha256(model),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "model-metadata.json"
    initialization_model = tmp_path / "initialization-model.pt"
    initialization_model.write_bytes(b"initialization-weights")
    native_prior = tmp_path / "native-prior.yaml"
    native_prior.write_text("prior: native-dingo\n", encoding="utf-8")
    projection = tmp_path / "prior-projection.json"
    projection.write_text(
        json.dumps(
            {
                "status": "passed",
                "publication_ready": True,
                "canonical_prior_sha256": file_sha256(artifacts["analysis_prior"]),
                "dingo_prior_config_sha256": file_sha256(native_prior),
                "dingo_training_config_sha256": file_sha256(
                    artifacts["training_config"]
                ),
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    report = freeze_pe_backend_model_metadata(
        backend="DINGO",
        model_path=model,
        training_config_path=artifacts["training_config"],
        training_data_manifest_path=artifacts["training_data_manifest"],
        analysis_prior_path=artifacts["analysis_prior"],
        selection_report_path=selection,
        native_conditioning_config_path=artifacts["native_conditioning_config"],
        output_path=output,
        population="BBH",
        source_ifos=["H1", "L1"],
        source_sample_rate_hz=4096,
        source_duration_seconds=16,
        source_post_trigger_seconds=2,
        analysis_waveform_approximant="IMRPhenomXPHM",
        native_model_waveform_approximant="IMRPhenomPv2",
        model_training_backend_version="0.5.8",
        native_inference_parameters=["chirp_mass", "mass_ratio"],
        reported_parameter_mapping=["chirp_mass=chirp_mass", "mass_ratio=mass_ratio"],
        native_prior_path=native_prior,
        prior_projection_report_path=projection,
        initialization_model_path=initialization_model,
    )
    assert report["model_sha256"] == file_sha256(model)
    assert report["selection_split"] == "validation"
    assert json.loads(output.read_text(encoding="utf-8"))["backend"] == "DINGO"

    bad_selection = json.loads(selection.read_text(encoding="utf-8"))
    bad_selection["selection_split"] = "test"
    selection.write_text(json.dumps(bad_selection), encoding="utf-8")
    with pytest.raises(ValueError, match="validation split"):
        freeze_pe_backend_model_metadata(
            backend="DINGO",
            model_path=model,
            training_config_path=artifacts["training_config"],
            training_data_manifest_path=artifacts["training_data_manifest"],
            analysis_prior_path=artifacts["analysis_prior"],
            selection_report_path=selection,
            native_conditioning_config_path=artifacts["native_conditioning_config"],
            output_path=output,
            population="BBH",
            source_ifos=["H1", "L1"],
            source_sample_rate_hz=4096,
            source_duration_seconds=16,
            source_post_trigger_seconds=2,
            analysis_waveform_approximant="IMRPhenomXPHM",
            native_model_waveform_approximant="IMRPhenomPv2",
            model_training_backend_version="0.5.8",
            native_inference_parameters=["chirp_mass", "mass_ratio"],
            reported_parameter_mapping=["chirp_mass=chirp_mass", "mass_ratio=mass_ratio"],
            native_prior_path=native_prior,
            prior_projection_report_path=projection,
            initialization_model_path=initialization_model,
        )


def test_amplfi_model_freeze_binds_native_prior_projection(tmp_path: Path) -> None:
    model = tmp_path / "amplfi.ckpt"
    model.write_bytes(b"amplfi-weights")
    training = tmp_path / "training.yaml"
    training.write_text("backend: AMPLFI\n", encoding="utf-8")
    manifest = tmp_path / "training.jsonl"
    manifest.write_text('{"split":"train","gps_block":"gps:1:64"}\n', encoding="utf-8")
    canonical_prior = tmp_path / "canonical-prior.yaml"
    canonical_prior.write_text("prior: common\n", encoding="utf-8")
    native_prior = tmp_path / "native-prior.yaml"
    native_prior.write_text("prior: amplfi\n", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "validation_selected_checkpoint",
                "publication_eligible": True,
                "selection_split": "validation",
                "selection_metric": "validation_loss",
                "selected_checkpoint_sha256": file_sha256(model),
            }
        ),
        encoding="utf-8",
    )
    conditioning = tmp_path / "conditioning.yaml"
    conditioning.write_text("backend: AMPLFI\n", encoding="utf-8")
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(
            {
                "status": "passed",
                "publication_ready": True,
                "canonical_prior_sha256": file_sha256(canonical_prior),
                "amplfi_prior_sha256": file_sha256(native_prior),
                "amplfi_training_config_sha256": file_sha256(training),
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    kwargs = {
        "backend": "AMPLFI",
        "model_path": model,
        "training_config_path": training,
        "training_data_manifest_path": manifest,
        "analysis_prior_path": canonical_prior,
        "selection_report_path": selection,
        "native_conditioning_config_path": conditioning,
        "output_path": tmp_path / "metadata.json",
        "population": "BBH",
        "source_ifos": ["H1", "L1"],
        "source_sample_rate_hz": 4096,
        "source_duration_seconds": 16,
        "source_post_trigger_seconds": 2,
        "analysis_waveform_approximant": "IMRPhenomXPHM",
        "native_model_waveform_approximant": "IMRPhenomXPHM",
        "model_training_backend_version": "0.6.0",
        "native_inference_parameters": ["chirp_mass", "distance"],
        "reported_parameter_mapping": [
            "chirp_mass=chirp_mass",
            "luminosity_distance=distance",
        ],
    }
    with pytest.raises(ValueError, match="requires native prior"):
        freeze_pe_backend_model_metadata(**kwargs)

    metadata = freeze_pe_backend_model_metadata(
        **kwargs,
        native_prior_path=native_prior,
        prior_projection_report_path=projection,
    )
    assert metadata["artifacts"]["native_prior"]["sha256"] == file_sha256(
        native_prior
    )
    assert metadata["artifacts"]["prior_projection_report"]["sha256"] == file_sha256(
        projection
    )

    projection_report = json.loads(projection.read_text(encoding="utf-8"))
    projection_report["canonical_prior_sha256"] = "0" * 64
    projection.write_text(json.dumps(projection_report), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical_prior_sha256"):
        freeze_pe_backend_model_metadata(
            **kwargs,
            native_prior_path=native_prior,
            prior_projection_report_path=projection,
        )


def test_amplfi_backend_lock_rejects_prior_projection_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path, monkeypatch)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata_path = Path(os.environ["TEST_AMPLFI_METADATA"])
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    projection_path = Path(metadata["artifacts"]["prior_projection_report"]["path"])
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["amplfi_prior_sha256"] = "0" * 64
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    metadata["artifacts"]["prior_projection_report"]["sha256"] = file_sha256(
        projection_path
    )
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    config["backends"]["AMPLFI"]["model_metadata_sha256"] = file_sha256(metadata_path)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = audit_pe_backend_lock(path)
    assert report["publication_ready"] is False
    assert any(
        "amplfi_prior_sha256 does not match" in failure
        for failure in report["failures"]
    )


def test_dingo_backend_lock_rejects_prior_projection_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path, monkeypatch)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata_path = Path(os.environ["TEST_DINGO_METADATA"])
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    projection_path = Path(metadata["artifacts"]["prior_projection_report"]["path"])
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["dingo_prior_config_sha256"] = "0" * 64
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    metadata["artifacts"]["prior_projection_report"]["sha256"] = file_sha256(
        projection_path
    )
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    config["backends"]["DINGO"]["model_metadata_sha256"] = file_sha256(
        metadata_path
    )
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = audit_pe_backend_lock(path)
    assert report["publication_ready"] is False
    assert any(
        "dingo_prior_config_sha256 does not match" in failure
        for failure in report["failures"]
    )


def test_pe_backend_lock_rejects_different_reported_parameter_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _config(tmp_path, monkeypatch)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata_path = Path(os.environ["TEST_AMPLFI_METADATA"])
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["reported_parameter_mapping"].pop("luminosity_distance")
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    config["backends"]["AMPLFI"]["model_metadata_sha256"] = file_sha256(metadata_path)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = audit_pe_backend_lock(path)
    assert report["publication_ready"] is False
    assert "DINGO/AMPLFI reported common parameter sets differ" in report["failures"]
