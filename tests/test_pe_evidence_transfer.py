from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.pe import (
    run_pe_robustness_evaluation,
    run_within_backend_pe_robustness_portfolio,
)
from gwyolo.pe_evidence_transfer import (
    export_within_backend_pe_evidence_bundle,
    import_within_backend_pe_evidence_bundle,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _identity(path: Path) -> tuple[str, str]:
    return str(path.resolve()), file_sha256(path)


def _source_rows(tmp_path: Path, backend: str) -> list[dict]:
    common = tmp_path / "common"
    base_manifest = _write(common / "base.jsonl", '{"injection_id":"i-1"}\n')
    native_config = _write(
        tmp_path / backend.lower() / "conditioning.yaml",
        "sample_rate: 2048\n",
    )
    common_prior = _write(common / "prior.yaml", "population: BBH\n")
    contamination = _write(common / "contamination.json", '{"glitch_id":"g-1"}\n')
    mask_artifact = _write(common / "mask.npz", "deterministic-mask")
    mask_model = _write(common / "mask-model.pt", "model")
    mask_policy = _write(common / "mask-policy.yaml", "policy: frozen\n")
    rows = []
    for condition_index, condition in enumerate(
        ("clean", "contaminated", "mask_conditioned")
    ):
        analysis = _write(
            common / f"{condition}.npz",
            f"physical-analysis-{condition}",
        )
        native = _write(
            tmp_path / backend.lower() / f"{condition}.hdf5",
            f"{backend}-native-{condition}",
        )
        posterior = tmp_path / backend.lower() / f"{condition}-posterior.npz"
        posterior.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            posterior,
            chirp_mass=np.asarray([29.0, 30.0, 31.0]) + condition_index * 0.1,
        )
        analysis_path, analysis_sha = _identity(analysis)
        base_path, base_sha = _identity(base_manifest)
        native_path, native_sha = _identity(native)
        config_path, config_sha = _identity(native_config)
        row = {
            "backend": backend,
            "backend_version": "test",
            "backend_model_hash": f"{backend.lower()}-model",
            "prior_hash": f"{backend.lower()}-prior",
            "waveform_approximant": f"{backend}-waveform",
            "detector_set": ["H1", "L1"],
            "calibration_version": "none",
            "source_event_hash": "source-event-1",
            "hardware": {"gpu": "test"},
            "latency_scope": "test-latency-scope",
            "sky_area_estimator": {"method": "test"},
            "injection_id": "i-1",
            "waveform_id": "w-1",
            "gps_block": "gps-1",
            "condition": condition,
            "split": "val",
            "truth": {"chirp_mass": 30.0},
            "posterior_path": str(posterior.resolve()),
            "posterior_sha256": file_sha256(posterior),
            "latency_seconds": 1.0 + condition_index,
            "effective_sample_size": 2.0,
            "sky_area_90_deg2": 10.0 + condition_index,
            "analysis_input_path": analysis_path,
            "analysis_input_sha256": analysis_sha,
            "input_sample_rate_hz": 2048,
            "input_duration_seconds": 4,
            "input_post_trigger_seconds": 1,
            "input_ifos": ["H1", "L1"],
            "base_injection_manifest_path": base_path,
            "base_injection_manifest_sha256": base_sha,
            "common_prior_path": str(common_prior.resolve()),
            "common_prior_sha256": file_sha256(common_prior),
            "native_conditioning_path": native_path,
            "native_conditioning_sha256": native_sha,
            "native_conditioning_config_path": config_path,
            "native_conditioning_config_sha256": config_sha,
        }
        if condition in {"contaminated", "mask_conditioned"}:
            contamination_path, contamination_sha = _identity(contamination)
            row.update(
                {
                    "glitch_id": "g-1",
                    "contamination_manifest_path": contamination_path,
                    "contamination_manifest_sha256": contamination_sha,
                }
            )
        if condition == "mask_conditioned":
            artifact_path, artifact_sha = _identity(mask_artifact)
            model_path, model_sha = _identity(mask_model)
            policy_path, policy_sha = _identity(mask_policy)
            row.update(
                {
                    "mask_conditioning_mode": "cleaned_strain",
                    "mask_artifact_path": artifact_path,
                    "mask_artifact_sha256": artifact_sha,
                    "mask_model_path": model_path,
                    "mask_model_sha256": model_sha,
                    "mask_policy_path": policy_path,
                    "mask_policy_sha256": policy_sha,
                }
            )
        rows.append(row)
    return rows


def _within_backend_summary(
    tmp_path: Path,
    backend: str,
    rows: list[dict],
) -> Path:
    root = tmp_path / backend.lower()
    manifest = root / "posterior_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    batch = root / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "status": (
                    "real_dingo_official_native_paired_robustness_batch_complete"
                    if backend == "DINGO"
                    else "real_amplfi_common_batch_complete"
                ),
                "rows": len(rows),
                "paired_injections": 1,
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": file_sha256(manifest),
                "run_identity": {"required_split": "val"},
            }
        ),
        encoding="utf-8",
    )
    robustness = root / "robustness.json"
    run_pe_robustness_evaluation(
        manifest,
        robustness,
        credible_level=0.8,
        bootstrap_replicates=20,
        require_publication_provenance=True,
        require_cross_backend_join=False,
        minimum_physical_groups=1,
    )
    summary = root / "summary.json"
    model = _write(root / "model.pt", f"{backend}-validation-model")
    native_prior = _write(root / "native-prior.yaml", f"backend: {backend}\n")
    model_metadata = root / "model-metadata.json"
    model_metadata.write_text(
        json.dumps(
            {
                "backend": backend,
                "model_path": str(model.resolve()),
                "model_sha256": file_sha256(model),
                "artifacts": {
                    "native_prior": {
                        "path": str(native_prior.resolve()),
                        "sha256": file_sha256(native_prior),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    summary.write_text(
        json.dumps(
            {
                "status": (
                    "validation_only_dingo_official_native_paired_smoke_complete"
                    if backend == "DINGO"
                    else "validation_only_amplfi_within_backend_paired_smoke_complete"
                ),
                "scientific_claim_allowed": False,
                "cross_backend_absolute_comparison_allowed": False,
                "test_rows_read": 0,
                "artifacts": {
                    "posterior_batch": {
                        "path": str(batch.resolve()),
                        "sha256": file_sha256(batch),
                    },
                    "robustness": {
                        "path": str(robustness.resolve()),
                        "sha256": file_sha256(robustness),
                    },
                    "model_metadata": {
                        "path": str(model_metadata.resolve()),
                        "sha256": file_sha256(model_metadata),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return summary


def test_portable_pe_evidence_bundle_survives_cross_machine_relocation(
    tmp_path: Path,
) -> None:
    imports = {}
    for backend in ("DINGO", "AMPLFI"):
        rows = _source_rows(tmp_path, backend)
        summary = _within_backend_summary(tmp_path, backend, rows)
        export_root = tmp_path / "exports" / backend.lower()
        exported = export_within_backend_pe_evidence_bundle(summary, export_root)
        assert exported["backend"] == backend
        assert exported["bundle_schema_version"] == 2
        assert exported["required_split"] == "val"
        assert exported["test_rows_read"] == 0
        assert exported["total_files"] > 4

        transferred_root = tmp_path / "transferred" / backend.lower()
        shutil.copytree(export_root, transferred_root)
        imported = import_within_backend_pe_evidence_bundle(
            transferred_root / "within_backend_pe_evidence_bundle.json",
            tmp_path / "imports" / backend.lower(),
        )
        assert imported["backend"] == backend
        projected_summary = json.loads(
            Path(imported["projected_summary_path"]).read_text(encoding="utf-8")
        )
        projected_metadata = json.loads(
            Path(
                projected_summary["artifacts"]["model_metadata"]["path"]
            ).read_text(encoding="utf-8")
        )
        assert Path(projected_metadata["model_path"]).is_file()
        assert (
            file_sha256(projected_metadata["model_path"])
            == projected_metadata["model_sha256"]
        )
        assert Path(
            projected_metadata["artifacts"]["native_prior"]["path"]
        ).is_file()
        assert imported["test_rows_read"] == 0
        imports[backend] = imported

    portfolio = run_within_backend_pe_robustness_portfolio(
        imports["DINGO"]["projected_batch_report_path"],
        imports["DINGO"]["projected_robustness_report_path"],
        imports["AMPLFI"]["projected_batch_report_path"],
        imports["AMPLFI"]["projected_robustness_report_path"],
        tmp_path / "portfolio.jsonl",
        tmp_path / "portfolio.json",
        credible_level=0.8,
        bootstrap_replicates=20,
        minimum_physical_groups=1,
    )
    assert portfolio["matched_event_gate"] is True
    assert portfolio["common_injection_count"] == 1
    assert portfolio["absolute_cross_backend_comparison_allowed"] is False


def test_import_rejects_tampered_content_object(tmp_path: Path) -> None:
    rows = _source_rows(tmp_path, "AMPLFI")
    summary = _within_backend_summary(tmp_path, "AMPLFI", rows)
    export_root = tmp_path / "export"
    receipt = export_within_backend_pe_evidence_bundle(summary, export_root)
    object_identity = next(
        identity for identity in receipt["files"] if identity["role"] == "content_object"
    )
    object_path = export_root / object_identity["relative_path"]
    object_path.write_bytes(object_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="failed hash replay"):
        import_within_backend_pe_evidence_bundle(
            export_root / "within_backend_pe_evidence_bundle.json",
            tmp_path / "import",
        )
