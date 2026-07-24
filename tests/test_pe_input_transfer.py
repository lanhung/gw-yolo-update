from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gwyolo.io import file_sha256
from gwyolo.pe_input_transfer import (
    export_paired_pe_input_bundle,
    import_paired_pe_input_bundle,
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    base = _write(source / "lineage/base.jsonl", '{"injection_id":"i-1"}\n')
    prior = _write(source / "lineage/prior.yaml", "population: BBH\n")
    contamination = _write(source / "lineage/contamination.jsonl", "{}\n")
    mask = _write(source / "lineage/mask.npz", "mask")
    model = _write(source / "lineage/model.pt", "model")
    policy = _write(source / "lineage/policy.yaml", "mode: cleaned_strain\n")
    receipt = _write(source / "lineage/endpoint.json", '{"test_rows_read":0}\n')
    common_rows = []
    for condition in ("clean", "contaminated", "mask_conditioned"):
        analysis = _write(source / f"common/{condition}.npz", condition)
        row = {
            "injection_id": "i-1",
            "waveform_id": "w-1",
            "condition": condition,
            "split": "val",
            "truth": {"chirp_mass": 30.0},
            "source_event_hash": "event-hash",
            "common_asd_sha256": "asd-hash",
            "analysis_input_path": str(analysis.resolve()),
            "analysis_input_sha256": file_sha256(analysis),
            "base_injection_manifest_path": str(base.resolve()),
            "base_injection_manifest_sha256": file_sha256(base),
            "common_prior_path": str(prior.resolve()),
            "common_prior_sha256": file_sha256(prior),
            "input_ifos": ["H1", "L1"],
            "input_sample_rate_hz": 4096,
            "input_duration_seconds": 16,
            "input_post_trigger_seconds": 2,
        }
        if condition != "clean":
            row.update(
                {
                    "contamination_manifest_path": str(contamination.resolve()),
                    "contamination_manifest_sha256": file_sha256(contamination),
                }
            )
        if condition == "mask_conditioned":
            row.update(
                {
                    "mask_artifact_path": str(mask.resolve()),
                    "mask_artifact_sha256": file_sha256(mask),
                    "mask_model_path": str(model.resolve()),
                    "mask_model_sha256": file_sha256(model),
                    "mask_policy_path": str(policy.resolve()),
                    "mask_policy_sha256": file_sha256(policy),
                }
            )
        common_rows.append(row)

    common_manifest = source / "common-sources/common_pe_inputs.jsonl"
    _write(
        common_manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in common_rows),
    )
    common_report = {
        "status": "backend_neutral_paired_pe_inputs_materialized",
        "scientific_claim_allowed": False,
        "required_split": "val",
        "paired_injections": 1,
        "rows": 3,
        "manifest_path": str(common_manifest.resolve()),
        "manifest_sha256": file_sha256(common_manifest),
    }
    _write(
        source / "common-sources/common_pe_inputs_report.json",
        json.dumps(common_report),
    )

    native_reports = {}
    for backend in ("DINGO", "AMPLFI"):
        config = _write(
            source / f"{backend.lower()}-native/config.yaml",
            f"backend: {backend}\n",
        )
        native_rows = []
        for row in common_rows:
            artifact = _write(
                source / f"{backend.lower()}-native/artifacts/{row['condition']}.hdf5",
                f"{backend}-{row['condition']}",
            )
            native_rows.append(
                {
                    **row,
                    "backend": backend,
                    "native_conditioning_path": str(artifact.resolve()),
                    "native_conditioning_sha256": file_sha256(artifact),
                    "native_conditioning_config_path": str(config.resolve()),
                    "native_conditioning_config_sha256": file_sha256(config),
                }
            )
        manifest = source / f"{backend.lower()}-native/{backend.lower()}_native_conditioning.jsonl"
        _write(
            manifest,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in native_rows),
        )
        report = {
            "status": "native_pe_conditioning_materialized",
            "scientific_claim_allowed": False,
            "backend": backend,
            "rows": 3,
            "paired_injections": 1,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": file_sha256(manifest),
            "run_identity": {
                "required_split": "val",
                "source_manifest_sha256": file_sha256(common_manifest),
            },
        }
        _write(
            source / f"{backend.lower()}-native/native_conditioning_report.json",
            json.dumps(report),
        )
        native_reports[backend] = report

    summary = source / "paired_pe_smoke_summary.json"
    _write(
        summary,
        json.dumps(
            {
                "status": "paired_pe_native_inputs_smoke_complete",
                "scientific_claim_allowed": False,
                "paired_injections": 1,
                "test_rows_read": 0,
                "source_receipts": {
                    "independent_validation_endpoint": {
                        "path": str(receipt.resolve()),
                        "sha256": file_sha256(receipt),
                    }
                },
                "reports": {
                    "common_sources": common_report,
                    "dingo_native": native_reports["DINGO"],
                    "amplfi_native": native_reports["AMPLFI"],
                },
            }
        ),
    )
    return summary


def test_paired_pe_input_bundle_survives_cross_machine_relocation(
    tmp_path: Path,
) -> None:
    summary = _fixture(tmp_path)
    export_root = tmp_path / "export"
    exported = export_paired_pe_input_bundle(summary, export_root)
    assert exported["required_split"] == "val"
    assert exported["test_rows_read"] == 0
    assert exported["rows"] == {
        "common_manifest": 3,
        "dingo_manifest": 3,
        "amplfi_manifest": 3,
    }
    assert exported["total_files"] > 7

    transferred = tmp_path / "other-machine"
    shutil.copytree(export_root, transferred)
    shutil.rmtree(summary.parent)
    imported = import_paired_pe_input_bundle(
        transferred / "paired_pe_input_bundle.json",
        tmp_path / "projection",
    )
    assert imported["passed"] is True
    projected_summary = json.loads(
        Path(imported["projected_summary_path"]).read_text(encoding="utf-8")
    )
    for backend, directory in (
        ("DINGO", "dingo-native"),
        ("AMPLFI", "amplfi-native"),
    ):
        report = projected_summary["reports"][f"{backend.lower()}_native"]
        assert report["backend"] == backend
        assert Path(report["manifest_path"]).is_file()
        rows = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
        assert len(rows) == 3
        assert all(Path(row["analysis_input_path"]).is_file() for row in rows)
        assert all(Path(row["native_conditioning_path"]).is_file() for row in rows)
        disk_report = json.loads(
            (tmp_path / "projection" / directory / "native_conditioning_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert disk_report == report


def test_paired_pe_input_bundle_rejects_tampered_object(tmp_path: Path) -> None:
    summary = _fixture(tmp_path)
    export_root = tmp_path / "export"
    exported = export_paired_pe_input_bundle(summary, export_root)
    content = next(item for item in exported["files"] if item["role"] == "content_object")
    (export_root / content["relative_path"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="failed hash replay"):
        import_paired_pe_input_bundle(
            export_root / "paired_pe_input_bundle.json",
            tmp_path / "projection",
        )


def test_paired_pe_input_bundle_rejects_backend_input_drift(tmp_path: Path) -> None:
    summary = _fixture(tmp_path)
    value = json.loads(summary.read_text(encoding="utf-8"))
    report = value["reports"]["amplfi_native"]
    manifest = Path(report["manifest_path"])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["source_event_hash"] = "different-event"
    _write(
        manifest,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    report["manifest_sha256"] = file_sha256(manifest)
    _write(manifest.parent / "native_conditioning_report.json", json.dumps(report))
    summary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="common input identity"):
        export_paired_pe_input_bundle(summary, tmp_path / "export")
