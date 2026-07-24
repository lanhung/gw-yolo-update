from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.io import file_sha256
from gwyolo.training_transfer import (
    export_detector_set_training_bundle,
    import_detector_set_training_bundle,
)


def _json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_detector_training_bundle_reprojects_overlap_clean_and_background(
    tmp_path: Path,
) -> None:
    overlap_train_artifact = tmp_path / "overlap-train.npz"
    overlap_train_artifact.write_bytes(b"overlap-train")
    overlap_val_artifact = tmp_path / "overlap-val.npz"
    overlap_val_artifact.write_bytes(b"overlap-val")
    overlap_train = _jsonl(
        tmp_path / "overlap-train.jsonl",
        [
            {
                "mixture_id": "m-train",
                "path": str(overlap_train_artifact),
                "sha256": file_sha256(overlap_train_artifact),
            }
        ],
    )
    overlap_val = _jsonl(
        tmp_path / "overlap-val.jsonl",
        [
            {
                "mixture_id": "m-val",
                "path": str(overlap_val_artifact),
                "sha256": file_sha256(overlap_val_artifact),
            }
        ],
    )
    train_report = _json(
        tmp_path / "train-report.json",
        {
            "split": "train",
            "manifest_path": str(overlap_train),
            "manifest_sha256": file_sha256(overlap_train),
        },
    )
    val_report = _json(
        tmp_path / "val-report.json",
        {
            "split": "val",
            "manifest_path": str(overlap_val),
            "manifest_sha256": file_sha256(overlap_val),
        },
    )
    readiness = _json(
        tmp_path / "readiness.json",
        {
            "status": "audited_detector_set_signal_bank_readiness",
            "signal_overlap_materialization_authorized": True,
            "detector_complete_clean_training_authorized": False,
            "detector_set_robustness_ablation_ready": False,
        },
    )
    audit = _json(tmp_path / "audit.json", {"passed": True})
    capacity = _json(tmp_path / "capacity.json", {"passed": True})
    artifacts = {
        "train_report": train_report,
        "validation_report": val_report,
        "joint_group_audit": audit,
        "expansion_readiness_audit": readiness,
        "capacity_report": capacity,
    }
    overlap_receipt = _json(
        tmp_path / "overlap-receipt.json",
        {
            "status": "verified_detector_set_overlap_robustness_corpus",
            "passed": True,
            "test_rows_read": 0,
            "test_evaluation": None,
            "same_distribution_data_scaling_claim_allowed": False,
            "artifacts": {
                label: {"path": str(path), "sha256": file_sha256(path)}
                for label, path in artifacts.items()
            },
        },
    )
    background = tmp_path / "background.hdf5"
    background.write_bytes(b"shared-background")
    clean_artifacts = []
    clean_manifests = []
    for split in ("train", "val"):
        signal = tmp_path / f"signal-{split}.npz"
        signal.write_bytes(f"signal-{split}".encode())
        clean_artifacts.append(signal)
        clean_manifests.append(
            _jsonl(
                tmp_path / f"clean-{split}.jsonl",
                [
                    {
                        "split": split,
                        "injection_id": f"i-{split}",
                        "waveform_id": f"w-{split}",
                        "materialized_path": str(signal),
                        "materialized_sha256": file_sha256(signal),
                        "background_source_files": {
                            "H1": {
                                "path": str(background),
                                "sha256": file_sha256(background),
                            }
                        },
                    }
                ],
            )
        )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("model: test\n", encoding="utf-8")
    bundle = export_detector_set_training_bundle(
        overlap_receipt,
        clean_manifests[0],
        clean_manifests[1],
        checkpoint,
        {"finetune": config},
        tmp_path / "bundle",
    )
    assert bundle["object_count"] == 12
    imported = import_detector_set_training_bundle(
        tmp_path / "bundle" / "detector_set_training_input_bundle.json",
        tmp_path / "imported",
    )
    assert imported["passed"] is True
    assert imported["detector_complete_clean_training_authorized"] is False
    projected_clean = json.loads(
        Path(imported["manifests"]["clean_train"]["path"])
        .read_text(encoding="utf-8")
        .strip()
    )
    assert Path(projected_clean["materialized_path"]).read_bytes() == b"signal-train"
    projected_background = Path(
        projected_clean["background_source_files"]["H1"]["path"]
    )
    assert projected_background.read_bytes() == b"shared-background"
    projected_overlap = json.loads(
        Path(imported["manifests"]["overlap_train"]["path"])
        .read_text(encoding="utf-8")
        .strip()
    )
    assert Path(projected_overlap["path"]).read_bytes() == b"overlap-train"


def test_detector_training_bundle_import_rejects_object_tamper(
    tmp_path: Path,
) -> None:
    # Reuse the complete round trip above through a direct pytest invocation fixture
    # would couple tests. A minimal malformed receipt is enough to exercise fail-closed
    # inventory verification.
    root = tmp_path / "bundle"
    object_path = root / "objects" / "00" / ("0" * 64)
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(b"tampered")
    receipt = _json(
        root / "detector_set_training_input_bundle.json",
        {
            "status": "portable_detector_set_training_input_bundle",
            "passed": True,
            "schema": "portable_detector_set_training_inputs_v1",
            "test_rows_read": 0,
            "test_evaluation": None,
            "detector_complete_clean_training_authorized": False,
            "object_count": 1,
            "objects": {
                "0" * 64: {
                    "path": str(object_path.relative_to(root)),
                    "bytes": len(b"tampered"),
                }
            },
        },
    )
    with pytest.raises(ValueError, match="object drift"):
        import_detector_set_training_bundle(receipt, tmp_path / "imported")
