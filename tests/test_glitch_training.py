from __future__ import annotations

import json

import numpy as np
import pytest

from gwyolo.glitch_training import (
    GravitySpyNumericDataset,
    audit_gravityspy_network_numeric_corpus,
    gravityspy_numeric_split_audit,
    inverse_label_sampling_weights,
)
from gwyolo.io import file_sha256


def _row(glitch_id: str, split: str, block: str, label: str = "Blip") -> dict:
    return {
        "glitch_id": glitch_id,
        "split": split,
        "network_gps_block": block,
        "ml_label": label,
    }


def test_gravityspy_numeric_split_audit_rejects_block_leakage() -> None:
    train = [_row("g1", "train", "shared")]
    validation = [_row("g2", "val", "shared")]
    with pytest.raises(ValueError, match="split leakage"):
        gravityspy_numeric_split_audit(train, validation)
    validation[0]["network_gps_block"] = "validation-only"
    assert gravityspy_numeric_split_audit(train, validation)["passed"]


def test_network_corpus_audit_rejects_cross_split_source_file(tmp_path) -> None:
    reports = []
    for split, glitch, block in (
        ("train", "g-train", "b-train"),
        ("val", "g-val", "b-val"),
    ):
        sample = tmp_path / f"{split}.npz"
        np.savez(sample, features=np.asarray([1 if split == "train" else 2]))
        manifest = tmp_path / f"{split}.jsonl"
        row = {
            "glitch_id": glitch,
            "split": split,
            "network_gps_block": block,
            "ml_label": "Blip",
            "observing_run": "O3a",
            "ifo": "H1",
            "available_ifos": ["H1", "L1"],
            "network_strain_sources": {
                "H1": {"hdf5_url": "https://example/shared.hdf5"},
                "L1": {"hdf5_url": f"https://example/{split}-L1.hdf5"},
            },
            "aligned_network_context": True,
            "path": str(sample),
            "sha256": file_sha256(sample),
        }
        manifest.write_text(json.dumps(row) + "\n")
        report = tmp_path / f"{split}-report.json"
        report.write_text(
            json.dumps(
                {
                    "status": "verified_merged_gravityspy_aligned_network_numeric_split",
                    "split": split,
                    "manifest_path": str(manifest),
                    "manifest_sha256": file_sha256(manifest),
                    "rows": 1,
                    "labels": {"Blip": 1},
                    "runs": {"O3a": 1},
                    "detector_subset_counts": {"H1L1": 1},
                }
            )
        )
        reports.append(report)
    with pytest.raises(ValueError, match="split leakage"):
        audit_gravityspy_network_numeric_corpus(
            reports[0], reports[1], tmp_path / "audit.json"
        )
    val_manifest = tmp_path / "val.jsonl"
    val_row = json.loads(val_manifest.read_text())
    val_row["network_strain_sources"]["H1"]["hdf5_url"] = (
        "https://example/val-H1.hdf5"
    )
    val_manifest.write_text(json.dumps(val_row) + "\n")
    val_report = json.loads(reports[1].read_text())
    val_report["manifest_sha256"] = file_sha256(val_manifest)
    reports[1].write_text(json.dumps(val_report))
    result = audit_gravityspy_network_numeric_corpus(
        reports[0], reports[1], tmp_path / "audit.json"
    )
    assert result["passed"]
    assert all(not values for values in result["split_audit"]["cross_split_overlaps"].values())


def test_inverse_label_weights_equalize_total_label_mass() -> None:
    rows = [
        _row("b1", "train", "b1", "Blip"),
        _row("b2", "train", "b2", "Blip"),
        _row("t1", "train", "t1", "Tomte"),
    ]
    weights = inverse_label_sampling_weights(rows)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[2] == pytest.approx(2 * weights[0])
    assert weights[:2].sum() == pytest.approx(weights[2])


def test_gravityspy_numeric_dataset_verifies_shape_and_hash(tmp_path) -> None:
    sample = tmp_path / "sample.npz"
    features = np.zeros((1, 3, 4, 5), dtype=np.float16)
    mask = np.ones((1, 3, 4, 5), dtype=np.float32)
    np.savez(sample, features=features, glitch_mask=mask)
    row = {
        **_row("g1", "train", "block"),
        "path": str(sample),
        "sha256": file_sha256(sample),
    }
    dataset = GravitySpyNumericDataset([row], 3, 4, 5)
    loaded_features, loaded_mask = dataset[0]
    assert loaded_features.shape == loaded_mask.shape == (3, 4, 5)
    assert loaded_features.dtype == loaded_mask.dtype == np.float32
