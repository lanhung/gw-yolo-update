from __future__ import annotations

import numpy as np
import pytest

from gwyolo.glitch_training import (
    GravitySpyNumericDataset,
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
