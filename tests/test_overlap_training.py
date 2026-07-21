from __future__ import annotations

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.overlap_training import (
    PhysicalOverlapDataset,
    glitch_family_sampling_weights,
    overlap_training_split_audit,
    summarize_glitch_family_counts,
)


def _row(split: str, suffix: str) -> dict:
    return {
        "split": split,
        "mixture_id": f"m-{suffix}",
        "injection_id": f"i-{suffix}",
        "waveform_id": f"w-{suffix}",
        "glitch_id": f"g-{suffix}",
        "injection_gps_block": f"ib-{suffix}",
        "network_gps_block": f"gb-{suffix}",
    }


def test_overlap_training_split_audit_covers_both_physical_group_families() -> None:
    report = overlap_training_split_audit([_row("train", "a")], [_row("val", "b")])
    assert report["passed"]
    assert all(not values for values in report["cross_split_overlaps"].values())
    leaked = _row("val", "b")
    leaked["glitch_id"] = "g-a"
    with pytest.raises(ValueError, match="split leakage"):
        overlap_training_split_audit([_row("train", "a")], [leaked])


def test_glitch_family_sampling_is_bounded_and_adds_no_physical_rows() -> None:
    rows = [
        {"ml_label": label}
        for label in ["common"] * 16 + ["medium"] * 4 + ["rare"]
    ]
    weights, report = glitch_family_sampling_weights(
        rows, exponent=0.5, maximum_weight_ratio=3.0, minimum_family_count=2
    )
    by_label = report["family_relative_weights"]
    assert by_label == {"common": 1.0, "medium": 2.0, "rare": 1.0}
    assert weights.shape == (21,)
    assert weights.mean() == pytest.approx(1.0)
    assert report["physical_rows"] == report["sample_draws_per_epoch"] == 21
    assert report["adds_independent_physical_examples"] is False
    assert report["families_below_minimum_count_not_boosted"] == ["rare"]


def test_glitch_family_metrics_use_hand_calculated_pixel_counts() -> None:
    summary = summarize_glitch_family_counts(
        {
            "Blip": np.asarray([[9, 0, 0], [2, 1, 1]]),
            "Tomte": np.asarray([[8, 0, 0], [3, 0, 1]]),
        },
        {"Blip": 2, "Tomte": 1},
    )
    assert summary["Blip"]["physical_rows"] == 2
    assert summary["Blip"]["iou"] == pytest.approx(0.5)
    assert summary["Blip"]["dice"] == pytest.approx(2 / 3)
    assert summary["Tomte"]["recall"] == pytest.approx(0.75)


def test_overlap_dataset_preserves_both_masks_and_availability(tmp_path) -> None:
    sample = tmp_path / "sample.npz"
    features = np.zeros((3, 2, 4, 5), dtype=np.float16)
    chirp = np.zeros_like(features, dtype=np.uint8)
    glitch = np.zeros_like(features, dtype=np.uint8)
    features[1] = 2
    chirp[1, :, 1, 2] = 1
    glitch[1, :, 2, 3] = 1
    np.savez(
        sample,
        features=features,
        chirp_mask=chirp,
        glitch_mask=glitch,
        detector_availability=np.asarray([0, 1, 0], dtype=np.uint8),
        ifos=np.asarray(["H1", "L1", "V1"]),
        q_values=np.asarray([4, 8], dtype=np.float32),
    )
    row = {**_row("train", "x"), "path": str(sample), "sha256": file_sha256(sample)}
    dataset = PhysicalOverlapDataset(
        [row], ("H1", "L1", "V1"), (4.0, 8.0), 4, 5
    )
    observed_features, targets, availability = dataset[0]
    assert observed_features.shape == (6, 4, 5)
    assert targets.shape == (2, 6, 4, 5)
    assert availability.tolist() == [0, 1, 0]
    assert int(targets[0].sum()) == 2
    assert int(targets[1].sum()) == 2


def test_overlap_dataset_rejects_nonzero_unavailable_planes(tmp_path) -> None:
    sample = tmp_path / "invalid.npz"
    features = np.zeros((2, 1, 2, 2), dtype=np.float32)
    features[1, 0, 0, 0] = 1
    np.savez(
        sample,
        features=features,
        chirp_mask=np.zeros_like(features),
        glitch_mask=np.zeros_like(features),
        detector_availability=np.asarray([1, 0]),
        ifos=np.asarray(["H1", "L1"]),
        q_values=np.asarray([4]),
    )
    row = {**_row("train", "x"), "path": str(sample), "sha256": file_sha256(sample)}
    dataset = PhysicalOverlapDataset([row], ("H1", "L1"), (4.0,), 2, 2)
    with pytest.raises(ValueError, match="must be zero"):
        dataset[0]
