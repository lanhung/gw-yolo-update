from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.physical_training import (
    build_snr_curriculum_manifest,
    build_snr_quota_manifest,
    gate_component_by_ifo_snr,
    mask_endpoint_timing_error_seconds,
    peak_to_endpoint_timing_error_seconds,
    focal_binary_cross_entropy,
    physical_split_audit,
    relative_component_mask,
    scale_component_for_transform,
    summarize_binary_mask_counts,
)


def test_binary_mask_counts_are_hand_calculated() -> None:
    metrics = summarize_binary_mask_counts(3, 2, 1)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.6)
    assert metrics["recall"] == pytest.approx(0.75)


def test_mask_endpoint_timing_error_is_hand_calculated() -> None:
    probability = np.asarray([[0.1, 0.8, 0.9, 0.2, 0.7, 0.1, 0.1, 0.1]])
    expected = np.asarray([[False, True, True, True, False, False, False, False]])
    result = mask_endpoint_timing_error_seconds(probability, expected, 0.5, 8.0)
    # Predicted endpoint bin 4 minus target endpoint bin 3 at one second per bin.
    assert result == {
        "target_present": True,
        "prediction_present": True,
        "absolute_error_seconds": 1.0,
    }
    missed = mask_endpoint_timing_error_seconds(
        np.zeros((1, 8)), expected, 0.5, 8.0
    )
    assert missed["target_present"] and not missed["prediction_present"]
    assert missed["absolute_error_seconds"] is None


def test_peak_to_endpoint_timing_error_is_hand_calculated() -> None:
    probability = np.asarray([[0.1, 0.3, 0.9, 0.2, 0.4, 0.1, 0.1, 0.1]])
    expected = np.asarray([[False, True, True, True, False, False, False, False]])
    # Peak bin 2 versus target endpoint bin 3 at one second per bin.
    assert peak_to_endpoint_timing_error_seconds(probability, expected, 8.0) == 1.0
    assert peak_to_endpoint_timing_error_seconds(
        probability, np.zeros_like(expected), 8.0
    ) is None


def test_focal_gamma_zero_matches_binary_cross_entropy() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 1, 1, 1))
    target = torch.ones_like(logits)
    positive = torch.ones((1, 1, 1, 1))
    ordinary = focal_binary_cross_entropy(logits, target, positive, 0.0)
    focal = focal_binary_cross_entropy(logits, target, positive, 2.0)
    assert float(ordinary) == pytest.approx(np.log(2.0))
    assert float(focal) == pytest.approx(np.log(2.0) / 4.0)


def test_relative_component_mask_handles_physical_amplitudes() -> None:
    power = np.zeros((1, 1, 2, 3), dtype=np.float64)
    power[0, 0, 1] = [1e-42, 1e-40, 5e-42]
    mask = relative_component_mask(power)
    assert mask.sum() == 1
    assert mask[0, 0, 1, 1] == 1


def test_component_scaling_prevents_physical_float32_power_underflow() -> None:
    component = np.asarray([[0.0, 1e-24, -2e-24], [0.0, 0.0, 0.0]])
    scaled = scale_component_for_transform(component)
    assert scaled[0].tolist() == pytest.approx([0.0, 0.5, -1.0])
    assert scaled[1].tolist() == [0.0, 0.0, 0.0]
    assert np.max(np.abs(scaled[0])) == 1.0


def test_component_visibility_gate_is_per_ifo() -> None:
    component = np.ones((2, 4))
    gated = gate_component_by_ifo_snr(
        component, ["H1", "L1"], {"H1": 1.5, "L1": 3.0}, 2.0
    )
    assert gated[0].tolist() == [0.0] * 4
    assert gated[1].tolist() == [1.0] * 4


def test_snr_curriculum_rescales_only_subfloor_training_rows(tmp_path) -> None:
    manifest = tmp_path / "train.jsonl"
    rows = [
        {
            "split": "train",
            "injection_id": f"i{index}",
            "waveform_id": f"w{index}",
            "gps_block": f"g{index}",
            "network_optimal_snr": snr,
        }
        for index, snr in enumerate((2.0, 6.0))
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = build_snr_curriculum_manifest(manifest, tmp_path / "out", seed=3)
    output = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert report["rescaled_rows"] == 1
    assert 4.0 <= output[0]["training_network_optimal_snr"] < 8.0
    assert output[0]["training_signal_scale"] > 1.0
    assert output[1]["training_network_optimal_snr"] == 6.0
    assert output[1]["training_signal_scale"] == 1.0


def test_snr_quota_assigns_exact_hand_calculated_counts(tmp_path) -> None:
    manifest = tmp_path / "train.jsonl"
    rows = [
        {
            "split": "train",
            "injection_id": f"i{index}",
            "waveform_id": f"w{index}",
            "gps_block": f"g{index}",
            "network_optimal_snr": 10.0,
        }
        for index in range(10)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = build_snr_quota_manifest(
        manifest, tmp_path / "quota", [(4, 8, 0.6), (8, 16, 0.4)], seed=7
    )
    assert report["achieved_counts"] == {"4-8": 6, "8-16": 4}
    output = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert len({row["injection_id"] for row in output}) == 10
    assert all(
        (4 <= row["training_network_optimal_snr"] < 8)
        if row["training_snr_quota_bin"] == "4-8"
        else (8 <= row["training_network_optimal_snr"] < 16)
        for row in output
    )
    assert report["validation_or_test_rows_modified"] == 0


def test_physical_split_audit_rejects_gps_or_waveform_leakage() -> None:
    train = [
        {
            "split": "train",
            "injection_id": "train-injection",
            "waveform_id": "shared-waveform",
            "gps_block": "train-block",
        }
    ]
    validation = [
        {
            "split": "val",
            "injection_id": "val-injection",
            "waveform_id": "shared-waveform",
            "gps_block": "val-block",
        }
    ]
    with pytest.raises(ValueError, match="split leakage"):
        physical_split_audit(train, validation)

    validation[0]["waveform_id"] = "val-waveform"
    report = physical_split_audit(train, validation)
    assert report["passed"]
    assert all(not values for values in report["cross_split_overlaps"].values())
