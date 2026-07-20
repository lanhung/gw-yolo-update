from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.physical_training import (
    build_snr_curriculum_manifest,
    physical_split_audit,
    relative_component_mask,
    scale_component_for_transform,
)


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
