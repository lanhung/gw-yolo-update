from __future__ import annotations

import numpy as np
import pytest

from gwyolo.physical_training import (
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
