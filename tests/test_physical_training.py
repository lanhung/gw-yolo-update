from __future__ import annotations

import numpy as np
import pytest

from gwyolo.physical_training import physical_split_audit, relative_component_mask


def test_relative_component_mask_handles_physical_amplitudes() -> None:
    power = np.zeros((1, 1, 2, 3), dtype=np.float64)
    power[0, 0, 1] = [1e-42, 1e-40, 5e-42]
    mask = relative_component_mask(power)
    assert mask.sum() == 1
    assert mask[0, 0, 1, 1] == 1


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
