from __future__ import annotations

import numpy as np
import pytest

from gwyolo.waveforms import place_waveform_samples, validate_recipe_identities


def test_place_waveform_samples_clips_both_edges_by_hand() -> None:
    inside = place_waveform_samples(10.0, 2, 6, 11.0, np.asarray([1, 2, 3]))
    assert inside.tolist() == [0, 0, 1, 2, 3, 0]
    left = place_waveform_samples(10.0, 2, 6, 9.0, np.asarray([1, 2, 3, 4]))
    assert left.tolist() == [3, 4, 0, 0, 0, 0]
    right = place_waveform_samples(10.0, 2, 6, 12.5, np.asarray([7, 8, 9]))
    assert right.tolist() == [0, 0, 0, 0, 0, 7]


def test_place_waveform_interpolates_subsample_epoch() -> None:
    result = place_waveform_samples(10.0, 4, 16, 10.125, np.ones(32))
    assert np.isfinite(result).all()
    assert result[4:12] == pytest.approx(np.ones(8), abs=0.04)


def test_recipe_identity_audit_rejects_gps_leakage() -> None:
    rows = [
        {"injection_id": "i1", "waveform_id": "w1", "split": "val", "gps_block": "g"},
        {"injection_id": "i2", "waveform_id": "w2", "split": "test", "gps_block": "g"},
    ]
    with pytest.raises(ValueError, match="GPS-block leakage"):
        validate_recipe_identities(rows)
