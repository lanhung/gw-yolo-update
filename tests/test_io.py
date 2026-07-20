from __future__ import annotations

import pytest

from gwyolo.io import training_tensor_config


def test_training_tensor_config_accepts_one_training_family() -> None:
    numeric = {"numeric_training": {"tensor": {"time_bins": 96}}}
    physical = {"physical_training": {"tensor": {"time_bins": 1024}}}
    assert training_tensor_config(numeric)["time_bins"] == 96
    assert training_tensor_config(physical)["time_bins"] == 1024
    with pytest.raises(ValueError, match="exactly one"):
        training_tensor_config({**numeric, **physical})
