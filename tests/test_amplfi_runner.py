from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _runner_module():
    path = Path(__file__).parents[1] / "scripts/run_amplfi_common_event.py"
    specification = importlib.util.spec_from_file_location("amplfi_common_runner", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_amplfi_runner_reconstructs_frozen_architecture_and_prior_bounds() -> None:
    root = Path(__file__).parents[1]
    training = yaml.safe_load(
        (root / "configs/amplfi_common_bbh_publication.yaml").read_text()
    )
    native_prior = yaml.safe_load(
        (root / "configs/amplfi_common_bbh_training_prior.yaml").read_text()
    )
    runner = _runner_module()
    arch, embedding, inference = runner._architecture_settings(training)
    assert inference == list(runner.INFERENCE_PARAMETERS)
    assert arch["transforms"] == 20
    assert arch["hidden_features"] == [512, 512, 512]
    assert embedding["time_context_dim"] == 8
    assert embedding["freq_context_dim"] == 128
    bounds = runner.native_bounds(native_prior, training)
    assert bounds["chirp_mass"] == (15.0, 100.0)
    assert bounds["mass_ratio"] == (0.125, 0.999)
    assert bounds["distance"] == (100.0, 3100.0)
