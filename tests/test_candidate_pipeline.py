from __future__ import annotations

import json

import pytest

from gwyolo.candidate_pipeline import (
    select_candidate_timing_method,
    validate_candidate_model_selection,
)
from gwyolo.io import file_sha256


def test_candidate_pipeline_selects_only_calibrated_local_cluster_method() -> None:
    calibration = {
        "methods": {
            "mask_profile_parabolic": {
                "calibration_gate_passed": False,
                "empirical_timing_uncertainty_seconds": 0.08,
            },
            "local_whitened_strain_envelope_per_mask_cluster_v1": {
                "calibration_gate_passed": True,
                "empirical_timing_uncertainty_seconds": 0.004,
            },
        }
    }
    method, uncertainty = select_candidate_timing_method(calibration)
    assert method == "local_whitened_strain_envelope_per_mask_cluster_v1"
    assert uncertainty == 0.004


def test_candidate_pipeline_refuses_resolution_only_timing() -> None:
    with pytest.raises(ValueError, match="exactly one passing"):
        select_candidate_timing_method(
            {
                "methods": {
                    "mask_profile_parabolic": {
                        "calibration_gate_passed": True,
                        "empirical_timing_uncertainty_seconds": 0.08,
                    }
                }
            }
        )


def test_candidate_pipeline_binds_five_seed_model_and_config(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("model: detector_set\n")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "completed_five_seed_source_safe_overlap_validation",
                "passed": True,
                "test_data_opened": False,
                "selected_checkpoint_sha256": file_sha256(checkpoint),
                "common_artifact_hashes": {
                    "config_file_sha256": file_sha256(config)
                },
            }
        )
    )
    result = validate_candidate_model_selection(selection, checkpoint, config)
    assert result["selected_checkpoint_sha256"] == file_sha256(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        validate_candidate_model_selection(selection, checkpoint, config)
