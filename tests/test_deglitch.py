from __future__ import annotations

import numpy as np

from gwyolo.deglitch import deglitch_metrics, mask_deglitch, summarize_deglitch_rows


def test_zero_mask_preserves_full_signal() -> None:
    rng = np.random.default_rng(2)
    strain = rng.normal(size=(1, 1024)).astype(np.float32)
    masks = np.zeros((1, 2, 16, 16), dtype=np.float32)
    cleaned, report = mask_deglitch(strain, 1024, masks, masks, window_size=128, hop_size=32)
    assert np.allclose(cleaned, strain, atol=1e-5)
    assert report["removed_tf_energy_fraction_by_ifo"][0] == 0.0


def test_chirp_protection_blocks_oracle_glitch_suppression() -> None:
    time = np.arange(1024) / 1024
    strain = np.sin(2 * np.pi * 80 * time)[None, :].astype(np.float32)
    glitch = np.ones((1, 1, 16, 16), dtype=np.float32)
    protected = np.ones_like(glitch)
    unprotected = np.zeros_like(glitch)
    cleaned_protected, _ = mask_deglitch(
        strain, 1024, protected, glitch, strength=1, window_size=128, hop_size=32
    )
    cleaned_unprotected, _ = mask_deglitch(
        strain, 1024, unprotected, glitch, strength=1, window_size=128, hop_size=32
    )
    assert np.std(cleaned_protected) > 0.6
    assert np.std(cleaned_unprotected) < 1e-5


def test_deglitch_metrics_match_hand_calculation() -> None:
    mixture = np.asarray([[3.0, 1.0]])
    clean = np.asarray([[1.0, 1.0]])
    cleaned = np.asarray([[2.0, 1.0]])
    chirp = np.asarray([[1.0, 0.0]])
    result = deglitch_metrics(mixture, cleaned, clean, chirp)
    assert result["mse_to_clean_before"] == 2.0
    assert result["mse_to_clean_after"] == 0.5
    assert result["mse_reduction_fraction"] == 0.75
    assert result["chirp_projection_before"] == 3.0
    assert result["chirp_projection_after"] == 2.0


def test_deglitch_summary_reports_distribution() -> None:
    rows = [
        {
            "scene_type": "overlap",
            "metrics": {
                "mse_reduction_fraction": 0.5,
                "chirp_projection_retention": 0.99,
                "waveform_change_rms": 0.2,
            },
        },
        {
            "scene_type": "overlap",
            "metrics": {
                "mse_reduction_fraction": 0.7,
                "chirp_projection_retention": 1.0,
                "waveform_change_rms": 0.4,
            },
        },
    ]
    result = summarize_deglitch_rows(rows)
    assert result["overlap"]["scenes"] == 2
    assert result["overlap"]["mse_reduction_fraction"]["mean"] == 0.6
    assert np.isclose(result["overlap"]["waveform_change_rms"]["median"], 0.3)
