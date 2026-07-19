from __future__ import annotations

import numpy as np
import pytest

from gwyolo.pe import evaluate_pe_rows, posterior_truth_metrics


def test_posterior_truth_metrics_match_quantiles_and_bias() -> None:
    result = posterior_truth_metrics({"mass": np.asarray([0, 1, 2, 3, 4])}, {"mass": 2}, 0.8)
    assert result["mass"]["mean"] == 2
    assert result["mass"]["bias"] == 0
    assert result["mass"]["credible_interval"] == pytest.approx([0.4, 3.6])
    assert result["mass"]["covered"]
    assert result["mass"]["mean_absolute_distance_to_truth"] == 1.2


def test_pe_evaluation_requires_and_compares_raw_cleaned_pairs(tmp_path) -> None:
    raw = tmp_path / "raw.npz"
    cleaned = tmp_path / "cleaned.npz"
    np.savez(raw, mass=np.asarray([2, 3, 4]))
    np.savez(cleaned, mass=np.asarray([1, 2, 3]))
    rows = [
        {
            "backend": "DINGO",
            "injection_id": "i-1",
            "condition": "raw",
            "posterior_path": str(raw),
            "latency_seconds": 2.0,
            "truth": {"mass": 2.0},
        },
        {
            "backend": "DINGO",
            "injection_id": "i-1",
            "condition": "cleaned",
            "posterior_path": str(cleaned),
            "latency_seconds": 2.5,
            "truth": {"mass": 2.0},
        },
    ]
    report = evaluate_pe_rows(rows, credible_level=0.8)
    comparison = report["comparisons"][0]
    assert comparison["parameters"]["mass"]["absolute_bias_change_cleaned_minus_raw"] == -1
    assert comparison["cleaning_latency_overhead_seconds"] == 0.5
    assert report["coverage"]["DINGO"]["cleaned"]["mass"]["rate"] == 1.0


def test_pe_evaluation_rejects_missing_pair(tmp_path) -> None:
    posterior = tmp_path / "raw.npz"
    np.savez(posterior, mass=np.asarray([1, 2, 3]))
    with pytest.raises(ValueError, match="Missing raw/cleaned"):
        evaluate_pe_rows(
            [
                {
                    "backend": "AMPLFI",
                    "injection_id": "i-1",
                    "condition": "raw",
                    "posterior_path": str(posterior),
                    "latency_seconds": 1.0,
                    "truth": {"mass": 2.0},
                }
            ]
        )
