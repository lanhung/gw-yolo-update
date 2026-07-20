from __future__ import annotations

import numpy as np
import pytest

from gwyolo.pe import (
    PUBLICATION_PROVENANCE_FIELDS,
    evaluate_pe_rows,
    evaluate_pe_robustness_rows,
    posterior_truth_metrics,
)


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
    report = evaluate_pe_rows(rows, credible_level=0.8, bootstrap_replicates=20)
    comparison = report["comparisons"][0]
    assert comparison["parameters"]["mass"]["absolute_bias_change_cleaned_minus_raw"] == -1
    assert comparison["cleaning_latency_overhead_seconds"] == 0.5
    assert report["coverage"]["DINGO"]["cleaned"]["mass"]["rate"] == 1.0
    summary = report["paired_summaries"]["DINGO"]
    assert summary["parameters"]["mass"][
        "absolute_bias_change_cleaned_minus_raw"
    ]["paired_bootstrap_95"] == [-1.0, -1.0]
    assert summary["cleaning_latency_overhead_seconds"]["mean"] == 0.5
    assert summary["parameters"]["mass"]["coverage_transitions"] == {"0->1": 1}


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


def test_pe_publication_gate_requires_and_matches_provenance(tmp_path) -> None:
    raw = tmp_path / "raw.npz"
    cleaned = tmp_path / "cleaned.npz"
    np.savez(raw, mass=np.asarray([1, 2, 3]))
    np.savez(cleaned, mass=np.asarray([1, 2, 3]))
    provenance = {field: f"fixed-{field}" for field in PUBLICATION_PROVENANCE_FIELDS}
    provenance["detector_set"] = ["H1", "L1"]
    base = {
        "backend": "AMPLFI",
        "injection_id": "i-1",
        "latency_seconds": 1.0,
        "truth": {"mass": 2.0},
        **provenance,
    }
    rows = [
        {**base, "condition": "raw", "posterior_path": str(raw)},
        {**base, "condition": "cleaned", "posterior_path": str(cleaned)},
    ]
    report = evaluate_pe_rows(
        rows,
        bootstrap_replicates=20,
        require_publication_provenance=True,
    )
    assert report["publication_provenance_required"]

    rows[1]["prior_hash"] = "different"
    with pytest.raises(ValueError, match="publication provenance mismatch"):
        evaluate_pe_rows(
            rows,
            bootstrap_replicates=20,
            require_publication_provenance=True,
        )


def test_pe_rejects_invalid_latency(tmp_path) -> None:
    posterior = tmp_path / "posterior.npz"
    np.savez(posterior, mass=np.asarray([1, 2, 3]))
    rows = [
        {
            "backend": "DINGO",
            "injection_id": "i-1",
            "condition": condition,
            "posterior_path": str(posterior),
            "latency_seconds": -1.0,
            "truth": {"mass": 2.0},
        }
        for condition in ("raw", "cleaned")
    ]
    with pytest.raises(ValueError, match="Invalid PE latency"):
        evaluate_pe_rows(rows, bootstrap_replicates=20)


def test_pe_robustness_triplet_recovers_hand_calculated_contamination(tmp_path) -> None:
    samples = {
        "clean": np.asarray([1.0, 2.0, 3.0]),
        "contaminated": np.asarray([2.0, 3.0, 4.0]),
        "mask_conditioned": np.asarray([1.0, 2.0, 3.0]),
    }
    latency = {"clean": 1.0, "contaminated": 2.0, "mask_conditioned": 2.5}
    ess = {"clean": 3.0, "contaminated": 2.0, "mask_conditioned": 3.0}
    sky = {"clean": 10.0, "contaminated": 30.0, "mask_conditioned": 12.0}
    rows = []
    for condition, values in samples.items():
        path = tmp_path / f"{condition}.npz"
        np.savez(path, mass=values)
        rows.append(
            {
                "backend": "DINGO",
                "injection_id": "i-1",
                "condition": condition,
                "posterior_path": str(path),
                "latency_seconds": latency[condition],
                "effective_sample_size": ess[condition],
                "sky_area_90_deg2": sky[condition],
                "truth": {"mass": 2.0},
            }
        )
    report = evaluate_pe_robustness_rows(
        rows,
        credible_level=0.8,
        bootstrap_replicates=20,
        require_publication_provenance=False,
    )
    comparison = report["comparisons"][0]
    parameter = comparison["parameters"]["mass"]
    assert parameter["contamination_absolute_bias_change"] == 1.0
    assert parameter["mask_absolute_bias_change_vs_contaminated"] == -1.0
    assert parameter["coverage_transition"] == "1->0->1"
    assert comparison["sky_area_contaminated_over_clean"] == 3.0
    assert comparison["sky_area_mask_over_contaminated"] == 0.4
    assert comparison["ess_rate_mask_over_contaminated"] == pytest.approx(1.2)
    assert comparison["latency_mask_minus_contaminated_seconds"] == 0.5
