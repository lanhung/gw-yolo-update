from __future__ import annotations

import math

import pytest

from gwyolo.multiseed import aggregate_numeric_seed_reports


def _report(seed: int, value: float) -> dict:
    return {
        "seed": seed,
        "manifest_sha256": "same",
        "config_family_hash": "same-config",
        "best_validation_mean_iou": value,
        "best_epoch": 3,
    }


def test_multiseed_mean_standard_deviation_and_student_interval_by_hand() -> None:
    report = aggregate_numeric_seed_reports([_report(1, 1.0), _report(2, 3.0)])
    metric = report["best_validation_mean_iou"]
    assert metric["mean"] == 2.0
    assert metric["sample_standard_deviation"] == pytest.approx(math.sqrt(2.0))
    assert metric["student_t_95_interval"] == pytest.approx([2 - 12.706, 2 + 12.706])
    assert not report["minimum_five_seed_gate_passed"]


def test_multiseed_rejects_duplicate_seeds_and_manifest_mismatch() -> None:
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate_numeric_seed_reports([_report(1, 1.0), _report(1, 2.0)])
    second = _report(2, 2.0)
    second["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="different manifests"):
        aggregate_numeric_seed_reports([_report(1, 1.0), second])
