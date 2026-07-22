from __future__ import annotations

import pytest

from gwyolo.injection_bootstrap import (
    PHYSICAL_INJECTION_BOOTSTRAP_METHOD,
    hierarchical_injection_bootstrap,
)


def test_hierarchical_injection_bootstrap_audits_physical_group_weights() -> None:
    rows = [
        {"injection_id": "i1", "waveform_id": "w1", "gps_block": "g1"},
        {"injection_id": "i2", "waveform_id": "w2", "gps_block": "g1"},
        {"injection_id": "i3", "waveform_id": "w3", "gps_block": "g2"},
        {"injection_id": "i4", "waveform_id": "w4", "gps_block": "g2"},
    ]

    result = hierarchical_injection_bootstrap(
        rows,
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 2.0, 2.0],
        bootstrap_replicates=200,
        seed=7,
        require_physical_groups=True,
        minimum_physical_groups=2,
    )

    audit = result["independence_audit"]
    assert audit["passed"] is True
    assert audit["method"] == PHYSICAL_INJECTION_BOOTSTRAP_METHOD
    assert audit["physical_groups"] == 2
    assert audit["maximum_rows_per_physical_group"] == 2
    assert audit["effective_physical_groups_by_vt_weight"] == pytest.approx(1.8)
    assert audit["maximum_vt_weight_fraction_per_physical_group"] == pytest.approx(2 / 3)
    assert 0 <= result["interval_95"][0] <= result["interval_95"][1] <= 1


def test_publication_bootstrap_rejects_nonunique_physical_identity() -> None:
    rows = [
        {"injection_id": "i1", "waveform_id": "w1", "gps_block": "g1"},
        {"injection_id": "i2", "waveform_id": "w1", "gps_block": "g2"},
    ]

    with pytest.raises(ValueError, match="unique injection/waveform IDs"):
        hierarchical_injection_bootstrap(
            rows,
            [1.0, 0.0],
            [1.0, 1.0],
            bootstrap_replicates=10,
            seed=1,
            require_physical_groups=True,
            minimum_physical_groups=2,
        )
