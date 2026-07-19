from __future__ import annotations

import numpy as np

from gwyolo.factory import (
    _allocate_counts,
    multiresolution_power,
    plan_recipes,
    run_data_factory,
    synthesize_scene,
)
from gwyolo.io import load_yaml
from gwyolo.provenance import audit_provenance


def test_count_allocation_preserves_total() -> None:
    counts = _allocate_counts(17, {"chirp_only": 0.25, "noise_only": 0.25, "overlap": 0.4, "quiet": 0.1})
    assert sum(counts.values()) == 17


def test_pilot_recipe_plan_is_deterministic_and_leak_free() -> None:
    config = load_yaml("configs/data_factory_pilot.yaml")
    first = plan_recipes(config)
    second = plan_recipes(config)
    assert [item.scene_id for item in first] == [item.scene_id for item in second]
    assert len(first) == 104
    assert audit_provenance(first)["passed"]


def test_multiresolution_tensor_shape() -> None:
    strain = np.zeros((2, 256), dtype=np.float64)
    strain[:, 100] = 1.0
    result = multiresolution_power(strain, 256, (4.0, 8.0), 20, 24, 10.0, 100.0)
    assert result.shape == (2, 2, 20, 24)
    assert np.isfinite(result).all()


def test_overlap_scene_has_both_nonempty_masks() -> None:
    recipe = next(item for item in plan_recipes(load_yaml("configs/data_factory_pilot.yaml")) if item.scene_type == "overlap")
    arrays = synthesize_scene(
        recipe,
        {"frequency_bins": 24, "time_bins": 24, "fmin": 16, "fmax": 400},
    )
    assert arrays["features"].shape == (3, 3, 24, 24)
    assert arrays["chirp_mask"].sum() > 0
    assert arrays["glitch_mask"].sum() > 0
    assert np.isfinite(arrays["features"]).all()


def test_recipe_only_factory_does_not_materialize_tensors(tmp_path) -> None:
    report = run_data_factory("configs/data_factory_research.yaml", tmp_path, limit=12)
    assert report["planned_scenes"] == 12
    assert report["generated_scenes"] == 0
    assert report["materialization"] == "recipe_only"
    assert report["storage"]["materialized_bytes"] == 0
    assert report["provenance_audit"]["passed"]
