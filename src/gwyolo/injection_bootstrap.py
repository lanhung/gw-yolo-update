from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


PHYSICAL_INJECTION_BOOTSTRAP_METHOD = "gps_block_then_paired_injection_hierarchical_bootstrap_v1"
LEGACY_ROW_BOOTSTRAP_METHOD = "row_iid_legacy_fallback_v1"


def _identity_audit(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row.get(field) for row in rows]
    complete = all(value not in (None, "") for value in values)
    strings = [str(value) for value in values] if complete else []
    return {
        "field": field,
        "complete": complete,
        "unique": complete and len(strings) == len(set(strings)),
        "unique_values": len(set(strings)) if complete else 0,
    }


def hierarchical_injection_bootstrap(
    rows: Iterable[dict[str, Any]],
    numerators: Iterable[float],
    denominators: Iterable[float],
    bootstrap_replicates: int,
    seed: int,
    *,
    output_scale: float = 1.0,
    require_physical_groups: bool = False,
    minimum_physical_groups: int = 2,
) -> dict[str, Any]:
    """Bootstrap a paired weighted ratio using nested physical noise groups.

    GPS blocks are the outer independent units.  Injection/waveform rows are
    resampled within every selected block, preserving pairing between methods.
    A row-IID fallback is retained only for legacy diagnostics and is explicitly
    marked in the returned audit so it cannot enter publication evidence.
    """

    records = list(rows)
    numerator = np.asarray(list(numerators), dtype=np.float64)
    denominator = np.asarray(list(denominators), dtype=np.float64)
    if not records or len(numerator) != len(records) or len(denominator) != len(records):
        raise ValueError("hierarchical injection bootstrap arrays must align")
    if bootstrap_replicates <= 0 or minimum_physical_groups < 1:
        raise ValueError("hierarchical injection bootstrap settings are invalid")
    if (
        not math.isfinite(output_scale)
        or output_scale <= 0
        or np.any(~np.isfinite(numerator))
        or np.any(~np.isfinite(denominator))
        or np.any(denominator < 0)
        or float(denominator.sum()) <= 0
    ):
        raise ValueError("hierarchical injection bootstrap weights are invalid")

    injection_identity = _identity_audit(records, "injection_id")
    waveform_identity = _identity_audit(records, "waveform_id")
    gps_complete = all(row.get("gps_block") not in (None, "") for row in records)
    if require_physical_groups and (
        not gps_complete or not injection_identity["unique"] or not waveform_identity["unique"]
    ):
        raise ValueError(
            "publication injection bootstrap requires unique injection/waveform IDs "
            "and explicit GPS blocks"
        )

    if gps_complete:
        group_values = [str(row["gps_block"]) for row in records]
        group_field = "gps_block"
        method = PHYSICAL_INJECTION_BOOTSTRAP_METHOD
        physical_groups = True
    else:
        group_values = [f"legacy-row-{index}" for index in range(len(records))]
        group_field = "row_index"
        method = LEGACY_ROW_BOOTSTRAP_METHOD
        physical_groups = False

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_values):
        grouped[group].append(index)
    ordered_groups = sorted(grouped)
    group_indices = [np.asarray(grouped[group], dtype=np.int64) for group in ordered_groups]
    group_count = len(group_indices)
    if require_physical_groups and group_count < minimum_physical_groups:
        raise ValueError(
            f"publication injection bootstrap has {group_count} GPS blocks; "
            f"requires {minimum_physical_groups}"
        )

    group_weights = np.asarray(
        [float(denominator[indices].sum()) for indices in group_indices],
        dtype=np.float64,
    )
    total_group_weight = float(group_weights.sum())
    squared_group_weight = float(np.square(group_weights).sum())
    effective_groups = (
        total_group_weight * total_group_weight / squared_group_weight
        if squared_group_weight > 0
        else 0.0
    )
    maximum_group_weight_fraction = float(group_weights.max() / total_group_weight)

    rng = np.random.default_rng(seed)
    estimates = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        selected_groups = rng.integers(0, group_count, size=group_count)
        sampled_numerator = 0.0
        sampled_denominator = 0.0
        for group_position in selected_groups:
            indices = group_indices[int(group_position)]
            selected_rows = indices[rng.integers(0, len(indices), size=len(indices))]
            sampled_numerator += float(numerator[selected_rows].sum())
            sampled_denominator += float(denominator[selected_rows].sum())
        if sampled_denominator <= 0:
            raise ValueError("hierarchical injection bootstrap sampled zero total weight")
        estimates[replicate] = sampled_numerator / sampled_denominator * output_scale

    gates = {
        "explicit_gps_blocks": gps_complete,
        "unique_injection_ids": bool(injection_identity["unique"]),
        "unique_waveform_ids": bool(waveform_identity["unique"]),
        "minimum_physical_groups": physical_groups and group_count >= minimum_physical_groups,
        "bootstrap_complete": bool(np.all(np.isfinite(estimates))),
    }
    return {
        "interval_95": [
            float(np.percentile(estimates, 2.5)),
            float(np.percentile(estimates, 97.5)),
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "independence_audit": {
            "status": "injection_bootstrap_independence_audit_v1",
            "passed": all(gates.values()),
            "method": method,
            "physical_group_field": group_field,
            "rows": len(records),
            "physical_groups": group_count if physical_groups else 0,
            "minimum_physical_groups": minimum_physical_groups,
            "effective_physical_groups_by_vt_weight": effective_groups,
            "maximum_rows_per_physical_group": max(map(len, group_indices)),
            "maximum_vt_weight_fraction_per_physical_group": (maximum_group_weight_fraction),
            "injection_identity": injection_identity,
            "waveform_identity": waveform_identity,
            "gates": gates,
        },
    }
