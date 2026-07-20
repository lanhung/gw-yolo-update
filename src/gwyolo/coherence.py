from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


def stack_detector_set(
    tensors_by_ifo: Mapping[str, np.ndarray], detector_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Stack a variable detector set without confusing missing IFOs with valid zero data."""
    order = tuple(str(ifo) for ifo in detector_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("detector order must be non-empty and unique")
    unknown = set(tensors_by_ifo) - set(order)
    if unknown:
        raise ValueError(f"detector tensors contain unconfigured IFOs: {sorted(unknown)}")
    if not tensors_by_ifo:
        raise ValueError("at least one detector tensor is required")

    shape = None
    dtype = None
    checked: dict[str, np.ndarray] = {}
    for ifo, value in tensors_by_ifo.items():
        array = np.asarray(value)
        if array.ndim < 2 or not np.isfinite(array).all():
            raise ValueError(f"detector tensor for {ifo} must be finite and at least 2D")
        if shape is None:
            shape = array.shape
            dtype = array.dtype
        elif array.shape != shape:
            raise ValueError("all available detector tensors must have the same shape")
        checked[ifo] = array
    assert shape is not None and dtype is not None

    availability = np.asarray([ifo in checked for ifo in order], dtype=np.float32)
    stacked = np.stack(
        [checked[ifo] if ifo in checked else np.zeros(shape, dtype=dtype) for ifo in order]
    )
    return stacked, availability


def _pair_limit(
    first: str, second: str, limits_seconds: Mapping[str, float]
) -> float:
    direct = f"{first}-{second}"
    reverse = f"{second}-{first}"
    if direct in limits_seconds:
        value = float(limits_seconds[direct])
    elif reverse in limits_seconds:
        value = float(limits_seconds[reverse])
    else:
        raise ValueError(f"missing physical delay limit for detector pair {direct}")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"physical delay limit for {direct} must be finite and positive")
    return value


def arrival_time_coherence_gate(
    arrival_times_seconds: Mapping[str, float],
    limits_seconds: Mapping[str, float],
    timing_uncertainty_seconds: float = 0.0,
) -> dict[str, Any]:
    """Check pairwise arrivals against configured light-travel limits plus uncertainty."""
    if len(arrival_times_seconds) < 2:
        raise ValueError("network coherence requires at least two detector arrivals")
    uncertainty = float(timing_uncertainty_seconds)
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("timing uncertainty must be finite and non-negative")
    arrivals = {ifo: float(value) for ifo, value in arrival_times_seconds.items()}
    if not all(np.isfinite(value) for value in arrivals.values()):
        raise ValueError("detector arrivals must be finite")

    pairs = []
    for first, second in combinations(sorted(arrivals), 2):
        physical_limit = _pair_limit(first, second, limits_seconds)
        observed = abs(arrivals[second] - arrivals[first])
        allowed = physical_limit + 2.0 * uncertainty
        pairs.append(
            {
                "pair": f"{first}-{second}",
                "observed_delay_seconds": observed,
                "physical_limit_seconds": physical_limit,
                "timing_uncertainty_seconds": uncertainty,
                "allowed_delay_seconds": allowed,
                "passed": observed <= allowed or np.isclose(
                    observed, allowed, rtol=0.0, atol=1e-12
                ),
            }
        )
    return {
        "passed": all(item["passed"] for item in pairs),
        "detectors": sorted(arrivals),
        "pairs": pairs,
        "claim": "arrival-time consistency gate, not a standalone astrophysical ranking",
    }


def pairwise_lag_coherence(
    whitened_by_ifo: Mapping[str, np.ndarray],
    sample_rate: float,
    limits_seconds: Mapping[str, float],
    timing_uncertainty_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Measure normalized pairwise correlation inside physical lag windows.

    Inputs are expected to be aligned, zero-mean whitened ROI time series. The returned features
    are suitable for a validation-only coherence head or reranker; they are not calibrated FARs.
    """
    if len(whitened_by_ifo) < 2:
        raise ValueError("pairwise lag coherence requires at least two detectors")
    rate = float(sample_rate)
    uncertainty = float(timing_uncertainty_seconds)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    if not np.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError("timing uncertainty must be finite and non-negative")

    series: dict[str, np.ndarray] = {}
    length = None
    for ifo, value in whitened_by_ifo.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
            raise ValueError(f"whitened ROI for {ifo} must be a finite 1D array")
        if length is None:
            length = array.size
        elif array.size != length:
            raise ValueError("all whitened detector ROIs must have equal length")
        series[ifo] = array
    assert length is not None

    features = []
    for first, second in combinations(sorted(series), 2):
        physical_limit = _pair_limit(first, second, limits_seconds)
        search_limit = physical_limit + 2.0 * uncertainty
        maximum_lag = min(int(np.ceil(search_limit * rate)), length - 1)
        best_lag = 0
        best_signed = 0.0
        best_absolute = -1.0
        for lag in range(-maximum_lag, maximum_lag + 1):
            if lag >= 0:
                left = series[first][: length - lag]
                right = series[second][lag:]
            else:
                left = series[first][-lag:]
                right = series[second][: length + lag]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            signed = float(np.dot(left, right) / denominator) if denominator > 0 else 0.0
            absolute = abs(signed)
            if absolute > best_absolute:
                best_lag = lag
                best_signed = signed
                best_absolute = absolute
        features.append(
            {
                "pair": f"{first}-{second}",
                "lag_samples": best_lag,
                "lag_seconds": best_lag / rate,
                "signed_coherence": best_signed,
                "absolute_coherence": best_absolute,
                "physical_limit_seconds": physical_limit,
                "search_limit_seconds": search_limit,
            }
        )
    return features
