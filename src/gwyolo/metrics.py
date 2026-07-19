from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    accuracy = ratio(tp + tn, tp + fp + fn + tn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - margin, center + margin


def snr_binned_hit_rate(
    records: Iterable[dict[str, Any]],
    bins: tuple[float, ...] = (0.0, 8.0, 10.0, 12.0, 15.0, 20.0, float("inf")),
) -> list[dict[str, Any]]:
    rows = list(records)
    output: list[dict[str, Any]] = []
    for low, high in zip(bins, bins[1:]):
        selected = [row for row in rows if row.get("snr") is not None and low <= row["snr"] < high]
        hits = sum(bool(row.get("hit")) for row in selected)
        lower, upper = wilson_interval(hits, len(selected))
        output.append(
            {
                "snr_min": low,
                "snr_max": None if math.isinf(high) else high,
                "hits": hits,
                "total": len(selected),
                "rate": hits / len(selected) if selected else None,
                "wilson_95": [lower, upper] if selected else [None, None],
            }
        )
    return output
