from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json


EXPECTED_PROVENANCE_FIELDS = (
    "waveform_id",
    "injection_id",
    "glitch_id",
    "gps_block",
    "ifo",
    "observing_run",
    "source_family",
    "snr",
    "duration",
    "q_plane",
    "overlap_severity",
)

DEFAULT_LEARNING_CURVE = (250, 500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000)
BASELINE_FRACTIONS = {
    "chirp_only": 0.25,
    "noise_only": 0.25,
    "chirp+noise": 0.40,
    "empty": 0.10,
}
RESEARCH_FRACTIONS = {
    "chirp_only": 0.25,
    "noise_only": 0.20,
    "chirp+noise": 0.40,
    "empty": 0.15,
}


def _as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "0")
    return int(value) if value else 0


def _composition(class_0: int, class_1: int) -> str:
    if class_0 and class_1:
        return "chirp+noise"
    if class_0:
        return "chirp_only"
    if class_1:
        return "noise_only"
    return "empty"


def _target_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    counts = {key: round(total * fraction) for key, fraction in fractions.items()}
    counts["chirp+noise"] += total - sum(counts.values())
    return counts


def analyze_manifest(path: str | Path) -> dict[str, Any]:
    manifest = Path(path)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required = {"group_id", "split", "class_0", "class_1"}
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    group_sizes = Counter(len(group_rows) for group_rows in groups.values())
    split_groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for group_id, group_rows in groups.items():
        splits = {row["split"] for row in group_rows}
        if len(splits) != 1:
            raise ValueError(f"Physical group crosses splits: {group_id} -> {sorted(splits)}")
        split_groups[next(iter(splits))][group_id] = group_rows

    split_summary: dict[str, Any] = {}
    for split, grouped in sorted(split_groups.items()):
        compositions: Counter[str] = Counter()
        for group_rows in grouped.values():
            class_0 = sum(_as_int(row, "class_0") for row in group_rows)
            class_1 = sum(_as_int(row, "class_1") for row in group_rows)
            compositions[_composition(class_0, class_1)] += 1
        split_rows = [row for group_rows in grouped.values() for row in group_rows]
        split_summary[split] = {
            "images": len(split_rows),
            "physical_groups": len(grouped),
            "group_composition": dict(sorted(compositions.items())),
            "class_0_instances": sum(_as_int(row, "class_0") for row in split_rows),
            "class_1_instances": sum(_as_int(row, "class_1") for row in split_rows),
        }

    return {
        "manifest": str(manifest.resolve()),
        "images": len(rows),
        "physical_groups": len(groups),
        "images_per_group": len(rows) / len(groups) if groups else None,
        "group_size_distribution": {str(key): value for key, value in sorted(group_sizes.items())},
        "images_in_multi_image_groups": sum(
            size * count for size, count in group_sizes.items() if size > 1
        ),
        "splits": split_summary,
        "available_provenance_fields": sorted(fields & set(EXPECTED_PROVENANCE_FIELDS)),
        "missing_provenance_fields": sorted(set(EXPECTED_PROVENANCE_FIELDS) - fields),
    }


def make_scaling_plan(
    audit: dict[str, Any],
    baseline_target: int = 10_000,
    research_target: int = 200_000,
    seeds: int = 3,
) -> dict[str, Any]:
    if baseline_target <= 0 or research_target < baseline_target:
        raise ValueError("Targets must be positive and research_target >= baseline_target")
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    train = audit.get("splits", {}).get("train", {})
    current_groups = int(train.get("physical_groups", 0))
    current_composition = {
        key: int(train.get("group_composition", {}).get(key, 0)) for key in BASELINE_FRACTIONS
    }
    baseline_counts = _target_counts(baseline_target, BASELINE_FRACTIONS)
    research_counts = _target_counts(research_target, RESEARCH_FRACTIONS)
    schedule = []
    for groups in DEFAULT_LEARNING_CURVE:
        schedule.append(
            {
                "physical_groups": groups,
                "seeds": seeds,
                "available_now": groups <= current_groups,
                "additional_groups_needed": max(0, groups - current_groups),
            }
        )
    return {
        "current": {
            "training_physical_groups": current_groups,
            "training_group_composition": current_composition,
        },
        "baseline_target": {
            "target_physical_groups": baseline_target,
            "target_composition": baseline_counts,
            "gap_by_composition": {
                key: max(0, baseline_counts[key] - current_composition[key])
                for key in baseline_counts
            },
            "expansion_factor": baseline_target / current_groups if current_groups else None,
        },
        "research_target": {
            "target_physical_groups": research_target,
            "target_composition": research_counts,
            "expansion_factor": research_target / current_groups if current_groups else None,
        },
        "learning_curve": schedule,
        "evaluation_targets": {
            "validation_independent_scenes": [5_000, 10_000],
            "locked_test_injection_scenes": [20_000, 50_000],
            "minimum_examples_per_primary_stratum": [200, 500],
        },
        "promotion_blockers": {
            "missing_provenance_fields": audit.get("missing_provenance_fields", []),
            "evaluation_set_too_small": (
                int(audit.get("splits", {}).get("val", {}).get("physical_groups", 0)) < 5_000
                or int(audit.get("splits", {}).get("test", {}).get("physical_groups", 0)) < 20_000
            ),
            "learning_curve_not_yet_available": current_groups < 500,
        },
    }


def run_scale_plan(
    manifest: str | Path,
    output: str | Path,
    baseline_target: int = 10_000,
    research_target: int = 200_000,
    seeds: int = 3,
) -> dict[str, Any]:
    audit = analyze_manifest(manifest)
    result = {
        "audit": audit,
        "plan": make_scaling_plan(audit, baseline_target, research_target, seeds),
    }
    atomic_write_json(output, result)
    return result


def fit_power_law_curve(points: list[dict[str, float]]) -> dict[str, Any]:
    if len(points) < 3:
        raise ValueError("At least three learning-curve points are required")
    ordered = sorted(points, key=lambda item: float(item["physical_groups"]))
    groups = np.asarray([float(item["physical_groups"]) for item in ordered])
    metrics = np.asarray([float(item["metric"]) for item in ordered])
    if np.any(groups <= 0) or len(set(groups.tolist())) != len(groups):
        raise ValueError("physical_groups must be positive and unique")
    if not np.isfinite(metrics).all():
        raise ValueError("metrics must be finite")

    best = None
    for alpha in np.linspace(0.05, 2.0, 1951):
        scale = groups ** (-alpha)
        design = np.stack([np.ones_like(scale), scale], axis=1)
        coefficients, _, _, _ = np.linalg.lstsq(design, metrics, rcond=None)
        asymptote, slope = (float(item) for item in coefficients)
        amplitude = -slope
        if amplitude < 0 or asymptote < float(np.max(metrics)) or asymptote > 1.05:
            continue
        predicted = asymptote - amplitude * scale
        squared_error = float(np.sum((metrics - predicted) ** 2))
        if best is None or squared_error < best["squared_error"]:
            best = {
                "alpha": float(alpha),
                "asymptote": asymptote,
                "amplitude": amplitude,
                "squared_error": squared_error,
                "predicted": predicted,
            }
    if best is None:
        raise ValueError("No physically constrained power-law fit found")
    total_error = float(np.sum((metrics - np.mean(metrics)) ** 2))
    forecasts = {}
    for count in sorted({int(groups[-1] * 2), 10_000, 25_000, 50_000}):
        forecasts[str(count)] = float(
            best["asymptote"] - best["amplitude"] * count ** (-best["alpha"])
        )
    return {
        "model": "metric(N) = asymptote - amplitude * N^(-alpha)",
        "points": [
            {
                "physical_groups": int(group),
                "metric": float(metric),
                "fitted_metric": float(predicted),
                "residual": float(metric - predicted),
            }
            for group, metric, predicted in zip(groups, metrics, best["predicted"])
        ],
        "parameters": {
            "asymptote": best["asymptote"],
            "amplitude": best["amplitude"],
            "alpha": best["alpha"],
        },
        "r_squared": 1.0 - best["squared_error"] / total_error if total_error else 1.0,
        "forecasts": forecasts,
        "warning": "Exploratory extrapolation only; repeat with multiple seeds and real data.",
    }


def run_curve_fit(points_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    with Path(points_path).open("r", encoding="utf-8") as handle:
        points = json.load(handle)
    if not isinstance(points, list):
        raise ValueError("Learning-curve points file must contain a list")
    report = fit_power_law_curve(points)
    atomic_write_json(output_path, report)
    return report
