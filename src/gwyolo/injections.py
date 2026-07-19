from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256


DEFAULT_POPULATION = {
    "BBH": {"fraction": 0.45, "maximum_distance_mpc": 5000.0},
    "BNS": {"fraction": 0.30, "maximum_distance_mpc": 500.0},
    "NSBH": {"fraction": 0.25, "maximum_distance_mpc": 1500.0},
}


def _allocate(total: int, population: dict[str, dict[str, float]]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("injection count must be positive")
    fractions = {key: float(value["fraction"]) for key, value in population.items()}
    fraction_sum = sum(fractions.values())
    if fraction_sum <= 0:
        raise ValueError("population fractions must have positive sum")
    exact = {key: total * value / fraction_sum for key, value in fractions.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    for key in sorted(exact, key=lambda item: exact[item] - counts[item], reverse=True)[
        : total - sum(counts.values())
    ]:
        counts[key] += 1
    return counts


def _source_parameters(family: str, rng: np.random.Generator) -> dict[str, float]:
    if family == "BBH":
        mass_1 = float(rng.uniform(20, 100))
        mass_2 = float(rng.uniform(5, mass_1))
    elif family == "BNS":
        mass_1 = float(rng.uniform(1.1, 2.2))
        mass_2 = float(rng.uniform(1.0, mass_1))
    elif family == "NSBH":
        mass_1 = float(rng.uniform(5, 30))
        mass_2 = float(rng.uniform(1.0, 2.5))
    else:
        raise ValueError(f"Unsupported source family: {family}")
    return {
        "mass_1_msun": mass_1,
        "mass_2_msun": mass_2,
        "spin_1z": float(rng.uniform(-0.95, 0.95)),
        "spin_2z": float(rng.uniform(-0.5, 0.5)),
        "inclination": float(np.arccos(rng.uniform(-1, 1))),
        "right_ascension": float(rng.uniform(0, 2 * np.pi)),
        "declination": float(np.arcsin(rng.uniform(-1, 1))),
        "polarization": float(rng.uniform(0, np.pi)),
    }


def plan_injection_recipes(
    background_rows: list[dict[str, Any]],
    live_time_years_by_split: dict[str, float],
    counts_by_split: dict[str, int],
    population: dict[str, dict[str, float]] | None = None,
    seed: int = 20260719,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = population or DEFAULT_POPULATION
    recipes = []
    rng = np.random.default_rng(seed)
    for split, requested_count in counts_by_split.items():
        windows = [row for row in background_rows if row["split"] == split]
        if not windows:
            raise ValueError(f"No background windows for split {split}")
        live_time = float(live_time_years_by_split[split])
        if live_time <= 0:
            raise ValueError(f"Non-positive background live time for split {split}")
        family_counts = _allocate(int(requested_count), population)
        families = [family for family in sorted(family_counts) for _ in range(family_counts[family])]
        rng.shuffle(families)
        window_order = rng.permutation(len(windows))
        per_family_weight = {}
        fraction_sum = sum(float(item["fraction"]) for item in population.values())
        for family, count in family_counts.items():
            maximum_distance = float(population[family]["maximum_distance_mpc"])
            mixture_fraction = float(population[family]["fraction"]) / fraction_sum
            survey_vt = (
                mixture_fraction * (4.0 * np.pi / 3.0) * maximum_distance**3 * live_time
            )
            per_family_weight[family] = survey_vt / count
        for index, family in enumerate(families):
            window = windows[int(window_order[index % len(window_order)])]
            maximum_distance = float(population[family]["maximum_distance_mpc"])
            distance = maximum_distance * float(rng.random()) ** (1.0 / 3.0)
            identity = f"{split}-{index:09d}-{seed}"
            recipes.append(
                {
                    "injection_id": f"injection-{identity}",
                    "waveform_id": f"waveform-{identity}",
                    "split": split,
                    "source_family": family,
                    "waveform_backend": "unassigned_requires_lal_or_validated_equivalent",
                    "background_window_id": window["window_id"],
                    "gps_block": window["gps_block"],
                    "gps_time": float(
                        rng.uniform(float(window["gps_start"]) + 1, float(window["gps_end"]) - 1)
                    ),
                    "ifos": list(window["ifos"]),
                    "luminosity_distance_mpc": distance,
                    "maximum_distance_mpc": maximum_distance,
                    "vt_weight": per_family_weight[family],
                    "vt_weight_unit": "Mpc^3 yr",
                    "seed": seed + index,
                    **_source_parameters(family, rng),
                }
            )
    injection_ids = {row["injection_id"] for row in recipes}
    waveform_ids = {row["waveform_id"] for row in recipes}
    split_ids = {
        split: {row["injection_id"] for row in recipes if row["split"] == split}
        for split in counts_by_split
    }
    overlaps = {
        f"{left}:{right}": sorted(split_ids[left] & split_ids[right])
        for left_index, left in enumerate(split_ids)
        for right in list(split_ids)[left_index + 1 :]
    }
    report = {
        "status": "injection_recipe_plan_requires_validated_waveform_backend",
        "scientific_claim_allowed": False,
        "seed": seed,
        "recipes": len(recipes),
        "unique_injection_ids": len(injection_ids),
        "unique_waveform_ids": len(waveform_ids),
        "counts_by_split": dict(sorted(Counter(row["split"] for row in recipes).items())),
        "counts_by_family": dict(sorted(Counter(row["source_family"] for row in recipes).items())),
        "unique_background_windows": len({row["background_window_id"] for row in recipes}),
        "unique_gps_blocks": len({row["gps_block"] for row in recipes}),
        "cross_split_injection_overlaps": overlaps,
        "total_vt_weight_by_split": {
            split: sum(float(row["vt_weight"]) for row in recipes if row["split"] == split)
            for split in counts_by_split
        },
        "population": population,
    }
    return recipes, report


def run_injection_plan(
    background_manifest: str | Path,
    background_report: str | Path,
    output_dir: str | Path,
    validation_count: int,
    test_count: int,
    seed: int = 20260719,
) -> dict[str, Any]:
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with Path(background_report).open("r", encoding="utf-8") as handle:
        exposure = json.load(handle)
    live_times = {
        split: float(exposure["splits"][split]["live_time_years"])
        for split in ("val", "test")
    }
    recipes, report = plan_injection_recipes(
        rows,
        live_times,
        {"val": validation_count, "test": test_count},
        seed=seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "injection_recipes.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in recipes),
    )
    result = {
        **report,
        "background_manifest_sha256": file_sha256(background_manifest),
        "background_report_sha256": file_sha256(background_report),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "plan_hash": canonical_hash(report),
    }
    atomic_write_json(output / "injection_plan_report.json", result)
    return result
