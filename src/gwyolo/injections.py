from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .cosmology import FlatLambdaCDMGrid
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256


DEFAULT_POPULATION = {
    "BBH": {
        "fraction": 0.45,
        "maximum_distance_mpc": 5000.0,
        "approximant": "IMRPhenomXAS",
        "f_lower_hz": 20.0,
    },
    "BNS": {
        "fraction": 0.30,
        "maximum_distance_mpc": 500.0,
        "approximant": "IMRPhenomXAS_NRTidalv3",
        "f_lower_hz": 20.0,
    },
    "NSBH": {
        "fraction": 0.25,
        "maximum_distance_mpc": 1500.0,
        "approximant": "IMRPhenomNSBH",
        "f_lower_hz": 20.0,
    },
}


def _allocate(total: int, population: dict[str, dict[str, Any]]) -> dict[str, int]:
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


def _source_parameters(family: str, rng: np.random.Generator) -> dict[str, Any]:
    if family == "BBH":
        mass_1 = float(rng.uniform(20, 100))
        mass_2 = float(rng.uniform(5, mass_1))
    elif family == "BNS":
        mass_1 = float(rng.uniform(1.1, 2.2))
        mass_2 = float(rng.uniform(1.0, mass_1))
    elif family == "NSBH":
        mass_1 = float(rng.uniform(5, 30))
        mass_2 = float(rng.uniform(1.0, 2.2))
    else:
        raise ValueError(f"Unsupported source family: {family}")
    if family == "BBH":
        spin_1z = float(rng.uniform(-0.95, 0.95))
        spin_2z = float(rng.uniform(-0.95, 0.95))
        lambda_1 = 0.0
        lambda_2 = 0.0
    elif family == "BNS":
        spin_1z = float(rng.uniform(-0.05, 0.05))
        spin_2z = float(rng.uniform(-0.05, 0.05))
        lambda_1 = float(np.clip(400.0 * (1.4 / mass_1) ** 6, 0.0, 5000.0))
        lambda_2 = float(np.clip(400.0 * (1.4 / mass_2) ** 6, 0.0, 5000.0))
    else:
        spin_1z = float(rng.uniform(-0.95, 0.95))
        spin_2z = float(rng.uniform(-0.05, 0.05))
        lambda_1 = 0.0
        lambda_2 = float(np.clip(400.0 * (1.4 / mass_2) ** 6, 0.0, 5000.0))
    return {
        "mass_1_msun": mass_1,
        "mass_2_msun": mass_2,
        "mass_frame": "source",
        "spin_1z": spin_1z,
        "spin_2z": spin_2z,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "tidal_proposal": "provisional_mass_scaling_not_eos_validated",
        "inclination": float(np.arccos(rng.uniform(-1, 1))),
        "right_ascension": float(rng.uniform(0, 2 * np.pi)),
        "declination": float(np.arcsin(rng.uniform(-1, 1))),
        "polarization": float(rng.uniform(0, np.pi)),
        "coalescence_phase": float(rng.uniform(0, 2 * np.pi)),
    }


def plan_injection_recipes(
    background_rows: list[dict[str, Any]],
    live_time_years_by_split: dict[str, float],
    counts_by_split: dict[str, int],
    population: dict[str, dict[str, Any]] | None = None,
    seed: int = 20260719,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = population or DEFAULT_POPULATION
    cosmology = FlatLambdaCDMGrid()
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
        family_survey = {}
        fraction_sum = sum(float(item["fraction"]) for item in population.values())
        for family, count in family_counts.items():
            if count == 0:
                continue
            maximum_distance = float(population[family]["maximum_distance_mpc"])
            maximum_redshift = float(cosmology.redshift_at_luminosity_distance(maximum_distance))
            maximum_comoving = float(cosmology.distances_at_redshift(maximum_redshift)[0])
            mixture_fraction = float(population[family]["fraction"]) / fraction_sum
            family_survey[family] = {
                "maximum_redshift": maximum_redshift,
                "maximum_comoving_distance_mpc": maximum_comoving,
                "proposal_comoving_volume_mpc3": 4.0 * np.pi / 3.0 * maximum_comoving**3,
                "base_weight": mixture_fraction
                * (4.0 * np.pi / 3.0)
                * maximum_comoving**3
                * live_time
                / count,
            }
        for index, family in enumerate(families):
            window = windows[int(window_order[index % len(window_order)])]
            maximum_distance = float(population[family]["maximum_distance_mpc"])
            survey = family_survey[family]
            comoving_distance = float(survey["maximum_comoving_distance_mpc"]) * float(
                rng.random()
            ) ** (1.0 / 3.0)
            redshift = float(cosmology.redshift_at_comoving_distance(comoving_distance))
            distance = (1.0 + redshift) * comoving_distance
            source = _source_parameters(family, rng)
            identity = f"{split}-{index:09d}-{seed}"
            recipes.append(
                {
                    "injection_id": f"injection-{identity}",
                    "waveform_id": f"waveform-{identity}",
                    "split": split,
                    "source_family": family,
                    "waveform_backend": "pycbc_lalsimulation_requires_validation",
                    "waveform_approximant": str(
                        population[family].get("approximant", "unassigned")
                    ),
                    "f_lower_hz": float(population[family].get("f_lower_hz", 20.0)),
                    "background_window_id": window["window_id"],
                    "gps_block": window["gps_block"],
                    "gps_time": float(
                        rng.uniform(float(window["gps_start"]) + 1, float(window["gps_end"]) - 1)
                    ),
                    "ifos": list(window["ifos"]),
                    "luminosity_distance_mpc": distance,
                    "comoving_distance_mpc": comoving_distance,
                    "redshift": redshift,
                    "maximum_distance_mpc": maximum_distance,
                    "vt_weight": float(survey["base_weight"]) / (1.0 + redshift),
                    "vt_weight_unit": "Mpc^3 yr",
                    "vt_measure": "comoving_volume_times_source_frame_time",
                    "seed": seed + index,
                    **source,
                    "mass_1_detector_msun": source["mass_1_msun"] * (1.0 + redshift),
                    "mass_2_detector_msun": source["mass_2_msun"] * (1.0 + redshift),
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
        "status": "cosmological_injection_recipe_plan_requires_validated_waveform_backend",
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
        "cosmology": cosmology.metadata(),
        "proposal_measure": "uniform_in_comoving_volume_with_1_over_1_plus_z_source_time_weight",
        "population_model_status": "broad_pilot_not_GWTC_population_fit",
        "approximant_domain_audit": {
            "nsbh_detector_frame_neutron_star_mass_maximum_msun": max(
                (
                    float(row["mass_2_detector_msun"])
                    for row in recipes
                    if row["source_family"] == "NSBH"
                ),
                default=None,
            ),
            "imrphenom_nsbh_neutron_star_mass_limit_msun": 3.0,
            "passed": all(
                row["source_family"] != "NSBH"
                or float(row["mass_2_detector_msun"]) <= 3.0
                for row in recipes
            ),
        },
    }
    if not report["approximant_domain_audit"]["passed"]:
        raise ValueError("Planned NSBH detector-frame masses exceed waveform approximant domain")
    return recipes, report


def run_injection_plan(
    background_manifest: str | Path,
    background_report: str | Path,
    output_dir: str | Path,
    validation_count: int,
    test_count: int,
    seed: int = 20260719,
    training_count: int = 0,
) -> dict[str, Any]:
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    with Path(background_report).open("r", encoding="utf-8") as handle:
        exposure = json.load(handle)
    requested = {"train": training_count, "val": validation_count, "test": test_count}
    if any(count < 0 for count in requested.values()) or not any(requested.values()):
        raise ValueError("Split injection counts must be non-negative with at least one positive")
    counts = {split: count for split, count in requested.items() if count > 0}
    live_times = {
        split: float(exposure["splits"][split]["live_time_years"]) for split in counts
    }
    recipes, report = plan_injection_recipes(
        rows,
        live_times,
        counts,
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
        "requested_counts_by_split": requested,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "plan_hash": canonical_hash(report),
    }
    atomic_write_json(output / "injection_plan_report.json", result)
    return result
