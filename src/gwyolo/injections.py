from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .cosmology import FlatLambdaCDMGrid
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256
from .runtime import execution_provenance


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


def run_paired_background_remap(
    source_recipe_manifest: str | Path,
    target_background_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
    split: str = "train",
    seed: int = 20260724,
) -> dict[str, Any]:
    """Move a fixed astrophysical population onto disjoint GPS-domain backgrounds.

    Source/injection identities and all source parameters are held fixed. Only the
    background window, absolute injection time and available detector set change. This
    creates a paired data-domain arm rather than confounding GPS diversity with a newly
    sampled waveform population.
    """
    if split not in {"train", "val"}:
        raise ValueError("paired background remap supports train or val only")
    source_path = Path(source_recipe_manifest)
    with source_path.open("r", encoding="utf-8") as handle:
        all_source_rows = [json.loads(line) for line in handle if line.strip()]
    if any(str(row.get("split")) == "test" for row in all_source_rows):
        raise ValueError("paired background remap refuses source manifests containing test rows")
    source_rows = [row for row in all_source_rows if str(row.get("split")) == split]
    if not source_rows:
        raise ValueError(f"source recipe manifest contains no {split} rows")

    validation_path = Path(validation_manifest)
    with validation_path.open("r", encoding="utf-8") as handle:
        all_validation_rows = [json.loads(line) for line in handle if line.strip()]
    if any(str(row.get("split")) == "test" for row in all_validation_rows):
        raise ValueError("paired background remap refuses validation manifests with test rows")
    validation_rows = [
        row for row in all_validation_rows if str(row.get("split")) == "val"
    ]
    if not validation_rows:
        raise ValueError("paired background remap requires a non-empty validation split")

    source_ids = [str(row["injection_id"]) for row in source_rows]
    source_waveforms = [str(row["waveform_id"]) for row in source_rows]
    if len(set(source_ids)) != len(source_ids) or len(set(source_waveforms)) != len(
        source_waveforms
    ):
        raise ValueError("source recipes repeat injection or waveform identities")
    validation_ids = {str(row["injection_id"]) for row in validation_rows}
    validation_waveforms = {str(row["waveform_id"]) for row in validation_rows}
    validation_blocks = {str(row["gps_block"]) for row in validation_rows}
    if set(source_ids) & validation_ids or set(source_waveforms) & validation_waveforms:
        raise ValueError("source recipe identities overlap the comparison validation split")

    source_blocks = {str(row["gps_block"]) for row in source_rows}
    background_path = Path(target_background_manifest)
    target_rows = []
    with background_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            block = str(row.get("gps_block"))
            if (
                str(row.get("split")) == split
                and block not in source_blocks
                and block not in validation_blocks
            ):
                target_rows.append(row)
    if not target_rows:
        raise ValueError("target background has no new split-safe GPS windows")
    target_window_ids = [str(row["window_id"]) for row in target_rows]
    if len(set(target_window_ids)) != len(target_window_ids):
        raise ValueError("target background repeats window identities")

    windows_by_block: dict[str, list[dict[str, Any]]] = {}
    for row in target_rows:
        block = str(row["gps_block"])
        windows_by_block.setdefault(block, []).append(row)
    blocks = sorted(
        windows_by_block,
        key=lambda block: canonical_hash(
            {"seed": seed, "gps_block": block, "purpose": "paired_background_remap"},
            64,
        ),
    )
    for block, rows in windows_by_block.items():
        rows.sort(
            key=lambda row: canonical_hash(
                {"seed": seed, "gps_block": block, "window_id": row["window_id"]},
                64,
            )
        )

    remapped = []
    allowed_changes = {
        "background_window_id",
        "gps_block",
        "gps_time",
        "ifos",
        "background_remap",
    }
    for index, source in enumerate(source_rows):
        block = blocks[index % len(blocks)]
        block_cycle = index // len(blocks)
        candidates = windows_by_block[block]
        window = candidates[block_cycle % len(candidates)]
        start = float(window["gps_start"])
        end = float(window["gps_end"])
        if end - start <= 2.0:
            raise ValueError(f"target window is too short for an injection margin: {window}")
        fraction = int(
            canonical_hash(
                {
                    "seed": seed,
                    "injection_id": source["injection_id"],
                    "window_id": window["window_id"],
                },
                16,
            ),
            16,
        ) / float(16**16 - 1)
        updated = {
            **source,
            "background_window_id": str(window["window_id"]),
            "gps_block": block,
            "gps_time": start + 1.0 + fraction * (end - start - 2.0),
            "ifos": list(window["ifos"]),
            "background_remap": {
                "protocol": "paired_source_population_new_disjoint_gps_v1",
                "seed": seed,
                "source_background_window_id": str(source["background_window_id"]),
                "source_gps_block": str(source["gps_block"]),
                "source_gps_time": float(source["gps_time"]),
                "source_ifos": list(source["ifos"]),
            },
        }
        changed = {
            key
            for key in set(source) | set(updated)
            if source.get(key) != updated.get(key)
        }
        if not changed <= allowed_changes:
            raise RuntimeError(
                f"paired background remap changed source physics fields: {sorted(changed)}"
            )
        remapped.append(updated)

    remapped_blocks = {str(row["gps_block"]) for row in remapped}
    if remapped_blocks & source_blocks or remapped_blocks & validation_blocks:
        raise RuntimeError("paired background remap produced a GPS group overlap")
    physics_fields = sorted(
        set().union(*(set(row) for row in source_rows)) - allowed_changes
    )
    source_physics = [
        {field: row.get(field) for field in physics_fields} for row in source_rows
    ]
    remapped_physics = [
        {field: row.get(field) for field in physics_fields} for row in remapped
    ]
    source_physics_hash = canonical_hash(source_physics, 64)
    remapped_physics_hash = canonical_hash(remapped_physics, 64)
    if source_physics_hash != remapped_physics_hash:
        raise RuntimeError("paired background remap did not preserve source parameters")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"paired_background_remap_{split}.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in remapped),
    )
    report = {
        "status": "paired_source_population_remapped_to_disjoint_gps_domain",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "requires waveform materialization and fixed-update plus fixed-epoch model controls "
            "on one shared validation manifest"
        ),
        "test_recipe_rows_read": 0,
        "test_evaluation": None,
        "split": split,
        "seed": seed,
        "rows": len(remapped),
        "unique_injection_ids": len(set(source_ids)),
        "unique_waveform_ids": len(set(source_waveforms)),
        "source_unique_gps_blocks": len(source_blocks),
        "source_family_counts": dict(
            sorted(Counter(str(row["source_family"]) for row in source_rows).items())
        ),
        "target_available_windows": len(target_rows),
        "target_available_unique_gps_blocks": len(blocks),
        "remapped_unique_gps_blocks": len(remapped_blocks),
        "remapped_detector_subset_counts": dict(
            sorted(Counter("".join(row["ifos"]) for row in remapped).items())
        ),
        "source_target_gps_overlap": 0,
        "target_validation_gps_overlap": 0,
        "source_validation_injection_overlap": 0,
        "source_validation_waveform_overlap": 0,
        "source_parameters_preserved": True,
        "source_physics_hash": source_physics_hash,
        "remapped_physics_hash": remapped_physics_hash,
        "source_recipe_manifest_path": str(source_path.resolve()),
        "source_recipe_manifest_sha256": file_sha256(source_path),
        "target_background_manifest_path": str(background_path.resolve()),
        "target_background_manifest_sha256": file_sha256(background_path),
        "validation_manifest_path": str(validation_path.resolve()),
        "validation_manifest_sha256": file_sha256(validation_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "protocol": (
            "same injection and waveform identities, intrinsic/extrinsic source parameters, "
            "distance and VT weights; new disjoint GPS blocks, absolute times and detector sets"
        ),
        **execution_provenance(),
    }
    atomic_write_json(output / "paired_background_remap_report.json", report)
    return report


def audit_paired_data_domain_manifests(
    baseline_manifest: str | Path,
    independent_gps_manifest: str | Path,
    validation_manifest: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Prove that two materialized train arms differ by data domain, not population."""
    paths = {
        "baseline_fixed_gps": Path(baseline_manifest),
        "independent_gps": Path(independent_gps_manifest),
        "shared_validation": Path(validation_manifest),
    }
    loaded = {}
    for label, path in paths.items():
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            raise ValueError(f"paired data-domain audit received an empty {label} manifest")
        if any(str(row.get("split")) == "test" for row in rows):
            raise ValueError("paired data-domain audit refuses manifests containing test rows")
        expected = "val" if label == "shared_validation" else "train"
        selected = [row for row in rows if str(row.get("split")) == expected]
        if len(selected) != len(rows):
            raise ValueError(f"{label} manifest must be {expected}-only")
        loaded[label] = selected

    baseline = loaded["baseline_fixed_gps"]
    diverse = loaded["independent_gps"]
    baseline_by_id = {str(row["injection_id"]): row for row in baseline}
    diverse_by_id = {str(row["injection_id"]): row for row in diverse}
    if len(baseline_by_id) != len(baseline) or len(diverse_by_id) != len(diverse):
        raise ValueError("paired data-domain train manifests repeat injection identities")
    if set(baseline_by_id) != set(diverse_by_id):
        raise ValueError("paired data-domain arms do not share injection identities")

    source_fields = (
        "waveform_id",
        "source_family",
        "waveform_backend",
        "waveform_approximant",
        "f_lower_hz",
        "mass_1_msun",
        "mass_2_msun",
        "mass_1_detector_msun",
        "mass_2_detector_msun",
        "spin_1z",
        "spin_2z",
        "lambda_1",
        "lambda_2",
        "inclination",
        "right_ascension",
        "declination",
        "polarization",
        "coalescence_phase",
        "luminosity_distance_mpc",
        "comoving_distance_mpc",
        "redshift",
        "maximum_distance_mpc",
        "vt_weight",
        "vt_weight_unit",
        "vt_measure",
    )
    missing = {
        label: sorted(
            field
            for field in source_fields
            if any(field not in row for row in loaded[label])
        )
        for label in ("baseline_fixed_gps", "independent_gps")
    }
    missing = {label: fields for label, fields in missing.items() if fields}
    if missing:
        raise ValueError(f"paired data-domain manifests lack source fields: {missing}")
    mismatches = []
    paired_population = []
    for injection_id in sorted(baseline_by_id):
        left = baseline_by_id[injection_id]
        right = diverse_by_id[injection_id]
        changed = [field for field in source_fields if left[field] != right[field]]
        if changed:
            mismatches.append({"injection_id": injection_id, "fields": changed})
            if len(mismatches) >= 20:
                break
        paired_population.append(
            {
                "injection_id": injection_id,
                **{field: left[field] for field in source_fields},
            }
        )
    if mismatches:
        raise ValueError(f"paired data-domain source population changed: {mismatches}")

    identities = {
        label: {
            "injection_id": {str(row["injection_id"]) for row in rows},
            "waveform_id": {str(row["waveform_id"]) for row in rows},
            "gps_block": {str(row["gps_block"]) for row in rows},
        }
        for label, rows in loaded.items()
    }
    validation_overlaps = {}
    for label in ("baseline_fixed_gps", "independent_gps"):
        validation_overlaps[label] = {
            field: len(identities[label][field] & identities["shared_validation"][field])
            for field in identities[label]
        }
    if any(value for fields in validation_overlaps.values() for value in fields.values()):
        raise ValueError(
            f"paired data-domain train/validation group overlap: {validation_overlaps}"
        )
    train_gps_overlap = len(
        identities["baseline_fixed_gps"]["gps_block"]
        & identities["independent_gps"]["gps_block"]
    )
    if train_gps_overlap:
        raise ValueError("paired data-domain train arms are not GPS independent")

    baseline_blocks = len(identities["baseline_fixed_gps"]["gps_block"])
    diverse_blocks = len(identities["independent_gps"]["gps_block"])
    report = {
        "status": "paired_materialized_data_domain_audit_passed",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "requires controlled fixed-update and fixed-epoch validation comparisons"
        ),
        "test_rows_read": 0,
        "test_evaluation": None,
        "rows_per_train_arm": len(baseline),
        "unique_injections_per_train_arm": len(baseline_by_id),
        "unique_waveforms_per_train_arm": len(
            identities["baseline_fixed_gps"]["waveform_id"]
        ),
        "paired_population_hash": canonical_hash(paired_population, 64),
        "source_parameters_identical": True,
        "baseline_unique_gps_blocks": baseline_blocks,
        "independent_unique_gps_blocks": diverse_blocks,
        "independent_gps_diversity_factor": diverse_blocks / baseline_blocks,
        "cross_arm_gps_block_overlap": train_gps_overlap,
        "validation_group_overlaps": validation_overlaps,
        "manifests": {
            label: {
                "path": str(paths[label].resolve()),
                "sha256": file_sha256(paths[label]),
                "rows": len(loaded[label]),
                "unique_gps_blocks": len(identities[label]["gps_block"]),
                "source_family_counts": dict(
                    sorted(
                        Counter(str(row["source_family"]) for row in loaded[label]).items()
                    )
                ),
                "detector_subset_counts": dict(
                    sorted(Counter("".join(row["ifos"]) for row in loaded[label]).items())
                ),
            }
            for label in paths
        },
        **execution_provenance(),
    }
    atomic_write_json(output_path, report)
    return report


def run_nested_injection_scale_plan(
    base_recipe_manifest: str | Path,
    background_manifest: str | Path,
    background_report: str | Path,
    output_dir: str | Path,
    scales: tuple[int, ...] = (10_000, 25_000, 50_000),
    supplement_seed: int = 20260722,
) -> dict[str, Any]:
    """Extend a frozen train core into family-stratified nested physical scales."""
    ordered_scales = tuple(sorted(set(int(value) for value in scales)))
    if not ordered_scales or any(value <= 0 for value in ordered_scales):
        raise ValueError("nested injection scales must be positive")
    with Path(base_recipe_manifest).open("r", encoding="utf-8") as handle:
        base_rows = [json.loads(line) for line in handle if line.strip()]
    if not base_rows:
        raise ValueError("base injection recipe manifest cannot be empty")
    if any(row.get("split") == "test" for row in base_rows):
        raise ValueError("nested training scale planning refuses to read test recipes")
    base_train = [row for row in base_rows if row.get("split") == "train"]
    validation = [row for row in base_rows if row.get("split") == "val"]
    if len(base_train) != ordered_scales[0] or not validation:
        raise ValueError(
            "smallest scale must equal the frozen base train count and validation must be non-empty"
        )
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        background_rows = [json.loads(line) for line in handle if line.strip()]
    with Path(background_report).open("r", encoding="utf-8") as handle:
        background_exposure = json.load(handle)
    expected_base = _allocate(ordered_scales[0], DEFAULT_POPULATION)
    base_family_counts = Counter(str(row["source_family"]) for row in base_train)
    if dict(base_family_counts) != expected_base:
        raise ValueError(
            f"base family counts do not match the frozen population: {base_family_counts}"
        )
    supplement_count = ordered_scales[-1] - ordered_scales[0]
    supplement: list[dict[str, Any]] = []
    supplement_report = None
    if supplement_count:
        supplement, supplement_report = plan_injection_recipes(
            background_rows,
            {"train": float(background_exposure["splits"]["train"]["live_time_years"])},
            {"train": supplement_count},
            seed=supplement_seed,
        )
    base_ids = {str(row["injection_id"]) for row in base_rows}
    collisions = sorted(base_ids & {str(row["injection_id"]) for row in supplement})
    if collisions:
        raise ValueError(f"supplemental injection IDs collide with the frozen core: {collisions[:10]}")
    supplement_by_family = {
        family: sorted(
            [row for row in supplement if row["source_family"] == family],
            key=lambda row: canonical_hash(
                {"seed": supplement_seed, "injection_id": row["injection_id"]}, 64
            ),
        )
        for family in DEFAULT_POPULATION
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "injection_validation_frozen.jsonl"
    atomic_write_text(
        validation_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in validation),
    )
    validation_ids = {str(row["injection_id"]) for row in validation}
    validation_waveforms = {str(row["waveform_id"]) for row in validation}
    validation_blocks = {str(row["gps_block"]) for row in validation}
    scale_reports = []
    prior_ids: set[str] = set()
    for scale in ordered_scales:
        target_counts = _allocate(scale, DEFAULT_POPULATION)
        selected = list(base_train)
        for family in DEFAULT_POPULATION:
            needed = target_counts[family] - base_family_counts[family]
            if needed < 0 or len(supplement_by_family[family]) < needed:
                raise ValueError(f"insufficient supplemental {family} recipes for scale {scale}")
            selected.extend(supplement_by_family[family][:needed])
        selected.sort(key=lambda row: str(row["injection_id"]))
        ids = {str(row["injection_id"]) for row in selected}
        waveforms = {str(row["waveform_id"]) for row in selected}
        blocks = {str(row["gps_block"]) for row in selected}
        if len(ids) != scale or len(waveforms) != scale:
            raise ValueError(f"nested scale {scale} contains duplicate physical identities")
        if ids & validation_ids or waveforms & validation_waveforms or blocks & validation_blocks:
            raise ValueError(f"nested scale {scale} overlaps frozen validation identities")
        if prior_ids and not prior_ids < ids:
            raise ValueError(f"nested scale {scale} does not strictly contain the previous scale")
        manifest_path = output / f"injection_train_{scale}.jsonl"
        atomic_write_text(
            manifest_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        )
        increment = [row for row in selected if str(row["injection_id"]) not in prior_ids]
        increment_path = output / f"injection_train_increment_to_{scale}.jsonl"
        atomic_write_text(
            increment_path,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in increment),
        )
        scale_reports.append(
            {
                "scale": scale,
                "rows": len(selected),
                "unique_injection_ids": len(ids),
                "unique_waveform_ids": len(waveforms),
                "unique_gps_blocks": len(blocks),
                "family_counts": dict(
                    sorted(Counter(str(row["source_family"]) for row in selected).items())
                ),
                "manifest_path": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "increment_rows": len(increment),
                "increment_manifest_path": str(increment_path),
                "increment_manifest_sha256": file_sha256(increment_path),
                "contains_previous_scale": not prior_ids or prior_ids < ids,
                "validation_injection_overlap": 0,
                "validation_waveform_overlap": 0,
                "validation_gps_block_overlap": 0,
            }
        )
        prior_ids = ids
    gps_counts = [int(item["unique_gps_blocks"]) for item in scale_reports]
    result = {
        "status": "nested_physical_training_scale_plan",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "recipes require waveform validation/materialization/SNR annotation; current O4a "
            "background GPS diversity is fixed and must expand independently"
        ),
        "test_recipes_read": 0,
        "test_evaluation": None,
        "base_recipe_manifest_path": str(base_recipe_manifest),
        "base_recipe_manifest_sha256": file_sha256(base_recipe_manifest),
        "background_manifest_sha256": file_sha256(background_manifest),
        "background_report_sha256": file_sha256(background_report),
        "supplement_seed": supplement_seed,
        "supplement_rows": len(supplement),
        "supplement_plan_hash": (
            canonical_hash(supplement_report, 64) if supplement_report is not None else None
        ),
        "scales": scale_reports,
        "strictly_nested": all(
            bool(item["contains_previous_scale"]) for item in scale_reports
        ),
        "gps_diversity_counts": gps_counts,
        "gps_diversity_increases_with_scale": all(
            right > left for left, right in zip(gps_counts, gps_counts[1:])
        ),
        "gps_diversity_saturated": len(set(gps_counts)) == 1,
        "validation": {
            "rows": len(validation),
            "unique_injection_ids": len(validation_ids),
            "unique_waveform_ids": len(validation_waveforms),
            "unique_gps_blocks": len(validation_blocks),
            "manifest_path": str(validation_path),
            "manifest_sha256": file_sha256(validation_path),
        },
    }
    atomic_write_json(output / "nested_injection_scale_report.json", result)
    return result
