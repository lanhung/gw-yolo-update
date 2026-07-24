from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .cosmology import FlatLambdaCDMGrid
from .io import atomic_write_json, atomic_write_text, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance


ALLOWED_IFOS = {"H1", "L1", "V1"}
RESULT_FIELDS = {
    "candidate_score",
    "ranking_statistic",
    "posterior_samples",
    "model_prediction",
    "selected_threshold",
    "test_metric",
    "recovered",
    "far",
    "ifar",
    "strain",
    "time_series",
    "q_transform",
    "features",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL inventory is empty or invalid: {path}")
    return rows


def _allocate(total: int, settings: dict[str, dict[str, Any]]) -> dict[str, int]:
    weights = {key: float(value["fraction"]) for key, value in settings.items()}
    if total <= 0 or any(value <= 0 for value in weights.values()):
        raise ValueError("locked population counts and fractions must be positive")
    weight_sum = sum(weights.values())
    exact = {key: total * value / weight_sum for key, value in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        settings,
        key=lambda key: (exact[key] - counts[key], key),
        reverse=True,
    )
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _hash_order(values: list[str], seed: int, role: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: canonical_hash(
            {"protocol": "locked_hash_rank_v1", "role": role, "seed": seed, "id": value},
            64,
        ),
    )


def _range(settings: dict[str, Any], key: str) -> tuple[float, float]:
    values = settings.get(key)
    if (
        not isinstance(values, list)
        or len(values) != 2
        or isinstance(values[0], bool)
        or isinstance(values[1], bool)
    ):
        raise ValueError(f"locked population range is invalid: {key}")
    low, high = map(float, values)
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"locked population range is invalid: {key}")
    return low, high


def _population_sections(
    population_config: dict[str, Any], suite_settings: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    settings = population_config.get("gwtc5_locked_injection_population")
    if (
        not isinstance(settings, dict)
        or settings.get("schema") != "gwtc5_locked_injection_population_v1"
    ):
        raise ValueError("locked injection population config has another schema")
    families = settings.get("source_families")
    stress = settings.get("stress_protocols")
    required_families = suite_settings.get("endpoints", {}).get(
        "required_source_families"
    )
    required_stress = suite_settings.get("endpoints", {}).get("required_stress_strata")
    if (
        not isinstance(families, dict)
        or set(families) != set(required_families or [])
        or not isinstance(stress, dict)
        or set(stress) != set(required_stress or [])
    ):
        raise ValueError("population families/stress protocols differ from the locked suite")
    return settings, families, stress


def _source_parameters(
    family: str,
    family_settings: dict[str, Any],
    high_mass: bool,
    high_spin: bool,
    rng: random.Random,
) -> dict[str, Any]:
    zero = 0.0
    if family == "BBH":
        if high_mass:
            mass_1 = rng.uniform(*_range(family_settings, "high_mass_mass_1_msun"))
            mass_ratio = rng.uniform(*_range(family_settings, "high_mass_mass_ratio"))
            approximant = str(family_settings["high_mass_approximant"])
        else:
            mass_1 = rng.uniform(*_range(family_settings, "ordinary_mass_1_msun"))
            mass_ratio = rng.uniform(*_range(family_settings, "ordinary_mass_ratio"))
            approximant = str(
                family_settings[
                    "high_spin_approximant" if high_spin else "ordinary_approximant"
                ]
            )
        mass_2 = mass_1 * mass_ratio
        if high_spin:
            magnitude = rng.uniform(*_range(family_settings, "high_spin_magnitude"))
            tilt = rng.uniform(*_range(family_settings, "high_spin_tilt_radians"))
            azimuth = rng.uniform(0.0, 2.0 * math.pi)
            spin_1x = magnitude * math.sin(tilt) * math.cos(azimuth)
            spin_1y = magnitude * math.sin(tilt) * math.sin(azimuth)
            spin_1z = magnitude * math.cos(tilt)
        else:
            spin_1x = spin_1y = zero
            spin_1z = rng.uniform(*_range(family_settings, "aligned_spin_z"))
        spin_2x = spin_2y = zero
        spin_2z = rng.uniform(*_range(family_settings, "aligned_spin_z"))
        lambda_1 = lambda_2 = zero
    elif family == "BNS":
        mass_1 = rng.uniform(*_range(family_settings, "mass_1_msun"))
        mass_2_range = _range(family_settings, "mass_2_msun")
        mass_2 = rng.uniform(mass_2_range[0], min(mass_2_range[1], mass_1))
        approximant = str(family_settings["ordinary_approximant"])
        spin_1x = spin_1y = spin_2x = spin_2y = zero
        spin_1z = rng.uniform(*_range(family_settings, "aligned_spin_z"))
        spin_2z = rng.uniform(*_range(family_settings, "aligned_spin_z"))
        lambda_1 = float(np.clip(400.0 * (1.4 / mass_1) ** 6, 0.0, 5000.0))
        lambda_2 = float(np.clip(400.0 * (1.4 / mass_2) ** 6, 0.0, 5000.0))
    elif family == "NSBH":
        mass_1 = rng.uniform(*_range(family_settings, "black_hole_mass_msun"))
        mass_2 = rng.uniform(*_range(family_settings, "neutron_star_mass_msun"))
        approximant = str(family_settings["ordinary_approximant"])
        spin_1x = spin_1y = spin_2x = spin_2y = zero
        spin_1z = rng.uniform(*_range(family_settings, "black_hole_aligned_spin_z"))
        spin_2z = rng.uniform(*_range(family_settings, "neutron_star_aligned_spin_z"))
        lambda_1 = zero
        lambda_2 = float(np.clip(400.0 * (1.4 / mass_2) ** 6, 0.0, 5000.0))
    else:
        raise ValueError(f"unsupported locked source family: {family}")
    return {
        "mass_1_msun": mass_1,
        "mass_2_msun": mass_2,
        "mass_frame": "source",
        "mass_ratio": mass_2 / mass_1,
        "spin_1x": spin_1x,
        "spin_1y": spin_1y,
        "spin_1z": spin_1z,
        "spin_2x": spin_2x,
        "spin_2y": spin_2y,
        "spin_2z": spin_2z,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "tidal_proposal": "provisional_mass_scaling_not_eos_validated",
        "waveform_approximant": approximant,
    }


def audit_gwtc5_locked_injection_rows(
    rows: list[dict[str, Any]],
    availability_rows: list[dict[str, Any]],
    suite_settings: dict[str, Any],
    population_settings: dict[str, Any],
) -> dict[str, Any]:
    """Recompute physical strata and group identities without reading strain."""

    settings, families, stress = _population_sections(
        {"gwtc5_locked_injection_population": population_settings}, suite_settings
    )
    availability_policy = settings.get("availability_policy", {})
    minimum_planned = int(availability_policy.get("minimum_planned_injections", 0))
    if len(rows) < minimum_planned:
        raise ValueError("locked injection inventory is below its planned-attempt floor")
    availability_by_id = {
        str(row.get("availability_id", "")): row for row in availability_rows
    }
    if (
        len(availability_by_id) != len(availability_rows)
        or any(not key for key in availability_by_id)
    ):
        raise ValueError("locked availability rows have invalid identities")
    row_availability_ids = [str(row.get("availability_id", "")) for row in rows]
    if len(set(row_availability_ids)) != len(rows):
        raise ValueError("locked injections reuse an availability block")
    if availability_policy.get("use_every_frozen_gps_block_once") is not True or set(
        row_availability_ids
    ) != set(availability_by_id):
        raise ValueError("locked injections do not use every frozen GPS block exactly once")

    required_fields = {
        "injection_id",
        "waveform_id",
        "availability_id",
        "split",
        "source_family",
        "observing_run",
        "catalog_release",
        "gps_block",
        "gps_time",
        "ifos",
        "detector_subset",
        "stress_strata",
        "stress_evidence",
        "mass_1_msun",
        "mass_2_msun",
        "mass_1_detector_msun",
        "mass_2_detector_msun",
        "redshift",
        "waveform_approximant",
        "pre_access_vt_weight",
        "spin_1x",
        "spin_1y",
        "spin_1z",
        "spin_2x",
        "spin_2y",
        "spin_2z",
    }
    missing = [index for index, row in enumerate(rows) if required_fields - set(row)]
    if missing:
        raise ValueError(f"locked injections lack physical fields: {missing[:10]}")
    for field in ("injection_id", "waveform_id", "gps_block"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"locked injection inventory repeats {field}")

    required_stress = list(suite_settings["endpoints"]["required_stress_strata"])
    allowed_strata = set(required_stress) | {"nominal"}
    stress_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    detector_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        availability = availability_by_id[row_availability_ids[index]]
        ifos = [str(value) for value in row["ifos"]]
        expected_ifos = [str(value) for value in availability["available_ifos"]]
        if (
            row.get("split") != "test"
            or row.get("observing_run") != "O4b"
            or row.get("catalog_release") != "GWTC-5.0"
            or ifos != expected_ifos
            or row.get("detector_subset") != "+".join(expected_ifos)
            or row.get("gps_block") != availability.get("gps_block")
            or row.get("pre_access_vt_weight") is not None
        ):
            raise ValueError(f"locked injection identity differs from availability at row {index}")
        family = str(row["source_family"])
        if family not in families:
            raise ValueError(f"unknown locked source family at row {index}")
        family_counts[family] += 1
        detector_counts[str(row["detector_subset"])] += 1
        family_settings = families[family]
        allowed_approximants = {
            str(value)
            for key, value in family_settings.items()
            if key.endswith("_approximant")
        }
        if str(row["waveform_approximant"]) not in allowed_approximants:
            raise ValueError(f"locked waveform approximant differs at row {index}")
        context = float(row.get("required_context_duration_seconds", 0.0))
        if context != float(family_settings["context_duration_seconds"]):
            raise ValueError(f"locked context duration differs at row {index}")
        guard = float(availability_policy["coalescence_guard_seconds"])
        low = float(availability["gps_start"]) + context / 2.0 + guard
        high = float(availability["gps_end"]) - context / 2.0 - guard
        if not low <= float(row["gps_time"]) <= high:
            raise ValueError(f"locked coalescence time lacks context at row {index}")

        mass_1 = float(row["mass_1_msun"])
        mass_2 = float(row["mass_2_msun"])
        redshift = float(row["redshift"])
        if (
            not math.isfinite(mass_1 + mass_2 + redshift)
            or mass_1 < mass_2
            or mass_2 <= 0
            or redshift < 0
            or not math.isclose(
                float(row["mass_1_detector_msun"]), mass_1 * (1.0 + redshift), rel_tol=1e-12
            )
            or not math.isclose(
                float(row["mass_2_detector_msun"]), mass_2 * (1.0 + redshift), rel_tol=1e-12
            )
        ):
            raise ValueError(f"locked masses/redshift are inconsistent at row {index}")
        for prefix in ("spin_1", "spin_2"):
            magnitude = math.sqrt(
                sum(float(row[f"{prefix}{axis}"]) ** 2 for axis in ("x", "y", "z"))
            )
            if not math.isfinite(magnitude) or magnitude > 0.9900000001:
                raise ValueError(f"locked spin vector is unphysical at row {index}")

        strata = row["stress_strata"]
        evidence = row["stress_evidence"]
        if (
            not isinstance(strata, list)
            or len(strata) != len(set(strata))
            or not set(strata) <= allowed_strata
            or not isinstance(evidence, dict)
        ):
            raise ValueError(f"locked stress inventory is malformed at row {index}")
        if strata == ["nominal"]:
            if evidence:
                raise ValueError(f"nominal locked row carries stress evidence at row {index}")
        elif "nominal" in strata or set(evidence) != set(strata):
            raise ValueError(f"locked stress labels/evidence differ at row {index}")

        missing_detector = len(ifos) < 3
        if ("missing_detector" in strata) != missing_detector:
            raise ValueError(f"missing-detector label is not physical at row {index}")
        if missing_detector:
            item = evidence["missing_detector"]
            if item.get("predicate") != stress["missing_detector"]["predicate"]:
                raise ValueError(f"missing-detector evidence differs at row {index}")

        if "high_mass_unequal_mass" in strata:
            protocol = stress["high_mass_unequal_mass"]
            if (
                family != protocol["family"]
                or mass_1 < float(protocol["predicate_mass_1_minimum_msun"])
                or mass_2 / mass_1 > float(protocol["predicate_mass_ratio_maximum"])
            ):
                raise ValueError(f"high-mass unequal-mass label is not physical at row {index}")
        if "high_spin_precessing" in strata:
            protocol = stress["high_spin_precessing"]
            in_plane = math.hypot(float(row["spin_1x"]), float(row["spin_1y"]))
            total = math.sqrt(
                float(row["spin_1x"]) ** 2
                + float(row["spin_1y"]) ** 2
                + float(row["spin_1z"]) ** 2
            )
            if (
                family != protocol["family"]
                or row["waveform_approximant"] != protocol["approximant"]
                or in_plane < float(protocol["minimum_in_plane_spin"])
                or total < float(protocol["minimum_total_spin"])
            ):
                raise ValueError(f"high-spin precessing label is not physical at row {index}")
        if "waveform_systematics" in strata:
            protocol = stress["waveform_systematics"]
            expected = protocol["alternative_by_primary"].get(row["waveform_approximant"])
            item = evidence["waveform_systematics"]
            if (
                not expected
                or row.get("alternative_waveform_approximant") != expected
                or item.get("alternative_approximant") != expected
                or item.get("primary_approximant") != row["waveform_approximant"]
            ):
                raise ValueError(f"waveform-systematics evidence differs at row {index}")
        elif "alternative_waveform_approximant" in row:
            raise ValueError(f"unlabelled waveform alternative at row {index}")
        if "calibration_perturbation" in strata:
            protocol = stress["calibration_perturbation"]
            scenarios = {str(value["id"]): value for value in protocol["scenarios"]}
            item = evidence["calibration_perturbation"]
            scenario = row.get("calibration_scenario")
            if (
                not isinstance(scenario, dict)
                or scenario.get("id") not in scenarios
                or scenario != scenarios[scenario["id"]]
                or item.get("scenario") != scenario
            ):
                raise ValueError(f"calibration evidence differs at row {index}")
        elif "calibration_scenario" in row:
            raise ValueError(f"unlabelled calibration scenario at row {index}")
        if "glitch_overlap" in strata:
            protocol = stress["glitch_overlap"]
            item = evidence["glitch_overlap"]
            if (
                item.get("selector_protocol") != protocol["selector_protocol"]
                or item.get("assignment_protocol") != protocol["assignment_protocol"]
                or item.get("unavailable_policy") != protocol["unavailable_policy"]
                or item.get("auxiliary_veto_allowed") is not False
                or not item.get("assignment_key")
            ):
                raise ValueError(f"glitch-overlap evidence differs at row {index}")
        stress_counts.update(strata)

    for family, config in families.items():
        if family_counts[family] < int(config["minimum_rows"]):
            raise ValueError(f"locked family quota failed: {family}")
    detector_minima = settings.get("detector_subset_minimum_rows", {})
    for subset, minimum in detector_minima.items():
        if detector_counts[str(subset)] < int(minimum):
            raise ValueError(f"locked detector-subset quota failed: {subset}")
    for name, protocol in stress.items():
        if stress_counts[name] < int(protocol["minimum_rows"]):
            raise ValueError(f"locked stress quota failed: {name}")
    required_families = set(suite_settings["endpoints"]["required_source_families"])
    required_subsets = set(suite_settings["endpoints"]["required_detector_subsets"])
    if not (
        required_families <= set(family_counts)
        and required_subsets <= set(detector_counts)
        and set(required_stress) <= set(stress_counts)
    ):
        raise ValueError("locked injection inventory lacks a required suite stratum")
    return {
        "passed": True,
        "rows": len(rows),
        "unique_injection_ids": len(rows),
        "unique_waveform_ids": len(rows),
        "unique_gps_blocks": len(rows),
        "source_family_counts": dict(sorted(family_counts.items())),
        "detector_subset_counts": dict(sorted(detector_counts.items())),
        "stress_stratum_counts": dict(sorted(stress_counts.items())),
        "one_injection_per_frozen_gps_block": True,
        "physical_stress_predicates_passed": True,
        "pre_access_vt_weights_absent": True,
    }


def run_gwtc5_locked_injection_inventory(
    availability_manifest_path: str | Path,
    availability_report_path: str | Path,
    suite_config_path: str | Path,
    population_config_path: str | Path,
    access_log_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Bind a physical proposal population to score-blind O4b availability metadata."""

    availability_manifest = Path(availability_manifest_path).resolve()
    availability_report_file = Path(availability_report_path).resolve()
    suite_config_file = Path(suite_config_path).resolve()
    population_config_file = Path(population_config_path).resolve()
    access_log = Path(access_log_path).resolve()
    output = Path(output_dir).resolve()
    inputs = (
        availability_manifest,
        availability_report_file,
        suite_config_file,
        population_config_file,
    )
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("locked injection inventory inputs are absent")
    if access_log.exists():
        raise FileExistsError("GWTC-5 access log exists before injection planning")
    suite_config = load_yaml(suite_config_file)
    suite_settings = suite_config.get("locked_evaluation_suite")
    if (
        not isinstance(suite_settings, dict)
        or suite_settings.get("schema") != "locked_suite_v2"
        or suite_settings.get("corpus_label") != "GWTC-5.0_O4b_locked_suite_v2"
        or suite_settings.get("required_split") != "test"
        or suite_settings.get("observing_runs") != ["O4b"]
        or suite_settings.get("catalog_release") != "GWTC-5.0"
    ):
        raise ValueError("locked injection plan requires the exact GWTC-5.0/O4b suite")
    population_config = load_yaml(population_config_file)
    settings, families, stress = _population_sections(population_config, suite_settings)
    seed = int(settings["seed"])
    availability_report = json.loads(
        availability_report_file.read_text(encoding="utf-8")
    )
    availability_rows = _load_jsonl(availability_manifest)
    if (
        availability_report.get("status")
        != "score_blind_gwtc5_o4b_availability_inventory"
        or availability_report.get("passed") is not True
        or availability_report.get("manifest_sha256") != file_sha256(availability_manifest)
        or Path(str(availability_report.get("manifest_path", ""))).resolve()
        != availability_manifest
        or availability_report.get("suite_config_sha256") != file_sha256(suite_config_file)
        or Path(str(availability_report.get("access_log_path", ""))).resolve()
        != access_log
        or availability_report.get("candidate_catalog_queried") is not False
        or availability_report.get("candidate_scores_inspected") is not False
        or availability_report.get("event_level_parameters_inspected") is not False
        or int(availability_report.get("test_strain_files_downloaded", -1)) != 0
        or int(availability_report.get("test_strain_bytes_read", -1)) != 0
        or int(availability_report.get("test_strain_rows_read", -1)) != 0
        or int(availability_report.get("availability_blocks", -1)) != len(availability_rows)
    ):
        raise ValueError("locked availability evidence is not a score-blind replay")
    exposed = sorted(
        field for row in availability_rows for field in RESULT_FIELDS if field in row
    )
    if exposed:
        raise ValueError(f"locked availability exposes result fields: {sorted(set(exposed))}")
    availability_ids_seen: set[str] = set()
    availability_blocks_seen: set[str] = set()
    for index, row in enumerate(availability_rows):
        required = {
            "availability_id",
            "split",
            "observing_run",
            "catalog_release",
            "gps_start",
            "gps_end",
            "gps_block",
            "available_ifos",
            "compatible_detector_subsets",
        }
        if required - set(row):
            raise ValueError(f"locked availability row lacks identity fields: {index}")
        availability_id = str(row["availability_id"])
        gps_block = str(row["gps_block"])
        ifos = [str(value) for value in row["available_ifos"]]
        compatible = [str(value) for value in row["compatible_detector_subsets"]]
        if (
            not availability_id
            or availability_id in availability_ids_seen
            or not gps_block
            or gps_block in availability_blocks_seen
            or row["split"] != "test"
            or row["observing_run"] != "O4b"
            or row["catalog_release"] != "GWTC-5.0"
            or ifos != sorted(set(ifos))
            or len(ifos) < 2
            or not set(ifos) <= ALLOWED_IFOS
            or "+".join(ifos) not in compatible
            or float(row["gps_end"]) <= float(row["gps_start"])
        ):
            raise ValueError(f"locked availability identity is invalid at row {index}")
        availability_ids_seen.add(availability_id)
        availability_blocks_seen.add(gps_block)

    identity = {
        "schema": "score_blind_gwtc5_locked_injection_inventory_v1",
        "availability_manifest_path": str(availability_manifest),
        "availability_manifest_sha256": file_sha256(availability_manifest),
        "availability_report_path": str(availability_report_file),
        "availability_report_sha256": file_sha256(availability_report_file),
        "suite_config_path": str(suite_config_file),
        "suite_config_sha256": file_sha256(suite_config_file),
        "population_config_path": str(population_config_file),
        "population_config_sha256": file_sha256(population_config_file),
        "access_log_path": str(access_log),
        "seed": seed,
    }
    manifest_path = output / "gwtc5_locked_injection_inventory.jsonl"
    report_path = output / "gwtc5_locked_injection_inventory_report.json"
    if report_path.is_file():
        completed = json.loads(report_path.read_text(encoding="utf-8"))
        if completed.get("freeze_identity") != identity:
            raise ValueError("existing locked injection inventory has another identity")
        if (
            not manifest_path.is_file()
            or completed.get("manifest_sha256") != file_sha256(manifest_path)
        ):
            raise ValueError("existing locked injection inventory changed")
        return completed
    if manifest_path.exists():
        raise FileExistsError("partial locked injection inventory exists")

    availability_ids = [str(row["availability_id"]) for row in availability_rows]
    family_counts = _allocate(len(availability_rows), families)
    family_by_id: dict[str, str] = {}
    ordered = _hash_order(availability_ids, seed, "source_family")
    cursor = 0
    for family in sorted(family_counts):
        count = family_counts[family]
        for availability_id in ordered[cursor : cursor + count]:
            family_by_id[availability_id] = family
        cursor += count
    bbh_ids = [value for value in availability_ids if family_by_id[value] == "BBH"]
    high_mass_count = int(stress["high_mass_unequal_mass"]["minimum_rows"])
    high_spin_count = int(stress["high_spin_precessing"]["minimum_rows"])
    if high_mass_count + high_spin_count > len(bbh_ids):
        raise ValueError("BBH allocation cannot satisfy disjoint mass/spin stress quotas")
    high_mass_ids = set(
        _hash_order(bbh_ids, seed, "high_mass_unequal_mass")[:high_mass_count]
    )
    high_spin_order = [
        value
        for value in _hash_order(bbh_ids, seed, "high_spin_precessing")
        if value not in high_mass_ids
    ]
    high_spin_ids = set(high_spin_order[:high_spin_count])

    selected_by_role: dict[str, set[str]] = {}
    for role in ("glitch_overlap", "calibration_perturbation", "waveform_systematics"):
        protocol = stress[role]
        count = max(
            int(protocol["minimum_rows"]),
            int(round(float(protocol["target_fraction"]) * len(availability_rows))),
        )
        selected_by_role[role] = set(_hash_order(availability_ids, seed, role)[:count])
    calibration_ids = _hash_order(
        list(selected_by_role["calibration_perturbation"]),
        seed,
        "calibration_scenario",
    )
    calibration_scenarios = stress["calibration_perturbation"]["scenarios"]
    calibration_by_id = {
        value: calibration_scenarios[index % len(calibration_scenarios)]
        for index, value in enumerate(calibration_ids)
    }

    cosmology = FlatLambdaCDMGrid()
    family_survey = {}
    for family, count in family_counts.items():
        maximum_distance = float(families[family]["maximum_distance_mpc"])
        maximum_redshift = float(cosmology.redshift_at_luminosity_distance(maximum_distance))
        maximum_comoving = float(cosmology.distances_at_redshift(maximum_redshift)[0])
        family_survey[family] = {
            "maximum_distance_mpc": maximum_distance,
            "maximum_redshift": maximum_redshift,
            "maximum_comoving_distance_mpc": maximum_comoving,
            "proposal_comoving_volume_mpc3": 4.0 * math.pi / 3.0 * maximum_comoving**3,
            "proposal_family_fraction": count / len(availability_rows),
        }

    rows = []
    required_stress = list(suite_settings["endpoints"]["required_stress_strata"])
    for availability in availability_rows:
        availability_id = str(availability["availability_id"])
        row_seed = int(
            canonical_hash(
                {"seed": seed, "availability_id": availability_id, "role": "source_parameters"},
                16,
            ),
            16,
        )
        rng = random.Random(row_seed)
        family = family_by_id[availability_id]
        family_settings = dict(families[family])
        high_mass = availability_id in high_mass_ids
        high_spin = availability_id in high_spin_ids
        if family == "BBH":
            family_settings["high_mass_mass_1_msun"] = stress[
                "high_mass_unequal_mass"
            ]["mass_1_msun"]
            family_settings["high_mass_mass_ratio"] = stress[
                "high_mass_unequal_mass"
            ]["mass_ratio"]
            family_settings["high_spin_magnitude"] = stress["high_spin_precessing"][
                "primary_spin_magnitude"
            ]
            family_settings["high_spin_tilt_radians"] = stress[
                "high_spin_precessing"
            ]["primary_tilt_radians"]
        source = _source_parameters(
            family,
            family_settings,
            high_mass,
            high_spin,
            rng,
        )
        survey = family_survey[family]
        comoving_distance = survey["maximum_comoving_distance_mpc"] * rng.random() ** (
            1.0 / 3.0
        )
        redshift = float(cosmology.redshift_at_comoving_distance(comoving_distance))
        luminosity_distance = (1.0 + redshift) * comoving_distance
        context = float(family_settings["context_duration_seconds"])
        guard = float(settings["availability_policy"]["coalescence_guard_seconds"])
        gps_time = rng.uniform(
            float(availability["gps_start"]) + context / 2.0 + guard,
            float(availability["gps_end"]) - context / 2.0 - guard,
        )
        injection_id = "gwtc5-injection-" + canonical_hash(
            {"availability_id": availability_id, "seed": seed}, 24
        )
        waveform_id = "gwtc5-waveform-" + canonical_hash(
            {"injection_id": injection_id, "population": settings["schema"]}, 24
        )
        evidence: dict[str, Any] = {}
        if len(availability["available_ifos"]) < 3:
            evidence["missing_detector"] = {
                "predicate": stress["missing_detector"]["predicate"],
                "available_ifos": availability["available_ifos"],
            }
        if high_mass:
            evidence["high_mass_unequal_mass"] = {
                "mass_1_msun": source["mass_1_msun"],
                "mass_ratio": source["mass_ratio"],
                "predicate_mass_1_minimum_msun": stress["high_mass_unequal_mass"][
                    "predicate_mass_1_minimum_msun"
                ],
                "predicate_mass_ratio_maximum": stress["high_mass_unequal_mass"][
                    "predicate_mass_ratio_maximum"
                ],
            }
        if high_spin:
            evidence["high_spin_precessing"] = {
                "approximant": source["waveform_approximant"],
                "primary_spin_magnitude": math.sqrt(
                    sum(source[f"spin_1{axis}"] ** 2 for axis in ("x", "y", "z"))
                ),
                "primary_in_plane_spin": math.hypot(
                    source["spin_1x"], source["spin_1y"]
                ),
            }
        if availability_id in selected_by_role["waveform_systematics"]:
            alternative = stress["waveform_systematics"]["alternative_by_primary"].get(
                source["waveform_approximant"]
            )
            if not alternative:
                raise ValueError("waveform-systematics mapping lacks a primary approximant")
            evidence["waveform_systematics"] = {
                "assignment_protocol": stress["waveform_systematics"][
                    "assignment_protocol"
                ],
                "primary_approximant": source["waveform_approximant"],
                "alternative_approximant": alternative,
            }
        if availability_id in selected_by_role["calibration_perturbation"]:
            evidence["calibration_perturbation"] = {
                "assignment_protocol": stress["calibration_perturbation"][
                    "assignment_protocol"
                ],
                "scenario": calibration_by_id[availability_id],
            }
        if availability_id in selected_by_role["glitch_overlap"]:
            evidence["glitch_overlap"] = {
                "selector_protocol": stress["glitch_overlap"]["selector_protocol"],
                "assignment_protocol": stress["glitch_overlap"]["assignment_protocol"],
                "assignment_key": canonical_hash(
                    {
                        "injection_id": injection_id,
                        "protocol": stress["glitch_overlap"]["assignment_protocol"],
                    },
                    32,
                ),
                "unavailable_policy": stress["glitch_overlap"]["unavailable_policy"],
                "auxiliary_veto_allowed": False,
            }
        strata = [name for name in required_stress if name in evidence]
        if not strata:
            strata = ["nominal"]
        ifos = [str(value) for value in availability["available_ifos"]]
        row = {
            "injection_id": injection_id,
            "waveform_id": waveform_id,
            "availability_id": availability_id,
            "background_window_id": availability_id,
            "split": "test",
            "source_family": family,
            "observing_run": "O4b",
            "catalog_release": "GWTC-5.0",
            "gps_block": availability["gps_block"],
            "gps_time": gps_time,
            "ifos": ifos,
            "detector_subset": "+".join(ifos),
            "required_context_duration_seconds": context,
            "waveform_backend": "pycbc_lalsimulation_requires_locked_runtime_validation",
            "f_lower_hz": float(family_settings["f_lower_hz"]),
            "luminosity_distance_mpc": luminosity_distance,
            "comoving_distance_mpc": comoving_distance,
            "redshift": redshift,
            "maximum_distance_mpc": survey["maximum_distance_mpc"],
            "proposal_family_fraction": survey["proposal_family_fraction"],
            "proposal_comoving_volume_mpc3": survey[
                "proposal_comoving_volume_mpc3"
            ],
            "source_frame_time_factor": 1.0 / (1.0 + redshift),
            "pre_access_vt_weight": None,
            "vt_weight_policy": settings["proposal_weighting"]["live_time_policy"],
            "stress_strata": strata,
            "stress_evidence": evidence,
            "seed": row_seed,
            "inclination": math.acos(rng.uniform(-1.0, 1.0)),
            "right_ascension": rng.uniform(0.0, 2.0 * math.pi),
            "declination": math.asin(rng.uniform(-1.0, 1.0)),
            "polarization": rng.uniform(0.0, math.pi),
            "coalescence_phase": rng.uniform(0.0, 2.0 * math.pi),
            **source,
            "mass_1_detector_msun": source["mass_1_msun"] * (1.0 + redshift),
            "mass_2_detector_msun": source["mass_2_msun"] * (1.0 + redshift),
        }
        if "waveform_systematics" in evidence:
            row["alternative_waveform_approximant"] = evidence[
                "waveform_systematics"
            ]["alternative_approximant"]
        if "calibration_perturbation" in evidence:
            row["calibration_scenario"] = evidence["calibration_perturbation"][
                "scenario"
            ]
        rows.append(row)

    audit = audit_gwtc5_locked_injection_rows(
        rows,
        availability_rows,
        suite_settings,
        settings,
    )
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    report = {
        "status": "score_blind_gwtc5_locked_injection_inventory",
        "passed": True,
        "scientific_claim_allowed": False,
        "locked_suite_schema": "locked_suite_v2",
        "corpus_label": suite_settings["corpus_label"],
        "catalog_release": "GWTC-5.0",
        "observing_runs": ["O4b"],
        "required_split": "test",
        "freeze_identity": identity,
        "seed": seed,
        "rows": len(rows),
        "minimum_usable_after_dq": int(
            settings["availability_policy"]["minimum_usable_after_dq"]
        ),
        "candidate_catalog_queried": False,
        "candidate_scores_inspected": False,
        "event_level_parameters_inspected": False,
        "test_strain_files_downloaded": 0,
        "test_strain_bytes_read": 0,
        "test_strain_rows_read": 0,
        "pre_access_vt_weights_assigned": False,
        "post_access_dq_replacement_allowed": False,
        "physical_stress_predicates_passed": True,
        "audit": audit,
        "availability_manifest_path": str(availability_manifest),
        "availability_manifest_sha256": file_sha256(availability_manifest),
        "availability_report_path": str(availability_report_file),
        "availability_report_sha256": file_sha256(availability_report_file),
        "suite_config_path": str(suite_config_file),
        "suite_config_sha256": file_sha256(suite_config_file),
        "population_config_path": str(population_config_file),
        "population_config_sha256": file_sha256(population_config_file),
        "access_log_path": str(access_log),
        "access_log_exists": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "cosmology": cosmology.metadata(),
        **execution_provenance(),
    }
    atomic_write_json(report_path, report)
    return report
