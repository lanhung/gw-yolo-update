from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Calibration manifest row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError("Calibration perturbation manifests cannot be empty")
    return rows


def _observing_run(row: dict[str, Any], default: str | None = None) -> str:
    explicit = row.get("observing_run") or row.get("run")
    if explicit:
        return str(explicit)
    gps_block = str(row.get("gps_block", ""))
    if gps_block.startswith(("O1:", "O2:", "O3a:", "O3b:", "O4a:", "O4b:")):
        return gps_block.split(":", 1)[0]
    if default:
        return default
    raise ValueError("Calibration perturbation row lacks observing-run identity")


def _row_ifos(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("ifos") or row.get("source_ifos") or row.get("available_ifos")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Calibration perturbation row lacks detector identities")
    ifos = tuple(str(value) for value in raw)
    if len(ifos) != len(set(ifos)):
        raise ValueError("Calibration perturbation row repeats a detector")
    return ifos


def _manifest_identity(
    path: str | Path,
    rows: list[dict[str, Any]],
    id_field: str,
    required_split: str,
    default_observing_run: str | None,
) -> dict[str, Any]:
    if any(str(row.get("split")) != required_split for row in rows):
        raise ValueError(f"Calibration perturbation requires {required_split}-only manifests")
    identifiers = [str(row.get(id_field, "")) for row in rows]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Calibration perturbation manifest has invalid {id_field} values")
    gps_block_run_map = {
        str(row["gps_block"]): _observing_run(row, default_observing_run) for row in rows
    }
    runs = sorted(set(gps_block_run_map.values()))
    gps_blocks = sorted({str(row.get("gps_block", "")) for row in rows})
    if any(not value for value in gps_blocks):
        raise ValueError("Calibration perturbation manifest lacks GPS blocks")
    detector_subsets: dict[str, int] = {}
    for row in rows:
        subset = "".join(_row_ifos(row))
        detector_subsets[subset] = detector_subsets.get(subset, 0) + 1
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "id_field": id_field,
        "ids_hash": canonical_hash(sorted(identifiers), 64),
        "gps_blocks": gps_blocks,
        "gps_blocks_hash": canonical_hash(gps_blocks, 64),
        "observing_runs": runs,
        "gps_block_run_map": dict(sorted(gps_block_run_map.items())),
        "gps_block_run_map_sha256": canonical_hash(gps_block_run_map, 64),
        "detector_subset_counts": dict(sorted(detector_subsets.items())),
        "split": required_split,
    }


def _validate_template(
    name: str,
    template: dict[str, Any],
    anchors: np.ndarray,
) -> dict[str, Any]:
    amplitude = np.asarray(template.get("maximum_amplitude_fraction"), dtype=np.float64)
    phase_degrees = np.asarray(template.get("maximum_phase_degrees"), dtype=np.float64)
    if amplitude.shape != anchors.shape or phase_degrees.shape != anchors.shape:
        raise ValueError(f"Calibration envelope template {name} differs from anchor count")
    if (
        not np.isfinite(amplitude).all()
        or not np.isfinite(phase_degrees).all()
        or np.any(amplitude <= 0)
        or np.any(amplitude >= 0.5)
        or np.any(phase_degrees <= 0)
        or np.any(phase_degrees > 45)
    ):
        raise ValueError(f"Calibration envelope template {name} has invalid bounds")
    source = template.get("source")
    if not isinstance(source, dict) or not source.get("identity") or not source.get(
        "semantics"
    ):
        raise ValueError(f"Calibration envelope template {name} lacks source provenance")
    return {
        "maximum_amplitude_fraction": amplitude.tolist(),
        "maximum_phase_degrees": phase_degrees.tolist(),
        "source": source,
    }


def _response(
    anchors: np.ndarray,
    amplitude: np.ndarray,
    phase_degrees: np.ndarray,
    template_name: str,
) -> dict[str, Any]:
    return {
        "anchor_frequencies_hz": anchors.tolist(),
        "amplitude_fraction": amplitude.tolist(),
        "phase_radians": np.deg2rad(phase_degrees).tolist(),
        "envelope_template": template_name,
        "application": "multiplicative_rfft_strain_response_v1",
    }


def freeze_calibration_perturbation_plan(
    background_manifest_path: str | Path,
    injection_manifest_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze score-blind, run-correlated calibration stress scenarios on validation data."""

    target = Path(output_path)
    if target.exists():
        raise FileExistsError("Calibration perturbation plans are immutable")
    config = load_yaml(config_path)
    settings = config.get("calibration_perturbation")
    if not isinstance(settings, dict):
        raise ValueError("Configuration requires calibration_perturbation")
    if settings.get("protocol") != "run_correlated_frequency_response_v1":
        raise ValueError("Unsupported calibration perturbation protocol")
    required_split = str(settings.get("required_split", "val"))
    if required_split != "val":
        raise ValueError("Calibration perturbation selection must be validation-only")
    seed = int(settings.get("seed", 0))
    if seed <= 0:
        raise ValueError("Calibration perturbation seed must be positive")
    target_rate = int(settings.get("target_sample_rate", 0))
    anchors = np.asarray(settings.get("anchor_frequencies_hz"), dtype=np.float64)
    if (
        target_rate <= 0
        or anchors.ndim != 1
        or anchors.size < 3
        or not np.isfinite(anchors).all()
        or np.any(anchors <= 0)
        or np.any(np.diff(anchors) <= 0)
        or anchors[-1] >= target_rate / 2
    ):
        raise ValueError("Calibration perturbation frequency anchors are invalid")
    model_ifos = tuple(str(value) for value in settings.get("model_ifos", []))
    if len(model_ifos) < 2 or len(model_ifos) != len(set(model_ifos)):
        raise ValueError("Calibration perturbation needs a unique multi-IFO model set")
    default_observing_run = settings.get("default_observing_run")
    if default_observing_run is not None and not str(default_observing_run):
        raise ValueError("Calibration default observing run cannot be empty")
    raw_templates = settings.get("envelope_templates")
    assignments = settings.get("run_template_assignments")
    if not isinstance(raw_templates, dict) or not raw_templates:
        raise ValueError("Calibration perturbation requires envelope templates")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Calibration perturbation requires run-template assignments")
    templates = {
        str(name): _validate_template(str(name), value, anchors)
        for name, value in raw_templates.items()
        if isinstance(value, dict)
    }
    if len(templates) != len(raw_templates):
        raise ValueError("Calibration envelope templates must be mappings")

    background_rows = _read_jsonl(background_manifest_path)
    injection_rows = _read_jsonl(injection_manifest_path)
    identities = {
        "background": _manifest_identity(
            background_manifest_path,
            background_rows,
            "window_id",
            required_split,
            str(default_observing_run) if default_observing_run is not None else None,
        ),
        "injection": _manifest_identity(
            injection_manifest_path,
            injection_rows,
            "injection_id",
            required_split,
            str(default_observing_run) if default_observing_run is not None else None,
        ),
    }
    overlap = sorted(
        set(identities["background"]["gps_blocks"])
        & set(identities["injection"]["gps_blocks"])
    )
    if overlap:
        raise ValueError("Calibration background and injection purposes share GPS blocks")
    rows = background_rows + injection_rows
    observed_ifos = {ifo for row in rows for ifo in _row_ifos(row)}
    if not observed_ifos.issubset(model_ifos):
        raise ValueError("Calibration manifests use detectors outside model_ifos")
    runs = sorted(
        set(identities["background"]["observing_runs"])
        | set(identities["injection"]["observing_runs"])
    )
    unknown_runs = sorted(set(runs) - set(assignments))
    if unknown_runs:
        raise ValueError(f"Calibration envelope assignments omit runs: {unknown_runs}")
    for run in runs:
        if str(assignments[run]) not in templates:
            raise ValueError(f"Calibration run {run} selects an unknown template")
    random_draws = int(settings.get("random_draws", 0))
    if random_draws < 4:
        raise ValueError("Calibration robustness requires at least four random stress draws")
    rng = np.random.default_rng(seed)
    scenario_specs = [
        ("envelope_plus", "corner", 1.0, 1.0),
        ("envelope_minus", "corner", -1.0, -1.0),
        ("amplitude_plus_phase_minus", "corner", 1.0, -1.0),
    ]
    scenarios = []
    for scenario_id, kind, amplitude_sign, phase_sign in scenario_specs:
        responses = {}
        for run in runs:
            template_name = str(assignments[run])
            template = templates[template_name]
            amp_limit = np.asarray(template["maximum_amplitude_fraction"])
            phase_limit = np.asarray(template["maximum_phase_degrees"])
            responses[run] = {
                ifo: _response(
                    anchors,
                    amplitude_sign * amp_limit,
                    phase_sign * phase_limit,
                    template_name,
                )
                for ifo in model_ifos
            }
        scenarios.append({"scenario_id": scenario_id, "kind": kind, "responses": responses})
    for draw in range(random_draws):
        responses = {}
        for run in runs:
            template_name = str(assignments[run])
            template = templates[template_name]
            amp_limit = np.asarray(template["maximum_amplitude_fraction"])
            phase_limit = np.asarray(template["maximum_phase_degrees"])
            responses[run] = {
                ifo: _response(
                    anchors,
                    rng.uniform(-amp_limit, amp_limit),
                    rng.uniform(-phase_limit, phase_limit),
                    template_name,
                )
                for ifo in model_ifos
            }
        scenarios.append(
            {
                "scenario_id": f"random_draw_{draw:03d}",
                "kind": "bounded_uniform_stress_draw",
                "responses": responses,
            }
        )
    scenario_ids = [row["scenario_id"] for row in scenarios]
    result = {
        "status": "frozen_validation_calibration_perturbation_plan",
        "passed": True,
        "scientific_claim_allowed": False,
        "test_rows_read": 0,
        "candidate_scores_inspected": False,
        "physical_time_domain_perturbation": True,
        "fresh_time_frequency_transform_required": True,
        "response_correlation": "one response per scenario/observing-run/IFO across all rows",
        "envelope_interpretation": (
            "bounded validation stress envelope; not posterior calibration samples"
        ),
        "protocol": settings["protocol"],
        "seed": seed,
        "target_sample_rate": target_rate,
        "model_ifos": list(model_ifos),
        "default_observing_run": (
            str(default_observing_run) if default_observing_run is not None else None
        ),
        "observing_runs": runs,
        "purpose_gps_block_overlap": 0,
        "manifests": identities,
        "anchor_frequencies_hz": anchors.tolist(),
        "envelope_templates": templates,
        "run_template_assignments": {run: str(assignments[run]) for run in runs},
        "scenario_ids": scenario_ids,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result


def load_calibration_perturbation_scenario(
    plan_path: str | Path,
    manifest_path: str | Path,
    role: str,
    scenario_id: str,
    expected_sample_rate: int,
    expected_ifos: tuple[str, ...],
) -> dict[str, Any]:
    """Replay a frozen plan and return exactly one run-correlated response scenario."""

    plan_file = Path(plan_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "frozen_validation_calibration_perturbation_plan"
        or plan.get("passed") is not True
        or plan.get("test_rows_read") != 0
        or plan.get("candidate_scores_inspected") is not False
    ):
        raise ValueError("Calibration perturbation plan is not a frozen validation plan")
    if role not in {"background", "injection"}:
        raise ValueError("Calibration perturbation role must be background or injection")
    identity = plan.get("manifests", {}).get(role, {})
    if file_sha256(manifest_path) != identity.get("sha256"):
        raise ValueError(f"Calibration perturbation {role} manifest differs from the plan")
    if int(plan.get("target_sample_rate", -1)) != expected_sample_rate:
        raise ValueError("Calibration perturbation sample rate differs from the scorer")
    if tuple(str(value) for value in plan.get("model_ifos", [])) != expected_ifos:
        raise ValueError("Calibration perturbation detector set differs from the scorer")
    selected = [row for row in plan.get("scenarios", []) if row.get("scenario_id") == scenario_id]
    if len(selected) != 1:
        raise ValueError("Calibration perturbation scenario is absent or duplicated")
    scenario = selected[0]
    return {
        "plan_path": str(plan_file.resolve()),
        "plan_sha256": file_sha256(plan_file),
        "role": role,
        "manifest_sha256": identity["sha256"],
        "scenario_id": scenario_id,
        "scenario_kind": scenario["kind"],
        "responses": scenario["responses"],
        "response_sha256": canonical_hash(scenario["responses"], 64),
        "gps_block_run_map": identity["gps_block_run_map"],
        "gps_block_run_map_sha256": identity["gps_block_run_map_sha256"],
        "physical_time_domain_perturbation": True,
        "fresh_time_frequency_transform_required": True,
    }


def response_for_row(
    scenario: dict[str, Any], row: dict[str, Any], ifo: str
) -> dict[str, Any]:
    gps_block = str(row.get("gps_block", ""))
    run = scenario.get("gps_block_run_map", {}).get(gps_block)
    if not run:
        run = _observing_run(row)
    response = scenario.get("responses", {}).get(run, {}).get(ifo)
    if not isinstance(response, dict):
        raise ValueError(f"Calibration response is missing for {run}/{ifo}")
    return response


def apply_frequency_dependent_calibration_response(
    strain: np.ndarray,
    sample_rate: int,
    response: dict[str, Any],
) -> np.ndarray:
    """Apply a frozen multiplicative calibration response to physical time-domain strain."""

    values = np.asarray(strain, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Calibration response requires a finite 1D strain array")
    if sample_rate <= 0:
        raise ValueError("Calibration response sample rate must be positive")
    anchors = np.asarray(response.get("anchor_frequencies_hz"), dtype=np.float64)
    amplitude = np.asarray(response.get("amplitude_fraction"), dtype=np.float64)
    phase = np.asarray(response.get("phase_radians"), dtype=np.float64)
    if (
        response.get("application") != "multiplicative_rfft_strain_response_v1"
        or anchors.ndim != 1
        or anchors.size < 3
        or amplitude.shape != anchors.shape
        or phase.shape != anchors.shape
        or not np.isfinite(anchors).all()
        or not np.isfinite(amplitude).all()
        or not np.isfinite(phase).all()
        or np.any(np.diff(anchors) <= 0)
        or np.any(anchors <= 0)
        or anchors[-1] >= sample_rate / 2
        or np.any(amplitude <= -1)
        or np.any(np.abs(phase) > math.pi / 2)
    ):
        raise ValueError("Calibration response envelope is invalid")
    frequencies = np.fft.rfftfreq(values.size, 1.0 / sample_rate)
    interpolated_amplitude = np.interp(
        frequencies, anchors, amplitude, left=amplitude[0], right=amplitude[-1]
    )
    interpolated_phase = np.interp(
        frequencies, anchors, phase, left=phase[0], right=phase[-1]
    )
    interpolated_phase[0] = 0.0
    if values.size % 2 == 0:
        interpolated_phase[-1] = 0.0
    transfer = (1.0 + interpolated_amplitude) * np.exp(1j * interpolated_phase)
    transformed = np.fft.irfft(np.fft.rfft(values) * transfer, n=values.size)
    if not np.isfinite(transformed).all():
        raise ValueError("Calibration perturbation produced non-finite strain")
    return transformed
