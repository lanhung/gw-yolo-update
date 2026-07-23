from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .injection_bootstrap import hierarchical_injection_bootstrap
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


def _load_bound_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"Required calibration artifact is missing: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Calibration artifact is not a JSON object: {resolved}")
    return resolved, value


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def freeze_calibration_perturbation_scenario_result(
    plan_path: str | Path,
    background_score_report_path: str | Path,
    injection_score_report_path: str | Path,
    background_timing_application_report_path: str | Path,
    injection_timing_application_report_path: str | Path,
    background_search_report_path: str | Path,
    injection_ranking_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze one replayable calibration-stress candidate-search chain."""

    target = Path(output_path)
    if target.exists():
        raise FileExistsError("Calibration scenario receipts are immutable")
    plan_file, plan = _load_bound_json(plan_path)
    if (
        plan.get("status") != "frozen_validation_calibration_perturbation_plan"
        or plan.get("passed") is not True
        or plan.get("test_rows_read") != 0
        or plan.get("candidate_scores_inspected") is not False
    ):
        raise ValueError("Scenario receipt requires a frozen validation perturbation plan")
    inputs = {
        "background_score": _load_bound_json(background_score_report_path),
        "injection_score": _load_bound_json(injection_score_report_path),
        "background_timing": _load_bound_json(background_timing_application_report_path),
        "injection_timing": _load_bound_json(injection_timing_application_report_path),
        "background_search": _load_bound_json(background_search_report_path),
        "injection_ranking": _load_bound_json(injection_ranking_report_path),
    }
    expected_statuses = {
        "background_score": {"real_o4a_domain_transfer_diagnostic"},
        "injection_score": {
            "physical_waveform_real_noise_domain_transfer_diagnostic"
        },
        "background_timing": {"candidate_timing_calibration_applied"},
        "injection_timing": {"candidate_timing_calibration_applied"},
        "background_search": {
            "subwindow_clustered_time_slide_integration_only",
            "variable_detector_set_block_permutation_background",
        },
        "injection_ranking": {
            "physical_network_injection_candidate_rankings",
            "physical_variable_detector_set_injection_candidate_rankings",
        },
    }
    for name, (_, report) in inputs.items():
        if report.get("status") not in expected_statuses[name]:
            raise ValueError(f"Calibration scenario {name} has the wrong status")

    plan_sha = file_sha256(plan_file)
    score_reports = {name: inputs[name][1] for name in ("background_score", "injection_score")}
    roles = {"background_score": "background", "injection_score": "injection"}
    scenario_ids: set[str] = set()
    for name, report in score_reports.items():
        perturbation = report.get("calibration_perturbation")
        role = roles[name]
        if (
            not isinstance(perturbation, dict)
            or perturbation.get("plan_sha256") != plan_sha
            or perturbation.get("role") != role
            or perturbation.get("manifest_sha256")
            != plan.get("manifests", {}).get(role, {}).get("sha256")
            or report.get("required_split") != "val"
            or report.get("observed_splits") != ["val"]
            or report.get("physical_time_domain_perturbation") is not True
            or report.get("fresh_time_frequency_transform") is not True
            or int(report.get("failed_windows", report.get("failed_injections", -1))) != 0
            or file_sha256(report["triggers_path"]) != report.get("triggers_sha256")
        ):
            raise ValueError(f"Calibration scenario {name} is not a complete physical replay")
        scenario_ids.add(str(perturbation.get("scenario_id", "")))
    if len(scenario_ids) != 1 or next(iter(scenario_ids)) not in plan.get("scenario_ids", []):
        raise ValueError("Background and injection scores use different frozen scenarios")
    scenario_id = next(iter(scenario_ids))
    identity_fields = ("checkpoint_sha256", "config_sha256", "code_commit")
    if any(
        score_reports["background_score"].get(field) != score_reports["injection_score"].get(field)
        for field in identity_fields
    ):
        raise ValueError("Calibration scenario background/injection model identity differs")

    controlled_transfers = []
    for role in ("background", "injection"):
        score_name = f"{role}_score"
        timing_name = f"{role}_timing"
        score_path, score = inputs[score_name]
        _, timing = inputs[timing_name]
        scoring = timing.get("candidate_extraction_provenance", {}).get("scoring", {})
        calibration_scoring = timing.get("calibration_scoring_provenance", {})
        cross_commit = scoring.get("code_commit") != calibration_scoring.get(
            "code_commit"
        )
        if (
            timing.get("uncalibrated_candidates") != 0
            or timing.get("scoring_provenance_matches") is not True
            or scoring.get("score_report_sha256") != file_sha256(score_path)
            or scoring.get("trigger_manifest_sha256") != score.get("triggers_sha256")
            or timing.get("output_sha256") != file_sha256(timing["output_path"])
            or (
                cross_commit
                and (
                    not timing.get(
                        "calibration_timing_transfer_compatibility_report_sha256"
                    )
                    or timing.get("calibration_perturbation_plan_sha256") != plan_sha
                )
            )
        ):
            raise ValueError(f"Calibration scenario {role} candidate/timing chain is invalid")
        if cross_commit:
            transfer_path = Path(
                str(
                    timing.get(
                        "calibration_timing_transfer_compatibility_report_path", ""
                    )
                )
            ).resolve()
            transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
            if (
                file_sha256(transfer_path)
                != timing.get(
                    "calibration_timing_transfer_compatibility_report_sha256"
                )
                or transfer.get("status")
                != "calibration_timing_transfer_implementation_compatibility"
                or transfer.get("passed") is not True
                or transfer.get("differences") != []
                or transfer.get("reference_commit")
                != calibration_scoring.get("code_commit")
                or transfer.get("candidate_commit") != scoring.get("code_commit")
            ):
                raise ValueError("Calibration timing transfer proof failed replay")
            controlled_transfers.append(
                {
                    "reference_commit": transfer["reference_commit"],
                    "candidate_commit": transfer["candidate_commit"],
                    "path": str(transfer_path),
                    "sha256": file_sha256(transfer_path),
                }
            )
    if controlled_transfers and (
        len(controlled_transfers) != 2
        or controlled_transfers[0] != controlled_transfers[1]
    ):
        raise ValueError("Background/injection calibration timing transfers differ")
    background_timing = inputs["background_timing"][1]
    injection_timing = inputs["injection_timing"][1]
    timing_sha = background_timing.get("calibration_report_sha256")
    if not timing_sha or timing_sha != injection_timing.get("calibration_report_sha256"):
        raise ValueError("Calibration scenario must reuse one frozen timing calibration")

    _, background = inputs["background_search"]
    _, ranking = inputs["injection_ranking"]
    variable_detector_set = (
        background.get("status")
        == "variable_detector_set_block_permutation_background"
        and ranking.get("status")
        == "physical_variable_detector_set_injection_candidate_rankings"
    )
    if variable_detector_set != (
        background.get("status")
        == "variable_detector_set_block_permutation_background"
        or ranking.get("status")
        == "physical_variable_detector_set_injection_candidate_rankings"
    ):
        raise ValueError(
            "Calibration scenario background/ranking detector policies differ"
        )
    if (
        background.get("split") != "val"
        or background.get("publication_timing_gate_passed") is not True
        or background.get("candidate_timing_empirically_calibrated") is not True
        or background.get("timing_calibration_report_sha256") != timing_sha
        or background.get("candidate_manifest_sha256") != background_timing.get("output_sha256")
        or background.get("background_manifest_sha256")
        != plan.get("manifests", {}).get("background", {}).get("sha256")
        or file_sha256(background["manifest_path"]) != background.get("manifest_sha256")
        or float(background.get("equivalent_live_time_years", 0)) <= 0
    ):
        raise ValueError("Calibration scenario background search chain is invalid")
    if (
        ranking.get("split") != "val"
        or ranking.get("timing_calibration_consistent") is not True
        or ranking.get("candidate_scoring_provenance_consistent") is not True
        or ranking.get("timing_calibration_report_sha256") != timing_sha
        or ranking.get("candidate_manifest_sha256") != injection_timing.get("output_sha256")
        or ranking.get("injection_trigger_manifest_sha256")
        != score_reports["injection_score"].get("triggers_sha256")
        or file_sha256(ranking["manifest_path"]) != ranking.get("manifest_sha256")
    ):
        raise ValueError("Calibration scenario injection ranking chain is invalid")
    candidate_identity_fields = {
        "checkpoint_sha256": "candidate_checkpoint_sha256",
        "config_sha256": "candidate_config_sha256",
        "code_commit": "candidate_code_commit",
    }
    common_identity = {
        candidate_field: background.get(candidate_field)
        for candidate_field in candidate_identity_fields.values()
    } | {
        "timing_calibration_report_sha256": background.get("timing_calibration_report_sha256"),
        "empirical_timing_uncertainty_seconds": background.get(
            "empirical_timing_uncertainty_seconds"
        ),
    }
    if variable_detector_set:
        network_fields = (
            "required_detector_subsets",
            "pairwise_light_travel_time_seconds",
            "pairwise_allowed_peak_separation_seconds",
        )
        final_identity = {
            **common_identity,
            "network_coherence_policy": {
                field: background.get(field) for field in network_fields
            },
            "detector_set_policy": (
                "single_model_explicit_missing_ifo_validity_v1"
            ),
        }
    else:
        final_identity = {
            **common_identity,
            "physical_delay_limit_seconds": background.get(
                "physical_delay_limit_seconds"
            ),
            "reference_ifo": background.get("reference_ifo"),
            "second_ifo": background.get("shifted_ifo"),
        }
    if any(
        final_identity[candidate_field] != score_reports["background_score"].get(score_field)
        or ranking.get(candidate_field) != final_identity[candidate_field]
        for score_field, candidate_field in candidate_identity_fields.items()
    ):
        raise ValueError("Calibration scenario final candidate identity differs from scores")
    common_timing_fields = (
        "timing_calibration_report_sha256",
        "empirical_timing_uncertainty_seconds",
    )
    identity_invalid = any(
        final_identity[field] is None for field in common_timing_fields
    ) or any(
        ranking.get(field) != final_identity[field]
        for field in common_timing_fields
    )
    if variable_detector_set:
        network_policy = final_identity["network_coherence_policy"]
        identity_invalid = identity_invalid or any(
            network_policy[field] is None
            or ranking.get(field) != network_policy[field]
            for field in network_policy
        )
    else:
        pair_fields = (
            "physical_delay_limit_seconds",
            "reference_ifo",
        )
        identity_invalid = (
            identity_invalid
            or any(final_identity[field] is None for field in pair_fields)
            or final_identity["second_ifo"] is None
            or any(
                ranking.get(field) != final_identity[field]
                for field in pair_fields
            )
            or ranking.get("second_ifo") != final_identity["second_ifo"]
        )
    if identity_invalid:
        raise ValueError("Calibration scenario final timing/detector identity differs")

    result = {
        "status": "frozen_validation_calibration_perturbation_scenario_result",
        "passed": True,
        "scientific_claim_allowed": False,
        "locked_test_allowed": False,
        "test_rows_read": 0,
        "threshold_fitted_or_selected": False,
        "scenario_id": scenario_id,
        "plan": _artifact(plan_file),
        "model_identity": final_identity,
        "timing_calibration_report_sha256": timing_sha,
        "controlled_code_transfer": (
            controlled_transfers[0] if controlled_transfers else None
        ),
        "background_exposure": {
            "equivalent_live_time_years": float(background["equivalent_live_time_years"]),
            "background_pairing_method": background.get("background_pairing_method"),
            "slide_schedule_id": background.get("slide_schedule_id"),
            "slide_schedule_sha256": background.get("slide_schedule_sha256"),
            "manifest_sha256": background.get("manifest_sha256"),
        },
        "artifacts": {name: _artifact(path) for name, (path, _) in inputs.items()},
        "physical_time_domain_perturbation": True,
        "fresh_time_frequency_transform": True,
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result


def evaluate_calibration_perturbation_robustness(
    plan_path: str | Path,
    baseline_calibration_report_path: str | Path,
    scenario_receipt_paths: list[str | Path],
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Evaluate all frozen stresses at one unchanged validation FAR threshold."""

    target = Path(output_path)
    if target.exists():
        raise FileExistsError("Calibration robustness evaluations are immutable")
    plan_file, plan = _load_bound_json(plan_path)
    baseline_file, baseline = _load_bound_json(baseline_calibration_report_path)
    if (
        plan.get("status") != "frozen_validation_calibration_perturbation_plan"
        or plan.get("passed") is not True
        or baseline.get("status") != "frozen_validation_candidate_search_calibration"
        or baseline.get("publication_calibration_eligible") is not True
        or baseline.get("slide_schedule_audit", {}).get("passed") is not True
        or baseline.get("scientific_claim_allowed") is not False
        or baseline.get("test_evaluation") is not None
    ):
        raise ValueError("Calibration robustness requires eligible validation-only baselines")
    settings = load_yaml(config_path).get("calibration_robustness")
    if not isinstance(settings, dict):
        raise ValueError("Configuration requires calibration_robustness")
    minimum_scenarios = int(settings.get("minimum_scenarios", 0))
    maximum_loss = float(settings.get("maximum_absolute_weighted_efficiency_loss", -1))
    maximum_far_multiplier = float(settings.get("maximum_far_multiplier_of_target", 0))
    bootstrap_replicates = int(settings.get("bootstrap_replicates", 0))
    minimum_injection_gps_blocks = int(
        settings.get("minimum_injection_gps_blocks", 25)
    )
    required_detector_subsets = [
        str(value) for value in settings.get("required_detector_subsets", [])
    ]
    minimum_injections_per_detector_subset = int(
        settings.get("minimum_injections_per_detector_subset", 1)
    )
    seed = int(settings.get("seed", 0))
    if (
        minimum_scenarios < 1
        or not 0 <= maximum_loss < 1
        or maximum_far_multiplier < 1
        or bootstrap_replicates < 1
        or minimum_injection_gps_blocks < 2
        or len(required_detector_subsets)
        != len(set(required_detector_subsets))
        or any(not value for value in required_detector_subsets)
        or minimum_injections_per_detector_subset < 1
        or seed < 1
    ):
        raise ValueError("Calibration robustness policy is invalid")
    expected_ids = list(plan.get("scenario_ids", []))
    if len(expected_ids) < minimum_scenarios:
        raise ValueError("Frozen calibration plan has too few scenarios for the policy")

    baseline_ranking_path = Path(
        str(baseline.get("validation_injection_ranking_report_path", ""))
    ).resolve()
    baseline_background_path = Path(
        str(baseline.get("validation_time_slide_report_path", ""))
    ).resolve()
    if file_sha256(baseline_ranking_path) != baseline.get(
        "validation_injection_ranking_report_sha256"
    ) or file_sha256(baseline_background_path) != baseline.get(
        "validation_time_slide_report_sha256"
    ):
        raise ValueError("Baseline calibration artifact hashes do not replay")
    baseline_ranking = json.loads(baseline_ranking_path.read_text(encoding="utf-8"))
    baseline_background = json.loads(baseline_background_path.read_text(encoding="utf-8"))
    if file_sha256(baseline_ranking["manifest_path"]) != baseline_ranking.get(
        "manifest_sha256"
    ) or file_sha256(baseline_background["manifest_path"]) != baseline_background.get(
        "manifest_sha256"
    ):
        raise ValueError("Baseline candidate manifests changed after calibration")
    baseline_rows = _read_jsonl(baseline_ranking["manifest_path"])
    baseline_by_id = {str(row["injection_id"]): row for row in baseline_rows}
    if len(baseline_by_id) != len(baseline_rows):
        raise ValueError("Baseline injection ranking IDs are not unique")
    threshold = float(baseline.get("calibration", {}).get("threshold"))
    target_far = float(baseline.get("target_far_per_year"))
    if not math.isfinite(threshold) or not math.isfinite(target_far) or target_far <= 0:
        raise ValueError("Baseline calibration threshold or target FAR is invalid")
    plan_sha = file_sha256(plan_file)
    receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for value in scenario_receipt_paths:
        receipt_path, receipt = _load_bound_json(value)
        scenario_id = str(receipt.get("scenario_id", ""))
        if scenario_id in receipts:
            raise ValueError("Calibration robustness has duplicate scenario receipts")
        baseline_identity = baseline.get("identity", {})
        receipt_identity = receipt.get("model_identity", {})
        shared_identity_fields = [
            "candidate_checkpoint_sha256",
            "candidate_config_sha256",
            "timing_calibration_report_sha256",
            "empirical_timing_uncertainty_seconds",
        ]
        if baseline_identity.get("detector_set_policy") is not None:
            shared_identity_fields.extend(
                ["network_coherence_policy", "detector_set_policy"]
            )
        else:
            shared_identity_fields.extend(
                [
                    "physical_delay_limit_seconds",
                    "reference_ifo",
                    "second_ifo",
                ]
            )
        code_changed = receipt_identity.get(
            "candidate_code_commit"
        ) != baseline_identity.get("candidate_code_commit")
        transfer = receipt.get("controlled_code_transfer")
        transfer_path = (
            Path(str(transfer.get("path", ""))).resolve()
            if isinstance(transfer, dict)
            else None
        )
        if (
            receipt.get("status") != "frozen_validation_calibration_perturbation_scenario_result"
            or receipt.get("passed") is not True
            or receipt.get("test_rows_read") != 0
            or receipt.get("threshold_fitted_or_selected") is not False
            or receipt.get("plan", {}).get("sha256") != plan_sha
            or any(
                receipt_identity.get(field) != baseline_identity.get(field)
                for field in shared_identity_fields
            )
            or (
                code_changed
                and (
                    transfer_path is None
                    or not transfer_path.is_file()
                    or file_sha256(transfer_path) != transfer.get("sha256")
                    or transfer.get("reference_commit")
                    != baseline_identity.get("candidate_code_commit")
                    or transfer.get("candidate_commit")
                    != receipt_identity.get("candidate_code_commit")
                )
            )
            or (not code_changed and transfer is not None)
        ):
            raise ValueError("Calibration scenario receipt does not match the baseline")
        receipts[scenario_id] = (receipt_path, receipt)
    if set(receipts) != set(expected_ids):
        raise ValueError("Calibration robustness requires every frozen scenario exactly once")

    baseline_schedule = {
        field: baseline_background.get(field)
        for field in (
            "background_manifest_sha256",
            "background_pairing_method",
            "equivalent_live_time_years",
            "input_gps_blocks",
            "slide_schedule_id",
            "slide_schedule_sha256",
            "slide_count",
        )
    }
    baseline_weights = np.asarray(
        [float(row.get("vt_weight", row.get("weight", 1.0))) for row in baseline_rows],
        dtype=np.float64,
    )
    baseline_recovered = np.asarray(
        [float(row["ranking_score"]) >= threshold for row in baseline_rows],
        dtype=np.float64,
    )
    if (
        not np.isfinite(baseline_weights).all()
        or np.any(baseline_weights < 0)
        or float(baseline_weights.sum()) <= 0
    ):
        raise ValueError("Baseline calibration injection weights are invalid")
    baseline_efficiency = float(
        (baseline_weights * baseline_recovered).sum() / baseline_weights.sum()
    )
    reported_efficiency = float(
        baseline.get("validation_injection_diagnostic", {}).get("weighted_efficiency")
    )
    if not np.isclose(baseline_efficiency, reported_efficiency, rtol=0.0, atol=1e-12):
        raise ValueError("Baseline calibration diagnostic does not replay at its threshold")
    scenario_results = []
    detector_strata: dict[str, dict[str, Any]] = {}
    for scenario_index, scenario_id in enumerate(expected_ids):
        receipt_path, receipt = receipts[scenario_id]
        background_report_path = Path(receipt["artifacts"]["background_search"]["path"]).resolve()
        ranking_report_path = Path(receipt["artifacts"]["injection_ranking"]["path"]).resolve()
        injection_score_path = Path(receipt["artifacts"]["injection_score"]["path"]).resolve()
        for name, artifact_path in (
            ("background_search", background_report_path),
            ("injection_ranking", ranking_report_path),
            ("injection_score", injection_score_path),
        ):
            if file_sha256(artifact_path) != receipt["artifacts"][name]["sha256"]:
                raise ValueError(f"Calibration scenario {scenario_id} artifact changed")
        background_report = json.loads(background_report_path.read_text(encoding="utf-8"))
        ranking_report = json.loads(ranking_report_path.read_text(encoding="utf-8"))
        injection_score = json.loads(injection_score_path.read_text(encoding="utf-8"))
        if (
            file_sha256(background_report["manifest_path"])
            != background_report.get("manifest_sha256")
            or file_sha256(ranking_report["manifest_path"]) != ranking_report.get("manifest_sha256")
            or file_sha256(injection_score["triggers_path"])
            != injection_score.get("triggers_sha256")
        ):
            raise ValueError(f"Calibration scenario {scenario_id} manifest changed")
        scenario_schedule = {field: background_report.get(field) for field in baseline_schedule}
        if scenario_schedule != baseline_schedule:
            raise ValueError("Calibration scenarios must use identical frozen background exposure")
        background_rows = _read_jsonl(background_report["manifest_path"])
        ranking_rows = _read_jsonl(ranking_report["manifest_path"])
        scenario_by_id = {str(row["injection_id"]): row for row in ranking_rows}
        if len(scenario_by_id) != len(ranking_rows) or set(scenario_by_id) != set(baseline_by_id):
            raise ValueError("Calibration scenario does not share baseline injection IDs")
        scored_injections = {
            str(row["injection_id"]): row for row in _read_jsonl(injection_score["triggers_path"])
        }
        weights = []
        contributions = []
        joined = []
        paired_rows = []
        identity_fields = (
            "waveform_id",
            "source_family",
            "stratum",
            "gps_block",
            "gps_time",
            "vt_weight",
            "vt_weight_unit",
        )
        for injection_id in sorted(baseline_by_id):
            reference = baseline_by_id[injection_id]
            perturbed = scenario_by_id[injection_id]
            if any(reference.get(field) != perturbed.get(field) for field in identity_fields):
                raise ValueError(f"Calibration injection identity differs: {injection_id}")
            if injection_id not in scored_injections:
                raise ValueError("Calibration detector stratum is missing an injection")
            detector_subset = "+".join(scored_injections[injection_id]["valid_ifos"])
            weight = float(reference.get("vt_weight", reference.get("weight", 1.0)))
            recovered_baseline = float(reference["ranking_score"]) >= threshold
            recovered_scenario = float(perturbed["ranking_score"]) >= threshold
            weights.append(weight)
            contributions.append(weight * (recovered_scenario - recovered_baseline))
            joined.append((detector_subset, weight, recovered_baseline, recovered_scenario))
            paired_rows.append(reference)
        weights_array = np.asarray(weights, dtype=np.float64)
        contributions_array = np.asarray(contributions, dtype=np.float64)
        total_weight = float(weights_array.sum())
        if total_weight <= 0 or not np.isfinite(weights_array).all():
            raise ValueError("Calibration injection weights are invalid")
        delta_efficiency = float(contributions_array.sum() / total_weight)
        bootstrap = hierarchical_injection_bootstrap(
            paired_rows,
            contributions_array,
            weights_array,
            bootstrap_replicates,
            seed + scenario_index,
            require_physical_groups=True,
            minimum_physical_groups=minimum_injection_gps_blocks,
        )
        interval = bootstrap["interval_95"]
        exceedances = sum(float(row["ranking_score"]) >= threshold for row in background_rows)
        live_time = float(background_report["equivalent_live_time_years"])
        far_per_year = exceedances / live_time
        efficiency_gate = (
            interval[0] >= -maximum_loss
            and bootstrap["independence_audit"]["passed"]
        )
        far_gate = far_per_year <= target_far * maximum_far_multiplier
        per_detector = {}
        for subset in sorted({row[0] for row in joined}):
            subset_rows = [row for row in joined if row[0] == subset]
            subset_weight = sum(row[1] for row in subset_rows)
            per_detector[subset] = {
                "injections": len(subset_rows),
                "total_vt_weight": subset_weight,
                "baseline_weighted_efficiency": sum(row[1] * row[2] for row in subset_rows)
                / subset_weight,
                "scenario_weighted_efficiency": sum(row[1] * row[3] for row in subset_rows)
                / subset_weight,
            }
            detector_strata.setdefault(
                subset, {"scenario_count": 0, "injections": len(subset_rows)}
            )
            detector_strata[subset]["scenario_count"] += 1
        scenario_results.append(
            {
                "scenario_id": scenario_id,
                "threshold": threshold,
                "threshold_source": "unchanged_baseline_validation_calibration",
                "background_exceedances": exceedances,
                "equivalent_live_time_years": live_time,
                "far_per_year": far_per_year,
                "far_multiplier_of_target": far_per_year / target_far,
                "baseline_weighted_efficiency": baseline_efficiency,
                "scenario_weighted_efficiency": baseline_efficiency + delta_efficiency,
                "absolute_weighted_efficiency_delta": delta_efficiency,
                "paired_bootstrap_delta_95": interval,
                "injection_bootstrap_independence": bootstrap[
                    "independence_audit"
                ],
                "efficiency_noninferiority_passed": efficiency_gate,
                "far_robustness_passed": far_gate,
                "passed": bool(efficiency_gate and far_gate),
                "detector_strata": per_detector,
                "receipt": _artifact(receipt_path),
            }
        )
    required_detector_subsets_covered = all(
        detector_strata.get(subset, {}).get("scenario_count")
        == len(scenario_results)
        for subset in required_detector_subsets
    )
    required_detector_subset_minimums_passed = all(
        int(detector_strata.get(subset, {}).get("injections", 0))
        >= minimum_injections_per_detector_subset
        for subset in required_detector_subsets
    )
    passed = all(row["passed"] for row in scenario_results) and (
        required_detector_subsets_covered
        and required_detector_subset_minimums_passed
    )
    injection_bootstrap_independence = scenario_results[0][
        "injection_bootstrap_independence"
    ]
    result = {
        "status": "completed_validation_calibration_perturbation_robustness",
        "passed": passed,
        "scientific_claim_allowed": False,
        "locked_test_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "protocol": "all frozen physical perturbations evaluated at one unchanged validation threshold",
        "threshold": threshold,
        "threshold_source": "baseline_validation_candidate_search_calibration_only",
        "scenario_threshold_refits": 0,
        "target_far_per_year": target_far,
        "policy": {
            "minimum_scenarios": minimum_scenarios,
            "maximum_absolute_weighted_efficiency_loss": maximum_loss,
            "maximum_far_multiplier_of_target": maximum_far_multiplier,
            "bootstrap_replicates": bootstrap_replicates,
            "minimum_injection_gps_blocks": minimum_injection_gps_blocks,
            "required_detector_subsets": required_detector_subsets,
            "minimum_injections_per_detector_subset": (
                minimum_injections_per_detector_subset
            ),
            "seed": seed,
        },
        "scenario_count": len(scenario_results),
        "scenario_results": scenario_results,
        "injection_bootstrap_independence": injection_bootstrap_independence,
        "detector_strata": dict(sorted(detector_strata.items())),
        "detector_strata_audited": dict(sorted(detector_strata.items())),
        "required_detector_subsets": required_detector_subsets,
        "required_detector_subsets_covered": (
            required_detector_subsets_covered
        ),
        "minimum_injections_per_detector_subset": (
            minimum_injections_per_detector_subset
        ),
        "required_detector_subset_minimums_passed": (
            required_detector_subset_minimums_passed
        ),
        "physical_time_domain_perturbation": True,
        "fresh_time_frequency_transform": True,
        "envelope_interpretation": plan.get("envelope_interpretation"),
        "plan": _artifact(plan_file),
        "baseline_calibration": _artifact(baseline_file),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result
