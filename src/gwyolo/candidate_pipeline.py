from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .candidates import (
    run_apply_candidate_timing_calibration,
    run_candidate_extraction,
    run_candidate_time_slides,
    run_candidate_timing_calibration,
    run_injection_candidate_extraction,
    run_injection_candidate_rankings,
)
from .coherence import _pair_limit
from .injection_score import score_materialized_injections
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance
from .search import run_candidate_search_calibration
from .trigger import score_background_manifest


def select_candidate_timing_method(calibration: dict[str, Any]) -> tuple[str, float]:
    """Select the one publication-eligible per-cluster timing method."""

    passed = [
        (str(name), values)
        for name, values in calibration.get("methods", {}).items()
        if bool(values.get("calibration_gate_passed"))
        and str(name) == "local_whitened_strain_envelope_per_mask_cluster_v1"
    ]
    if len(passed) != 1:
        raise ValueError(
            "validation timing requires exactly one passing local per-cluster strain method"
        )
    name, values = passed[0]
    uncertainty = float(values["empirical_timing_uncertainty_seconds"])
    if uncertainty < 0:
        raise ValueError("selected empirical timing uncertainty is invalid")
    return name, uncertainty


def run_candidate_validation_pipeline(
    background_manifest: str | Path,
    injection_manifest: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    coherence_config: str | Path,
    output_dir: str | Path,
    reference_ifo: str = "H1",
    second_ifo: str = "L1",
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
    chirp_threshold: float = 0.3,
    minimum_bins: int = 1,
    timing_association_window_seconds: float = 0.25,
    timing_uncertainty_quantile: float = 0.99,
    minimum_timing_matches: int = 30,
    truth_association_window_seconds: float = 0.25,
    slide_count: int = 512,
    slide_step_seconds: float = 8.0,
    cluster_window_seconds: float = 0.1,
    target_far_per_year: float = 100.0,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Run the complete validation-only clustered candidate search chain."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "background_manifest_sha256": file_sha256(background_manifest),
        "injection_manifest_sha256": file_sha256(injection_manifest),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_sha256": file_sha256(config),
        "coherence_config_sha256": file_sha256(coherence_config),
        "reference_ifo": reference_ifo,
        "second_ifo": second_ifo,
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "context_duration": context_duration,
        "chirp_threshold": chirp_threshold,
        "minimum_bins": minimum_bins,
        "timing_association_window_seconds": timing_association_window_seconds,
        "timing_uncertainty_quantile": timing_uncertainty_quantile,
        "minimum_timing_matches": minimum_timing_matches,
        "truth_association_window_seconds": truth_association_window_seconds,
        "slide_count": slide_count,
        "slide_step_seconds": slide_step_seconds,
        "cluster_window_seconds": cluster_window_seconds,
        "target_far_per_year": target_far_per_year,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_validation_pipeline_report.json"
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("run_identity") != run_identity:
            raise ValueError("completed candidate validation pipeline has another identity")
        return prior
    background_score = score_background_manifest(
        background_manifest,
        checkpoint,
        config,
        output / "background_score",
        model_ifos,
        q_values,
        target_sample_rate,
        context_duration,
        True,
        "val",
        None,
        coherence_config,
    )
    injection_score = score_materialized_injections(
        injection_manifest,
        checkpoint,
        config,
        output / "injection_score",
        model_ifos,
        q_values,
        target_sample_rate,
        True,
        "val",
        None,
        coherence_config,
    )
    background_candidates = run_candidate_extraction(
        background_score["triggers_path"],
        output / "background_candidates",
        chirp_threshold,
        minimum_bins,
    )
    injection_candidates = run_injection_candidate_extraction(
        injection_score["triggers_path"],
        output / "injection_candidates",
        chirp_threshold,
        minimum_bins,
    )
    timing_report_path = output / "candidate_timing_calibration.json"
    timing = run_candidate_timing_calibration(
        injection_score["triggers_path"],
        timing_report_path,
        chirp_threshold,
        minimum_bins,
        timing_association_window_seconds,
        timing_uncertainty_quantile,
        minimum_timing_matches,
    )
    timing_method, timing_uncertainty = select_candidate_timing_method(timing)
    calibrated_background_path = output / "background_candidates_calibrated.jsonl"
    calibrated_injection_path = output / "injection_candidates_calibrated.jsonl"
    calibrated_background = run_apply_candidate_timing_calibration(
        background_candidates["manifest_path"],
        timing_report_path,
        calibrated_background_path,
    )
    calibrated_injection = run_apply_candidate_timing_calibration(
        injection_candidates["manifest_path"],
        timing_report_path,
        calibrated_injection_path,
    )
    if calibrated_background["uncalibrated_candidates"] or calibrated_injection[
        "uncalibrated_candidates"
    ]:
        raise RuntimeError("not every candidate received the frozen timing calibration")
    physics = load_yaml(coherence_config)["physics_coherent_pilot"]["coherence"]
    physical_delay = _pair_limit(
        reference_ifo,
        second_ifo,
        {
            str(pair): float(value)
            for pair, value in physics["maximum_pair_delay_seconds"].items()
        },
    )
    coincidence_window = physical_delay + 2.0 * timing_uncertainty
    slides = run_candidate_time_slides(
        calibrated_background_path,
        background_manifest,
        output / "time_slides",
        "val",
        reference_ifo,
        second_ifo,
        slide_count,
        slide_step_seconds,
        coincidence_window,
        cluster_window_seconds,
        physical_delay,
        timing_uncertainty,
    )
    injection_rankings = run_injection_candidate_rankings(
        injection_score["triggers_path"],
        calibrated_injection_path,
        output / "injection_rankings",
        "val",
        reference_ifo,
        second_ifo,
        physical_delay,
        timing_uncertainty,
        truth_association_window_seconds,
    )
    frozen_threshold_path = output / "frozen_candidate_search_calibration.json"
    frozen = run_candidate_search_calibration(
        output / "time_slides" / "val_candidate_time_slide_report.json",
        output / "injection_rankings" / "val_injection_candidate_ranking_report.json",
        target_far_per_year,
        frozen_threshold_path,
        bootstrap_replicates,
        seed,
    )
    result = {
        "status": "validation_only_clustered_candidate_search_pipeline",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "validation chain is frozen, but independent locked-test background/injections, "
            "adequate IFAR exposure and five-seed evidence remain required"
        ),
        "test_evaluation": None,
        "run_identity": run_identity,
        "timing_method": timing_method,
        "empirical_timing_uncertainty_seconds": timing_uncertainty,
        "physical_delay_limit_seconds": physical_delay,
        "coincidence_window_seconds": coincidence_window,
        "background_score_report_sha256": file_sha256(
            output / "background_score" / "trigger_score_report.json"
        ),
        "injection_score_report_sha256": file_sha256(
            output / "injection_score" / "injection_score_report.json"
        ),
        "timing_calibration_report_sha256": file_sha256(timing_report_path),
        "time_slide_report_sha256": file_sha256(
            output / "time_slides" / "val_candidate_time_slide_report.json"
        ),
        "injection_ranking_report_sha256": file_sha256(
            output / "injection_rankings" / "val_injection_candidate_ranking_report.json"
        ),
        "frozen_calibration_report_sha256": file_sha256(frozen_threshold_path),
        "timing_calibration": timing,
        "time_slides": slides,
        "injection_rankings": injection_rankings,
        "frozen_search": frozen,
        "pipeline_hash": canonical_hash(run_identity, 64),
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result
