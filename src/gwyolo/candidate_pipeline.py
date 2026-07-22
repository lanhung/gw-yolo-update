from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .candidates import (
    run_apply_candidate_timing_calibration,
    run_candidate_block_permutations,
    run_candidate_extraction,
    run_candidate_time_slides,
    run_candidate_timing_calibration,
    run_injection_candidate_extraction,
    run_injection_candidate_rankings,
)
from .coherence import _pair_limit
from .exposure import (
    CANDIDATE_BLOCK_PERMUTATION_METHOD,
    freeze_candidate_block_permutation_schedule,
)
from .injection_score import score_materialized_injections
from .injection_bootstrap import hierarchical_injection_bootstrap
from .io import atomic_write_json, canonical_hash, file_sha256, load_yaml
from .runtime import execution_provenance
from .search import run_candidate_search_calibration
from .streaming import evict_candidate_probability_artifacts
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


def validate_candidate_model_selection(
    selection_report_path: str | Path,
    checkpoint: str | Path,
    config: str | Path,
) -> dict[str, str]:
    """Bind candidate calibration to the validation-selected five-seed checkpoint."""

    path = Path(selection_report_path)
    selection = json.loads(path.read_text(encoding="utf-8"))
    if (
        selection.get("status")
        != "completed_five_seed_source_safe_overlap_validation"
        or not selection.get("passed")
        or selection.get("five_seed_stability", {}).get("status")
        != "five_seed_reproducibility_gate_v1"
        or selection.get("five_seed_stability", {}).get("passed") is not True
        or selection.get("test_data_opened") is not False
    ):
        raise ValueError("Candidate model selection is not a locked five-seed validation report")
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != str(selection.get("selected_checkpoint_sha256")):
        raise ValueError("Candidate checkpoint differs from five-seed selection")
    expected_config = selection.get("common_artifact_hashes", {}).get(
        "config_file_sha256"
    )
    if not expected_config or file_sha256(config) != str(expected_config):
        raise ValueError("Candidate config differs from five-seed selection")
    return {
        "model_selection_report_path": str(path),
        "model_selection_report_sha256": file_sha256(path),
        "selected_checkpoint_sha256": checkpoint_hash,
        "selected_config_file_sha256": str(expected_config),
    }


def compare_candidate_validation_pipelines(
    baseline_report_path: str | Path,
    promoted_report_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Gate continuous-background scaling with a paired validation comparison."""

    reports = {
        "baseline": json.loads(Path(baseline_report_path).read_text(encoding="utf-8")),
        "promoted": json.loads(Path(promoted_report_path).read_text(encoding="utf-8")),
    }
    if any(
        report.get("status") != "validation_only_clustered_candidate_search_pipeline"
        or report.get("test_evaluation") is not None
        for report in reports.values()
    ):
        raise ValueError("Candidate comparison requires validation-only completed pipelines")
    config = load_yaml(config_path)
    settings = config.get("candidate_validation_promotion")
    if not isinstance(settings, dict):
        raise ValueError("Candidate validation promotion configuration is missing")
    controlled_fields = (
        "background_manifest_sha256",
        "injection_manifest_sha256",
        "coherence_config_sha256",
        "reference_ifo",
        "second_ifo",
        "model_ifos",
        "q_values",
        "target_sample_rate",
        "context_duration",
        "chirp_threshold",
        "minimum_bins",
        "timing_association_window_seconds",
        "timing_uncertainty_quantile",
        "minimum_timing_matches",
        "maximum_timing_uncertainty_seconds",
        "truth_association_window_seconds",
        "slide_count",
        "slide_step_seconds",
        "cluster_window_seconds",
        "target_far_per_year",
        "bootstrap_replicates",
        "seed",
        "code_commit",
    )
    mismatches = {
        field: [reports[name]["run_identity"].get(field) for name in reports]
        for field in controlled_fields
        if len(
            {
                json.dumps(reports[name]["run_identity"].get(field), sort_keys=True)
                for name in reports
            }
        )
        != 1
    }
    if mismatches:
        raise ValueError(f"Candidate validation pipelines are not paired: {mismatches}")
    if reports["promoted"].get("model_selection") is None:
        raise ValueError("Promoted candidate pipeline lacks five-seed model selection")
    resampling_methods = {
        report.get("background_resampling_method") for report in reports.values()
    }
    if resampling_methods != {None} and resampling_methods != {
        CANDIDATE_BLOCK_PERMUTATION_METHOD
    }:
        raise ValueError("Candidate pipelines use different background resampling methods")

    ranking_rows = {}
    for name, report in reports.items():
        ranking = report["injection_rankings"]
        manifest = Path(ranking["manifest_path"])
        if file_sha256(manifest) != str(ranking["manifest_sha256"]):
            raise ValueError(f"Candidate {name} injection ranking hash mismatch")
        with manifest.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != int(ranking["ranked_injections"]):
            raise ValueError(f"Candidate {name} injection ranking count mismatch")
        by_id = {str(row["injection_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Candidate {name} rankings repeat injection IDs")
        ranking_rows[name] = by_id
    if set(ranking_rows["baseline"]) != set(ranking_rows["promoted"]):
        raise ValueError("Candidate pipelines evaluated different injection IDs")

    ids = sorted(ranking_rows["baseline"])
    for injection_id in ids:
        baseline = ranking_rows["baseline"][injection_id]
        promoted = ranking_rows["promoted"][injection_id]
        for field in (
            "waveform_id",
            "gps_block",
            "source_family",
            "stratum",
            "vt_weight",
            "vt_weight_unit",
        ):
            if baseline.get(field) != promoted.get(field):
                raise ValueError(
                    f"Candidate paired injection metadata differs for {injection_id}: {field}"
                )
    thresholds = {
        name: float(report["frozen_search"]["calibration"]["threshold"])
        for name, report in reports.items()
    }
    weights = np.asarray(
        [float(ranking_rows["baseline"][value]["vt_weight"]) for value in ids],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Candidate comparison has invalid VT weights")
    recovered = {
        name: np.asarray(
            [
                float(ranking_rows[name][value]["ranking_score"]) >= thresholds[name]
                for value in ids
            ],
            dtype=np.float64,
        )
        for name in reports
    }
    denominator = float(weights.sum())
    efficiencies = {
        name: float((weights * values).sum() / denominator)
        for name, values in recovered.items()
    }
    contributions = weights * (recovered["promoted"] - recovered["baseline"])
    delta = float(contributions.sum() / denominator)
    replicates = int(settings["bootstrap_replicates"])
    seed = int(settings["seed"])
    minimum_injection_gps_blocks = int(
        settings.get("minimum_injection_gps_blocks", 25)
    )
    if replicates <= 0 or minimum_injection_gps_blocks < 2:
        raise ValueError("Candidate promotion bootstrap count must be positive")
    bootstrap = hierarchical_injection_bootstrap(
        [ranking_rows["baseline"][value] for value in ids],
        contributions,
        weights,
        replicates,
        seed,
        require_physical_groups=True,
        minimum_physical_groups=minimum_injection_gps_blocks,
    )
    interval = bootstrap["interval_95"]
    strata = {}
    maximum_regression = float(settings["maximum_stratum_efficiency_regression"])
    regressed = []
    for stratum in sorted(
        {str(ranking_rows["baseline"][value].get("stratum", "all")) for value in ids}
    ):
        indices = np.asarray(
            [
                index
                for index, value in enumerate(ids)
                if str(ranking_rows["baseline"][value].get("stratum", "all"))
                == stratum
            ],
            dtype=np.int64,
        )
        stratum_weight = float(weights[indices].sum())
        values = {
            name: float((weights[indices] * recovered[name][indices]).sum() / stratum_weight)
            for name in reports
        }
        stratum_delta = values["promoted"] - values["baseline"]
        if stratum_delta < -maximum_regression:
            regressed.append(stratum)
        strata[stratum] = {
            "injections": int(indices.size),
            "weight": stratum_weight,
            "baseline_weighted_efficiency": values["baseline"],
            "promoted_weighted_efficiency": values["promoted"],
            "delta": stratum_delta,
        }
    timing = {
        name: float(report["empirical_timing_uncertainty_seconds"])
        for name, report in reports.items()
    }
    exposure_checks = {
        name: bool(report["frozen_search"]["publication_calibration_eligible"])
        for name, report in reports.items()
    }
    gates = {
        "paired_population": True,
        "complete_exposure_schedule": all(exposure_checks.values()),
        "minimum_weighted_efficiency_gain": delta
        >= float(settings["minimum_weighted_efficiency_gain"]),
        "paired_bootstrap_lower_bound_positive": interval[0] > 0,
        "injection_bootstrap_independence": bootstrap["independence_audit"][
            "passed"
        ],
        "maximum_regressed_strata": len(regressed)
        <= int(settings["maximum_regressed_strata"]),
        "promoted_timing_limit": timing["promoted"]
        <= float(settings["maximum_promoted_timing_uncertainty_seconds"]),
        "timing_regression": timing["promoted"] - timing["baseline"]
        <= float(settings["maximum_timing_uncertainty_regression_seconds"]),
    }
    result = {
        "status": "paired_validation_candidate_search_promotion",
        "passed": all(gates.values()),
        "scale_continuous_background": all(gates.values()),
        "scientific_claim_allowed": False,
        "test_data_opened": False,
        "gates": gates,
        "target_far_per_year": reports["baseline"]["run_identity"][
            "target_far_per_year"
        ],
        "thresholds": thresholds,
        "weighted_efficiencies": efficiencies,
        "weighted_efficiency_delta_promoted_minus_baseline": delta,
        "paired_bootstrap_95": interval,
        "bootstrap_replicates": replicates,
        "minimum_injection_gps_blocks": minimum_injection_gps_blocks,
        "injection_bootstrap_independence": bootstrap["independence_audit"],
        "seed": seed,
        "timing_uncertainty_seconds": timing,
        "regressed_strata": regressed,
        "strata": strata,
        "input_report_hashes": {
            "baseline": file_sha256(baseline_report_path),
            "promoted": file_sha256(promoted_report_path),
        },
        "config_path": str(config_path),
        "config_hash": canonical_hash(config),
        **execution_provenance(),
    }
    atomic_write_json(output_path, result)
    return result


def recalibrate_candidate_validation_pipeline_with_block_permutations(
    pipeline_report_path: str | Path,
    background_manifest: str | Path,
    calibrated_candidate_manifest: str | Path,
    injection_ranking_report: str | Path,
    output_dir: str | Path,
    zero_count_confidence: float = 0.90,
) -> dict[str, Any]:
    """Replace an engineering absolute-slide calibration with a frozen block schedule."""

    source_path = Path(pipeline_report_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("status") != "validation_only_clustered_candidate_search_pipeline"
        or source.get("test_evaluation") is not None
    ):
        raise ValueError("block recalibration requires a validation-only candidate pipeline")
    identity = source.get("run_identity", {})
    if file_sha256(background_manifest) != str(
        identity.get("background_manifest_sha256")
    ):
        raise ValueError("block recalibration background differs from the pipeline")
    if file_sha256(calibrated_candidate_manifest) != str(
        source.get("time_slides", {}).get("candidate_manifest_sha256")
    ):
        raise ValueError("block recalibration candidates differ from the pipeline")
    if file_sha256(injection_ranking_report) != str(
        source.get("injection_ranking_report_sha256")
    ):
        raise ValueError("block recalibration injection rankings differ from the pipeline")
    target_far = float(identity["target_far_per_year"])
    reference_ifo = str(identity["reference_ifo"])
    shifted_ifo = str(identity["second_ifo"])
    physical_delay = float(source["physical_delay_limit_seconds"])
    timing_uncertainty = float(source["empirical_timing_uncertainty_seconds"])
    coincidence_window = float(source["coincidence_window_seconds"])
    cluster_window = float(identity["cluster_window_seconds"])
    if not np.isclose(
        coincidence_window,
        physical_delay + 2 * timing_uncertainty,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("block recalibration timing contract differs from physics")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "candidate_validation_block_pipeline_report.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = prior.get("block_permutation_recalibration", {})
        if (
            metadata.get("source_pipeline_report_sha256")
            != file_sha256(source_path)
            or metadata.get("background_manifest_sha256")
            != file_sha256(background_manifest)
            or metadata.get("candidate_manifest_sha256")
            != file_sha256(calibrated_candidate_manifest)
            or metadata.get("injection_ranking_report_sha256")
            != file_sha256(injection_ranking_report)
            or not np.isclose(
                float(metadata.get("zero_count_confidence", -1)),
                zero_count_confidence,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("completed block-recalibrated pipeline has another identity")
        return prior

    schedule_path = output / "candidate_block_permutation_schedule.json"
    if not schedule_path.is_file():
        freeze_candidate_block_permutation_schedule(
            background_manifest,
            schedule_path,
            "val",
            reference_ifo,
            shifted_ifo,
            target_far,
            zero_count_confidence,
        )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if (
        not np.isclose(
            float(schedule.get("target_far_per_year", -1)),
            target_far,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            float(schedule.get("zero_count_confidence", -1)),
            zero_count_confidence,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("existing block schedule has another FAR target")
    block_dir = output / "candidate_block_background"
    block = run_candidate_block_permutations(
        calibrated_candidate_manifest,
        background_manifest,
        schedule_path,
        block_dir,
        "val",
        reference_ifo,
        shifted_ifo,
        coincidence_window,
        cluster_window,
        physical_delay,
        timing_uncertainty,
    )
    block_report_path = block_dir / "val_candidate_time_slide_report.json"
    calibration_path = output / "frozen_candidate_search_calibration.json"
    if calibration_path.is_file():
        frozen = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            frozen.get("validation_time_slide_report_sha256")
            != file_sha256(block_report_path)
            or frozen.get("validation_injection_ranking_report_sha256")
            != file_sha256(injection_ranking_report)
            or not np.isclose(
                float(frozen.get("target_far_per_year", -1)),
                target_far,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("existing block calibration has another identity")
    else:
        frozen = run_candidate_search_calibration(
            block_report_path,
            injection_ranking_report,
            target_far,
            calibration_path,
            int(identity["bootstrap_replicates"]),
            int(identity["seed"]),
            background_manifest,
        )
    result = {
        **source,
        "scientific_blocker": (
            "block-permutation validation calibration is frozen; paired model promotion, "
            "independent locked-test background and injections remain required"
        ),
        "time_slides": block,
        "frozen_search": frozen,
        "time_slide_report_sha256": file_sha256(block_report_path),
        "frozen_calibration_report_sha256": file_sha256(calibration_path),
        "background_resampling_method": block["background_pairing_method"],
        "block_permutation_recalibration": {
            "source_pipeline_report_path": str(source_path),
            "source_pipeline_report_sha256": file_sha256(source_path),
            "background_manifest_path": str(background_manifest),
            "background_manifest_sha256": file_sha256(background_manifest),
            "candidate_manifest_path": str(calibrated_candidate_manifest),
            "candidate_manifest_sha256": file_sha256(calibrated_candidate_manifest),
            "injection_ranking_report_path": str(injection_ranking_report),
            "injection_ranking_report_sha256": file_sha256(
                injection_ranking_report
            ),
            "schedule_path": str(schedule_path),
            "schedule_sha256": file_sha256(schedule_path),
            "block_report_path": str(block_report_path),
            "block_report_sha256": file_sha256(block_report_path),
            "calibration_path": str(calibration_path),
            "calibration_sha256": file_sha256(calibration_path),
            "zero_count_confidence": zero_count_confidence,
            **execution_provenance(),
        },
    }
    atomic_write_json(result_path, result)
    return result


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
    maximum_timing_uncertainty_seconds: float = 0.01,
    truth_association_window_seconds: float = 0.25,
    slide_count: int = 512,
    slide_step_seconds: float = 8.0,
    cluster_window_seconds: float = 0.1,
    target_far_per_year: float = 100.0,
    bootstrap_replicates: int = 10000,
    seed: int = 20260720,
    model_selection_report: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete validation-only clustered candidate search chain."""

    model_selection = (
        validate_candidate_model_selection(
            model_selection_report, checkpoint, config
        )
        if model_selection_report is not None
        else None
    )
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
        "maximum_timing_uncertainty_seconds": maximum_timing_uncertainty_seconds,
        "truth_association_window_seconds": truth_association_window_seconds,
        "slide_count": slide_count,
        "slide_step_seconds": slide_step_seconds,
        "cluster_window_seconds": cluster_window_seconds,
        "target_far_per_year": target_far_per_year,
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "model_selection_report_sha256": (
            model_selection["model_selection_report_sha256"]
            if model_selection is not None
            else None
        ),
        "code_commit": execution_provenance()["code_commit"],
    }
    report_path = output / "candidate_validation_pipeline_report.json"
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if prior.get("run_identity") != run_identity:
            raise ValueError("completed candidate validation pipeline has another identity")
        return prior
    background_score_path = output / "background_score" / "trigger_score_report.json"
    background_candidate_path = (
        output / "background_candidates" / "candidate_extraction_report.json"
    )
    background_eviction_path = output / "background_probability_eviction.json"
    if background_eviction_path.is_file():
        with background_eviction_path.open("r", encoding="utf-8") as handle:
            background_eviction = json.load(handle)
        with background_score_path.open("r", encoding="utf-8") as handle:
            background_score = json.load(handle)
        with background_candidate_path.open("r", encoding="utf-8") as handle:
            background_candidates = json.load(handle)
        if (
            background_eviction.get("status")
            != "verified_candidate_probability_eviction"
            or background_eviction.get("score_report_sha256")
            != file_sha256(background_score_path)
            or background_eviction.get("candidate_extraction_report_sha256")
            != file_sha256(background_candidate_path)
        ):
            raise ValueError("existing background probability eviction is inconsistent")
    else:
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
        background_candidates = run_candidate_extraction(
            background_score["triggers_path"],
            output / "background_candidates",
            chirp_threshold,
            minimum_bins,
        )
        background_eviction = evict_candidate_probability_artifacts(
            background_candidate_path,
            background_score_path,
            output / "background_score" / "probabilities",
            background_eviction_path,
        )

    timing_report_path = output / "candidate_timing_calibration.json"
    injection_score_path = output / "injection_score" / "injection_score_report.json"
    injection_candidate_path = (
        output / "injection_candidates" / "injection_candidate_extraction_report.json"
    )
    injection_eviction_path = output / "injection_probability_eviction.json"
    if injection_eviction_path.is_file():
        with injection_eviction_path.open("r", encoding="utf-8") as handle:
            injection_eviction = json.load(handle)
        with injection_score_path.open("r", encoding="utf-8") as handle:
            injection_score = json.load(handle)
        with injection_candidate_path.open("r", encoding="utf-8") as handle:
            injection_candidates = json.load(handle)
        with timing_report_path.open("r", encoding="utf-8") as handle:
            timing = json.load(handle)
        if (
            injection_eviction.get("status")
            != "verified_candidate_probability_eviction"
            or injection_eviction.get("score_report_sha256")
            != file_sha256(injection_score_path)
            or injection_eviction.get("candidate_extraction_report_sha256")
            != file_sha256(injection_candidate_path)
        ):
            raise ValueError("existing injection probability eviction is inconsistent")
    else:
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
        injection_candidates = run_injection_candidate_extraction(
            injection_score["triggers_path"],
            output / "injection_candidates",
            chirp_threshold,
            minimum_bins,
        )
        timing = run_candidate_timing_calibration(
            injection_score["triggers_path"],
            timing_report_path,
            chirp_threshold,
            minimum_bins,
            timing_association_window_seconds,
            timing_uncertainty_quantile,
            minimum_timing_matches,
            maximum_timing_uncertainty_seconds,
        )
        injection_eviction = evict_candidate_probability_artifacts(
            injection_candidate_path,
            injection_score_path,
            output / "injection_score" / "probabilities",
            injection_eviction_path,
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
        "model_selection": model_selection,
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
        "background_probability_eviction_report_sha256": file_sha256(
            background_eviction_path
        ),
        "injection_probability_eviction_report_sha256": file_sha256(
            injection_eviction_path
        ),
        "probability_files_removed": int(background_eviction["removed_files"])
        + int(injection_eviction["removed_files"]),
        "probability_bytes_removed": int(background_eviction["removed_bytes"])
        + int(injection_eviction["removed_bytes"]),
        "timing_calibration": timing,
        "time_slides": slides,
        "injection_rankings": injection_rankings,
        "frozen_search": frozen,
        "pipeline_hash": canonical_hash(run_identity, 64),
        **execution_provenance(),
    }
    atomic_write_json(report_path, result)
    return result
