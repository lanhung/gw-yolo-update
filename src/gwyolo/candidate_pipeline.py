from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .candidates import (
    run_apply_candidate_timing_calibration,
    run_candidate_block_permutations,
    run_detector_set_candidate_block_permutations,
    run_detector_set_injection_candidate_rankings,
    run_candidate_extraction,
    run_candidate_time_slides,
    run_candidate_timing_calibration,
    run_injection_candidate_extraction,
    run_injection_candidate_rankings,
)
from .coherence import _pair_limit
from .exposure import (
    CANDIDATE_BLOCK_PERMUTATION_METHOD,
    DETECTOR_SET_BLOCK_PERMUTATION_METHOD,
    freeze_candidate_block_permutation_schedule,
    freeze_detector_set_block_permutation_schedule,
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


def freeze_raw_mask_detector_set_ranking_successor(
    mask_timing_receipt: str | Path,
    raw_variable_ranking_report: str | Path,
    mask_variable_ranking_report: str | Path,
    network_config: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Bind raw/mask H1/L1/V1 rankings to their frozen timing artifacts."""

    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(
            "raw/mask detector-set ranking successors are immutable"
        )
    timing_path = Path(mask_timing_receipt).resolve()
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    config_path = Path(network_config).resolve()
    config = load_yaml(config_path)
    policy = config.get("network_coherence", {})
    if (
        timing.get("status")
        != "completed_validation_only_mask_timing_gate"
        or timing.get("coherent_background_scale_allowed") is not True
        or timing.get("raw_timing_gate_passed") is not True
        or timing.get("mask_timing_gate_passed") is not True
        or int(timing.get("test_rows_read", -1)) != 0
        or timing.get("locked_test_allowed") is not False
        or policy.get("schema")
        != "h1_l1_v1_pairwise_light_travel_v1"
    ):
        raise ValueError(
            "raw/mask detector-set successor requires frozen timing/network gates"
        )
    variable_paths = {
        "raw": Path(raw_variable_ranking_report).resolve(),
        "mask": Path(mask_variable_ranking_report).resolve(),
    }
    arms = {}
    for arm, variable_path in variable_paths.items():
        timing_identity = timing.get("timing_reports", {}).get(arm, {})
        timing_report_path = Path(
            str(timing_identity.get("path", ""))
        ).resolve()
        source_identity = timing.get(
            "injection_ranking_reports",
            {},
        ).get(arm, {})
        source_path = Path(
            str(source_identity.get("path", ""))
        ).resolve()
        candidate_report_path = Path(
            str(
                source_identity.get(
                    "candidate_extraction_report_path",
                    "",
                )
            )
        ).resolve()
        score_identity = timing.get(f"{arm}_score_report", {})
        score_path = Path(str(score_identity.get("path", ""))).resolve()
        if (
            not timing_report_path.is_file()
            or timing_identity.get("sha256")
            != file_sha256(timing_report_path)
            or not source_path.is_file()
            or source_identity.get("sha256") != file_sha256(source_path)
            or not candidate_report_path.is_file()
            or source_identity.get("candidate_extraction_report_sha256")
            != file_sha256(candidate_report_path)
            or not score_path.is_file()
            or score_identity.get("sha256") != file_sha256(score_path)
            or not variable_path.is_file()
        ):
            raise ValueError(
                f"{arm} raw/mask detector-set source replay failed"
            )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        score = json.loads(score_path.read_text(encoding="utf-8"))
        variable = json.loads(variable_path.read_text(encoding="utf-8"))
        trigger_path = Path(
            str(score.get("triggers_path", ""))
        ).resolve()
        calibrated_path = (
            candidate_report_path.parent.parent
            / f"{arm}_injection_candidates_calibrated.jsonl"
        ).resolve()
        manifest_path = Path(
            str(variable.get("manifest_path", ""))
        ).resolve()
        if (
            source.get("status")
            != "physical_network_injection_candidate_rankings"
            or source.get("split") != "val"
            or not trigger_path.is_file()
            or source.get("injection_trigger_manifest_sha256")
            != file_sha256(trigger_path)
            or not calibrated_path.is_file()
            or source.get("candidate_manifest_sha256")
            != file_sha256(calibrated_path)
            or variable.get("status")
            != "physical_variable_detector_set_injection_candidate_rankings"
            or variable.get("split") != "val"
            or variable.get("config_sha256") != file_sha256(config_path)
            or variable.get("injection_trigger_manifest_sha256")
            != file_sha256(trigger_path)
            or variable.get("candidate_manifest_sha256")
            != file_sha256(calibrated_path)
            or variable.get("timing_calibration_report_sha256")
            != file_sha256(timing_report_path)
            or variable.get("candidate_checkpoint_sha256")
            != source.get("candidate_checkpoint_sha256")
            or variable.get("candidate_config_sha256")
            != source.get("candidate_config_sha256")
            or variable.get("candidate_code_commit")
            != source.get("candidate_code_commit")
            or variable.get("timing_calibration_consistent") is not True
            or variable.get("candidate_scoring_provenance_consistent")
            is not True
            or variable.get("required_detector_subsets")
            != [
                "+".join(str(value) for value in subset)
                for subset in policy["detector_subsets"]
            ]
            or not manifest_path.is_file()
            or variable.get("manifest_sha256")
            != file_sha256(manifest_path)
        ):
            raise ValueError(
                f"{arm} variable-detector ranking failed lineage replay"
            )
        arms[arm] = {
            "source_ranking_report": {
                "path": str(source_path),
                "sha256": file_sha256(source_path),
            },
            "variable_ranking_report": {
                "path": str(variable_path),
                "sha256": file_sha256(variable_path),
            },
            "injection_trigger_manifest": {
                "path": str(trigger_path),
                "sha256": file_sha256(trigger_path),
            },
            "calibrated_candidate_manifest": {
                "path": str(calibrated_path),
                "sha256": file_sha256(calibrated_path),
            },
            "timing_report": {
                "path": str(timing_report_path),
                "sha256": file_sha256(timing_report_path),
            },
        }
    result = {
        "status": "variable_detector_set_raw_mask_ranking_successor_v1",
        "scientific_claim_allowed": False,
        "test_rows_read": 0,
        "test_evaluation": None,
        "source_mask_timing_receipt": {
            "path": str(timing_path),
            "sha256": file_sha256(timing_path),
        },
        "network_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
            "config_hash": canonical_hash(config, 64),
        },
        "arms": arms,
        **execution_provenance(),
    }
    atomic_write_json(target, result)
    return result


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
        "network_config_sha256",
        "detector_set_policy",
        "detectors",
        "detector_subsets",
        "pairwise_light_travel_time_seconds",
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
    } and resampling_methods != {
        DETECTOR_SET_BLOCK_PERMUTATION_METHOD
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


def recalibrate_candidate_validation_pipeline_with_detector_sets(
    pipeline_report_path: str | Path,
    background_manifest: str | Path,
    calibrated_background_candidate_manifest: str | Path,
    injection_trigger_manifest: str | Path,
    calibrated_injection_candidate_manifest: str | Path,
    network_config: str | Path,
    output_dir: str | Path,
    zero_count_confidence: float = 0.90,
    maximum_shifts: int | None = None,
    exposure_safety_factor: float = 1.0,
    expanded_background_merge_report: str | Path | None = None,
    background_plan_authorization: str | Path | None = None,
) -> dict[str, Any]:
    """Promote a validation pipeline to the frozen H1/L1/V1 candidate policy."""

    source_path = Path(pipeline_report_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("status")
        != "validation_only_clustered_candidate_search_pipeline"
        or source.get("scientific_claim_allowed") is not False
        or source.get("test_evaluation") is not None
    ):
        raise ValueError(
            "detector-set recalibration requires a validation-only pipeline"
        )
    identity = source.get("run_identity", {})
    source_slides = source.get("time_slides", {})
    source_rankings = source.get("injection_rankings", {})
    expanded_mode = (
        expanded_background_merge_report is not None
        or background_plan_authorization is not None
    )
    if expanded_mode and (
        expanded_background_merge_report is None
        or background_plan_authorization is None
    ):
        raise ValueError(
            "expanded detector-set recalibration requires merge and authorization"
        )
    inputs = {
        "background_manifest_sha256": file_sha256(background_manifest),
        "background_candidate_manifest_sha256": file_sha256(
            calibrated_background_candidate_manifest
        ),
        "injection_trigger_manifest_sha256": file_sha256(
            injection_trigger_manifest
        ),
        "injection_candidate_manifest_sha256": file_sha256(
            calibrated_injection_candidate_manifest
        ),
        "network_config_sha256": file_sha256(network_config),
    }
    if (
        inputs["injection_trigger_manifest_sha256"]
        != source_rankings.get("injection_trigger_manifest_sha256")
        or inputs["injection_candidate_manifest_sha256"]
        != source_rankings.get("candidate_manifest_sha256")
    ):
        raise ValueError(
            "detector-set recalibration inputs differ from the source pipeline"
        )
    expanded_lineage = None
    if expanded_mode:
        merge_path = Path(expanded_background_merge_report).resolve()
        authorization_path = Path(background_plan_authorization).resolve()
        merge = json.loads(merge_path.read_text(encoding="utf-8"))
        authorization = json.loads(
            authorization_path.read_text(encoding="utf-8")
        )
        parent_path = Path(
            str(authorization.get("parent_plan", {}).get("path", ""))
        ).resolve()
        endpoint_path = Path(
            str(
                authorization.get(
                    "independent_validation_endpoint",
                    {},
                ).get("path", "")
            )
        ).resolve()
        if (
            authorization.get("status")
            != "authorized_validation_candidate_continuous_background_plan"
            or authorization.get("passed") is not True
            or authorization.get("scientific_claim_allowed") is not False
            or authorization.get("candidate_scores_inspected") is not False
            or int(authorization.get("test_rows_read", -1)) != 0
            or authorization.get("test_evaluation") is not None
            or not parent_path.is_file()
            or authorization.get("parent_plan", {}).get("sha256")
            != file_sha256(parent_path)
            or not endpoint_path.is_file()
            or authorization.get(
                "independent_validation_endpoint",
                {},
            ).get("sha256")
            != file_sha256(endpoint_path)
        ):
            raise ValueError(
                "expanded detector-set background is not authorized"
            )
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        candidate_artifact = merge.get("candidate_manifests", {}).get(
            "val",
            {},
        )
        common = merge.get("common_run_identity", {})
        expected_common = {
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "config_sha256": identity.get("config_sha256"),
            "coherence_config_sha256": identity.get(
                "coherence_config_sha256"
            ),
            "model_ifos": identity.get("model_ifos"),
            "q_values": identity.get("q_values"),
            "target_sample_rate": identity.get("target_sample_rate"),
            "context_duration": identity.get("context_duration"),
            "chirp_threshold": identity.get("chirp_threshold"),
            "minimum_bins": identity.get("minimum_bins"),
            "code_commit": identity.get("code_commit"),
        }
        if (
            merge.get("status")
            != "verified_merged_streamed_candidate_background"
            or merge.get("scientific_claim_allowed") is not False
            or merge.get("complete_parent_plan") is not True
            or int(merge.get("split_counts", {}).get("test", -1)) != 0
            or common.get("parent_plan_sha256")
            != file_sha256(parent_path)
            or any(
                common.get(field) != expected
                for field, expected in expected_common.items()
            )
            or common.get("timing_calibration_report_sha256")
            != source.get("timing_calibration_report_sha256")
            or merge.get("background_manifest_sha256")
            != inputs["background_manifest_sha256"]
            or Path(
                str(merge.get("background_manifest_path", ""))
            ).resolve()
            != Path(background_manifest).resolve()
            or candidate_artifact.get("sha256")
            != inputs["background_candidate_manifest_sha256"]
            or Path(
                str(candidate_artifact.get("path", ""))
            ).resolve()
            != Path(calibrated_background_candidate_manifest).resolve()
            or endpoint.get("status")
            != "frozen_gps_and_purpose_disjoint_validation_endpoint"
            or endpoint.get("passed") is not True
            or endpoint.get("scientific_claim_allowed") is not False
            or int(endpoint.get("test_rows_read", -1)) != 0
            or endpoint.get("test_evaluation") is not None
            or int(endpoint.get("purpose_gps_block_overlap", -1)) != 0
        ):
            raise ValueError(
                "expanded detector-set merge/scorer/endpoint lineage differs"
            )
        authorization_identity = authorization.get(
            "authorization_identity",
            {},
        )
        target_far = float(
            authorization_identity["target_far_per_year"]
        )
        zero_count_confidence = float(
            authorization_identity["zero_count_confidence"]
        )
        expanded_lineage = {
            "merge_report_path": str(merge_path),
            "merge_report_sha256": file_sha256(merge_path),
            "authorization_path": str(authorization_path),
            "authorization_sha256": file_sha256(authorization_path),
            "authorization_id": authorization["authorization_id"],
            "parent_plan_path": str(parent_path),
            "parent_plan_sha256": file_sha256(parent_path),
            "independent_validation_endpoint_path": str(endpoint_path),
            "independent_validation_endpoint_sha256": file_sha256(
                endpoint_path
            ),
        }
    else:
        if (
            inputs["background_manifest_sha256"]
            != identity.get("background_manifest_sha256")
            or inputs["background_candidate_manifest_sha256"]
            != source_slides.get("candidate_manifest_sha256")
        ):
            raise ValueError(
                "detector-set background differs from the source pipeline"
            )
        target_far = float(identity["target_far_per_year"])
    timing_uncertainty = float(
        source["empirical_timing_uncertainty_seconds"]
    )
    cluster_window = float(identity["cluster_window_seconds"])
    bootstrap_replicates = int(identity["bootstrap_replicates"])
    seed = int(identity["seed"])
    if (
        not 0 < zero_count_confidence < 1
        or not np.isfinite(exposure_safety_factor)
        or exposure_safety_factor < 1
    ):
        raise ValueError("detector-set recalibration settings are invalid")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = (
        output
        / "candidate_validation_detector_set_block_pipeline_report.json"
    )
    successor_identity = {
        "source_pipeline_report_sha256": file_sha256(source_path),
        **inputs,
        "zero_count_confidence": zero_count_confidence,
        "maximum_shifts": maximum_shifts,
        "exposure_safety_factor": exposure_safety_factor,
        "expanded_background_lineage": expanded_lineage,
    }
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            prior.get("detector_set_block_recalibration", {}).get(
                "successor_identity"
            )
            != successor_identity
        ):
            raise ValueError(
                "completed detector-set pipeline has another identity"
            )
        return prior

    schedule_path = (
        output / "detector_set_block_permutation_schedule.json"
    )
    if not schedule_path.is_file():
        freeze_detector_set_block_permutation_schedule(
            background_manifest,
            network_config,
            schedule_path,
            "val",
            target_far,
            zero_count_confidence,
            maximum_shifts,
            exposure_safety_factor,
        )
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if (
        schedule.get("background_manifest_sha256")
        != inputs["background_manifest_sha256"]
        or schedule.get("network_config_sha256")
        != inputs["network_config_sha256"]
        or not np.isclose(
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
        or not np.isclose(
            float(schedule.get("exposure_safety_factor", -1)),
            exposure_safety_factor,
            rtol=0.0,
            atol=1e-12,
        )
        or schedule.get("required_detector_subsets_covered") is not True
    ):
        raise ValueError(
            "detector-set block schedule identity or subset coverage differs"
        )

    background_dir = output / "detector_set_block_background"
    background = run_detector_set_candidate_block_permutations(
        calibrated_background_candidate_manifest,
        background_manifest,
        schedule_path,
        background_dir,
        timing_uncertainty,
        cluster_window,
    )
    background_report_path = (
        background_dir
        / "val_detector_set_block_permutation_report.json"
    )
    ranking_dir = output / "detector_set_injection_rankings"
    rankings = run_detector_set_injection_candidate_rankings(
        injection_trigger_manifest,
        calibrated_injection_candidate_manifest,
        network_config,
        ranking_dir,
        "val",
        timing_uncertainty,
    )
    ranking_report_path = (
        ranking_dir
        / "val_variable_detector_set_injection_candidate_ranking_report.json"
    )
    calibration_path = output / "frozen_candidate_search_calibration.json"
    if calibration_path.is_file():
        frozen = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            frozen.get("validation_time_slide_report_sha256")
            != file_sha256(background_report_path)
            or frozen.get("validation_injection_ranking_report_sha256")
            != file_sha256(ranking_report_path)
        ):
            raise ValueError(
                "existing detector-set calibration has another identity"
            )
    else:
        frozen = run_candidate_search_calibration(
            background_report_path,
            ranking_report_path,
            target_far,
            calibration_path,
            bootstrap_replicates,
            seed,
            background_manifest,
        )

    policy = load_yaml(network_config)["network_coherence"]
    run_identity = {
        **identity,
        "background_manifest_sha256": inputs[
            "background_manifest_sha256"
        ],
        "target_far_per_year": target_far,
        "network_config_sha256": inputs["network_config_sha256"],
        "detector_set_policy": policy["schema"],
        "detectors": list(policy["detectors"]),
        "detector_subsets": [
            list(value) for value in policy["detector_subsets"]
        ],
        "pairwise_light_travel_time_seconds": dict(
            policy["pairwise_light_travel_time_seconds"]
        ),
    }
    result = {
        **source,
        "scientific_blocker": (
            "variable-detector validation calibration is frozen; paired "
            "five-seed promotion and one-time locked O4b evaluation remain"
        ),
        "run_identity": run_identity,
        "time_slides": background,
        "injection_rankings": rankings,
        "frozen_search": frozen,
        "time_slide_report_sha256": file_sha256(
            background_report_path
        ),
        "injection_ranking_report_sha256": file_sha256(
            ranking_report_path
        ),
        "frozen_calibration_report_sha256": file_sha256(
            calibration_path
        ),
        "background_resampling_method": background[
            "background_pairing_method"
        ],
        "detector_set_block_recalibration": {
            "successor_identity": successor_identity,
            "expanded_background_lineage": expanded_lineage,
            "source_pipeline_report_path": str(source_path),
            "schedule_path": str(schedule_path.resolve()),
            "schedule_sha256": file_sha256(schedule_path),
            "background_report_path": str(
                background_report_path.resolve()
            ),
            "background_report_sha256": file_sha256(
                background_report_path
            ),
            "injection_ranking_report_path": str(
                ranking_report_path.resolve()
            ),
            "injection_ranking_report_sha256": file_sha256(
                ranking_report_path
            ),
            "calibration_path": str(calibration_path.resolve()),
            "calibration_sha256": file_sha256(calibration_path),
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
