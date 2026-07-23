from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.calibration import (
    apply_frequency_dependent_calibration_response,
    evaluate_calibration_perturbation_robustness,
    freeze_calibration_perturbation_plan,
    freeze_calibration_perturbation_scenario_result,
    load_calibration_perturbation_scenario,
    response_for_row,
)
from gwyolo.cli import main
from gwyolo.io import file_sha256


def _write_config(path: Path) -> None:
    path.write_text(
        """
calibration_perturbation:
  protocol: run_correlated_frequency_response_v1
  required_split: val
  default_observing_run: O4a
  seed: 1234
  target_sample_rate: 1024
  model_ifos: [H1, L1, V1]
  anchor_frequencies_hz: [20, 100, 300, 480]
  random_draws: 4
  envelope_templates:
    test_envelope:
      maximum_amplitude_fraction: [0.08, 0.05, 0.04, 0.08]
      maximum_phase_degrees: [8, 5, 4, 8]
      source:
        identity: hand_calculated_test_envelope
        semantics: bounded_stress_test_not_posterior
  run_template_assignments:
    O4a: test_envelope
""".lstrip(),
        encoding="utf-8",
    )


def _write_manifests(tmp_path: Path, shared_block: bool = False) -> tuple[Path, Path]:
    background = tmp_path / "background.jsonl"
    injection = tmp_path / "injection.jsonl"
    background.write_text(
        json.dumps(
            {
                "window_id": "window-1",
                "split": "val",
                "observing_run": "O4a",
                "gps_block": "O4a:block-background",
                "ifos": ["H1", "L1"],
                "ranking_score": 1e9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    injection.write_text(
        json.dumps(
            {
                "injection_id": "injection-1",
                "split": "val",
                "gps_block": ("O4a:block-background" if shared_block else "O4a:block-injection"),
                "ifos": ["H1", "L1", "V1"],
                "ranking_score": -1e9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return background, injection


def test_frequency_dependent_calibration_has_hand_calculated_constant_gain() -> None:
    sample_rate = 1024
    time = np.arange(1024) / sample_rate
    strain = np.sin(2 * np.pi * 64 * time)
    response = {
        "application": "multiplicative_rfft_strain_response_v1",
        "anchor_frequencies_hz": [20, 100, 300, 480],
        "amplitude_fraction": [0.1, 0.1, 0.1, 0.1],
        "phase_radians": [0.0, 0.0, 0.0, 0.0],
    }

    perturbed = apply_frequency_dependent_calibration_response(strain, sample_rate, response)

    np.testing.assert_allclose(perturbed, 1.1 * strain, rtol=0, atol=1e-12)


def test_calibration_plan_is_validation_only_score_blind_and_run_correlated(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    plan_path = tmp_path / "plan.json"
    _write_config(config)
    background, injection = _write_manifests(tmp_path)

    assert (
        main(
            [
                "calibration-perturbation-plan-freeze",
                "--background-manifest",
                str(background),
                "--injection-manifest",
                str(injection),
                "--config",
                str(config),
                "--output",
                str(plan_path),
            ]
        )
        == 0
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["candidate_scores_inspected"] is False
    assert plan["test_rows_read"] == 0
    assert plan["scenario_count"] == 7
    assert plan["purpose_gps_block_overlap"] == 0
    assert plan["envelope_interpretation"].endswith("not posterior calibration samples")

    background_scenario = load_calibration_perturbation_scenario(
        plan_path,
        background,
        "background",
        "random_draw_000",
        1024,
        ("H1", "L1", "V1"),
    )
    injection_scenario = load_calibration_perturbation_scenario(
        plan_path,
        injection,
        "injection",
        "random_draw_000",
        1024,
        ("H1", "L1", "V1"),
    )
    background_row = json.loads(background.read_text(encoding="utf-8"))
    injection_row = json.loads(injection.read_text(encoding="utf-8"))
    assert response_for_row(background_scenario, background_row, "H1") == response_for_row(
        injection_scenario, injection_row, "H1"
    )


def test_calibration_plan_rejects_purpose_overlap_and_changed_manifest(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    background, injection = _write_manifests(tmp_path, shared_block=True)
    with pytest.raises(ValueError, match="share GPS blocks"):
        freeze_calibration_perturbation_plan(
            background, injection, config, tmp_path / "overlap-plan.json"
        )

    background, injection = _write_manifests(tmp_path, shared_block=False)
    plan_path = tmp_path / "plan.json"
    freeze_calibration_perturbation_plan(background, injection, config, plan_path)
    with injection.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="manifest differs"):
        load_calibration_perturbation_scenario(
            plan_path,
            injection,
            "injection",
            "envelope_plus",
            1024,
            ("H1", "L1", "V1"),
        )


def test_calibration_response_rejects_nonphysical_envelope() -> None:
    with pytest.raises(ValueError, match="envelope is invalid"):
        apply_frequency_dependent_calibration_response(
            np.ones(64),
            1024,
            {
                "application": "multiplicative_rfft_strain_response_v1",
                "anchor_frequencies_hz": [20, 100, 480],
                "amplitude_fraction": [-1.0, 0.0, 0.0],
                "phase_radians": [0.0, 0.0, 0.0],
            },
        )


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _make_robustness_inputs(tmp_path: Path) -> tuple[Path, Path, list[Path], Path]:
    config = tmp_path / "config.yaml"
    _write_config(config)
    with config.open("a", encoding="utf-8") as handle:
        handle.write(
            """
calibration_robustness:
  minimum_scenarios: 7
  maximum_absolute_weighted_efficiency_loss: 0.10
  maximum_far_multiplier_of_target: 2.0
  bootstrap_replicates: 200
  minimum_injection_gps_blocks: 2
  seed: 1234
"""
        )
    background_source, injection_source = _write_manifests(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = freeze_calibration_perturbation_plan(
        background_source, injection_source, config, plan_path
    )
    identity = {
        "candidate_checkpoint_sha256": "checkpoint",
        "candidate_config_sha256": "model-config",
        "candidate_code_commit": "commit",
        "timing_calibration_report_sha256": "timing",
        "physical_delay_limit_seconds": 0.01,
        "empirical_timing_uncertainty_seconds": 0.002,
        "reference_ifo": "H1",
        "second_ifo": "L1",
    }
    baseline_rows_path = tmp_path / "baseline-rankings.jsonl"
    baseline_rows_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {
                    "injection_id": "i1",
                    "waveform_id": "w1",
                    "source_family": "bbh",
                    "stratum": "near",
                    "gps_block": "O4a:inj-1",
                    "gps_time": 1.0,
                    "vt_weight": 1.0,
                    "vt_weight_unit": "arb",
                    "ranking_score": 0.6,
                },
                {
                    "injection_id": "i2",
                    "waveform_id": "w2",
                    "source_family": "bbh",
                    "stratum": "far",
                    "gps_block": "O4a:inj-2",
                    "gps_time": 2.0,
                    "vt_weight": 1.0,
                    "vt_weight_unit": "arb",
                    "ranking_score": 0.4,
                },
            )
        ),
        encoding="utf-8",
    )
    schedule = {
        "background_manifest_sha256": plan["manifests"]["background"]["sha256"],
        "background_pairing_method": "gps_block_rank_permutation_v1",
        "equivalent_live_time_years": 2.0,
        "input_gps_blocks": ["O4a:block-background"],
        "slide_schedule_id": "schedule",
        "slide_schedule_sha256": "schedule-sha",
        "slide_count": 1,
    }
    baseline_ranking_report = _write_json(
        tmp_path / "baseline-ranking-report.json",
        {
            "status": "physical_network_injection_candidate_rankings",
            "split": "val",
            "manifest_path": str(baseline_rows_path),
            "manifest_sha256": file_sha256(baseline_rows_path),
        },
    )
    baseline_background_manifest = tmp_path / "baseline-background.jsonl"
    baseline_background_manifest.write_text(
        json.dumps({"ranking_score": 0.6}) + "\n", encoding="utf-8"
    )
    baseline_background_report = _write_json(
        tmp_path / "baseline-background-report.json",
        {
            "status": "subwindow_clustered_time_slide_integration_only",
            "split": "val",
            "manifest_path": str(baseline_background_manifest),
            "manifest_sha256": file_sha256(baseline_background_manifest),
            **schedule,
        },
    )
    baseline = _write_json(
        tmp_path / "baseline-calibration.json",
        {
            "status": "frozen_validation_candidate_search_calibration",
            "scientific_claim_allowed": False,
            "test_evaluation": None,
            "publication_calibration_eligible": True,
            "slide_schedule_audit": {"passed": True},
            "identity": identity,
            "calibration": {"threshold": 0.5},
            "target_far_per_year": 1.0,
            "validation_injection_diagnostic": {"weighted_efficiency": 0.5},
            "validation_injection_ranking_report_path": str(baseline_ranking_report),
            "validation_injection_ranking_report_sha256": file_sha256(baseline_ranking_report),
            "validation_time_slide_report_path": str(baseline_background_report),
            "validation_time_slide_report_sha256": file_sha256(baseline_background_report),
        },
    )
    receipts = []
    for scenario_id in plan["scenario_ids"]:
        directory = tmp_path / scenario_id
        directory.mkdir()
        background_rows = directory / "background.jsonl"
        background_rows.write_text(json.dumps({"ranking_score": 0.6}) + "\n", encoding="utf-8")
        background_report = _write_json(
            directory / "background-report.json",
            {
                "manifest_path": str(background_rows),
                "manifest_sha256": file_sha256(background_rows),
                **schedule,
            },
        )
        ranking_rows = directory / "rankings.jsonl"
        scenario_rows = [
            {**row, "ranking_score": score}
            for row, score in zip(
                [json.loads(line) for line in baseline_rows_path.read_text().splitlines()],
                (0.7, 0.8),
            )
        ]
        ranking_rows.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in scenario_rows),
            encoding="utf-8",
        )
        ranking_report = _write_json(
            directory / "ranking-report.json",
            {
                "manifest_path": str(ranking_rows),
                "manifest_sha256": file_sha256(ranking_rows),
            },
        )
        trigger_rows = directory / "injection-triggers.jsonl"
        trigger_rows.write_text(
            "".join(
                json.dumps({"injection_id": value, "valid_ifos": ["H1", "L1"]}) + "\n"
                for value in ("i1", "i2")
            ),
            encoding="utf-8",
        )
        score_report = _write_json(
            directory / "injection-score-report.json",
            {
                "triggers_path": str(trigger_rows),
                "triggers_sha256": file_sha256(trigger_rows),
            },
        )
        artifacts = {
            "background_search": {
                "path": str(background_report),
                "sha256": file_sha256(background_report),
            },
            "injection_ranking": {
                "path": str(ranking_report),
                "sha256": file_sha256(ranking_report),
            },
            "injection_score": {
                "path": str(score_report),
                "sha256": file_sha256(score_report),
            },
        }
        receipt = _write_json(
            directory / "receipt.json",
            {
                "status": "frozen_validation_calibration_perturbation_scenario_result",
                "passed": True,
                "test_rows_read": 0,
                "threshold_fitted_or_selected": False,
                "scenario_id": scenario_id,
                "plan": {"sha256": file_sha256(plan_path)},
                "model_identity": identity,
                "artifacts": artifacts,
            },
        )
        receipts.append(receipt)
    return plan_path, baseline, receipts, config


def test_calibration_robustness_uses_one_fixed_threshold_and_hand_calculated_far(
    tmp_path: Path,
) -> None:
    plan, baseline, receipts, config = _make_robustness_inputs(tmp_path)
    output = tmp_path / "robustness.json"

    result = evaluate_calibration_perturbation_robustness(plan, baseline, receipts, config, output)

    assert result["passed"] is True
    assert result["scenario_count"] == 7
    assert result["scenario_threshold_refits"] == 0
    assert result["detector_strata"]["H1+L1"]["scenario_count"] == 7
    assert result["detector_strata_audited"]["H1+L1"]["scenario_count"] == 7
    assert result["injection_bootstrap_independence"]["physical_groups"] == 2
    for scenario in result["scenario_results"]:
        assert scenario["threshold"] == pytest.approx(0.5)
        assert scenario["far_per_year"] == pytest.approx(0.5)
        assert scenario["baseline_weighted_efficiency"] == pytest.approx(0.5)
        assert scenario["scenario_weighted_efficiency"] == pytest.approx(1.0)


def test_calibration_scenario_receipt_closes_variable_detector_candidate_chain(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    background_source, injection_source = _write_manifests(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = freeze_calibration_perturbation_plan(
        background_source, injection_source, config, plan_path
    )
    scenario_id = plan["scenario_ids"][0]
    plan_sha = file_sha256(plan_path)
    score_reports = {}
    trigger_paths = {}
    for role, status in (
        ("background", "real_o4a_domain_transfer_diagnostic"),
        ("injection", "physical_waveform_real_noise_domain_transfer_diagnostic"),
    ):
        triggers = tmp_path / f"{role}-triggers.jsonl"
        triggers.write_text(json.dumps({"id": role}) + "\n", encoding="utf-8")
        trigger_paths[role] = triggers
        score_reports[role] = _write_json(
            tmp_path / f"{role}-score.json",
            {
                "status": status,
                "calibration_perturbation": {
                    "plan_sha256": plan_sha,
                    "role": role,
                    "manifest_sha256": plan["manifests"][role]["sha256"],
                    "scenario_id": scenario_id,
                },
                "required_split": "val",
                "observed_splits": ["val"],
                "physical_time_domain_perturbation": True,
                "fresh_time_frequency_transform": True,
                ("failed_windows" if role == "background" else "failed_injections"): 0,
                "triggers_path": str(triggers),
                "triggers_sha256": file_sha256(triggers),
                "checkpoint_sha256": "checkpoint",
                "config_sha256": "model-config",
                "code_commit": "commit",
            },
        )
    timing_reports = {}
    timing_sha = "timing-calibration"
    for role in ("background", "injection"):
        output = tmp_path / f"{role}-calibrated.jsonl"
        output.write_text(json.dumps({"candidate_id": role}) + "\n", encoding="utf-8")
        timing_reports[role] = _write_json(
            tmp_path / f"{role}-timing.json",
            {
                "status": "candidate_timing_calibration_applied",
                "uncalibrated_candidates": 0,
                "scoring_provenance_matches": True,
                "calibration_report_sha256": timing_sha,
                "candidate_extraction_provenance": {
                    "scoring": {
                        "score_report_sha256": file_sha256(score_reports[role]),
                        "trigger_manifest_sha256": file_sha256(trigger_paths[role]),
                    }
                },
                "output_path": str(output),
                "output_sha256": file_sha256(output),
            },
        )
    search_manifest = tmp_path / "search.jsonl"
    search_manifest.write_text(json.dumps({"ranking_score": 0.2}) + "\n")
    common_identity = {
        "candidate_checkpoint_sha256": "checkpoint",
        "candidate_config_sha256": "model-config",
        "candidate_code_commit": "commit",
        "timing_calibration_report_sha256": timing_sha,
        "empirical_timing_uncertainty_seconds": 0.002,
    }
    detector_subsets = ["H1+L1", "H1+V1", "L1+V1", "H1+L1+V1"]
    physical_limits = {"H1-L1": 0.01, "H1-V1": 0.027, "L1-V1": 0.026}
    allowed_separations = {
        key: value + 0.004 for key, value in physical_limits.items()
    }
    background_search = _write_json(
        tmp_path / "search-report.json",
        {
            "status": "variable_detector_set_block_permutation_background",
            "split": "val",
            "publication_timing_gate_passed": True,
            "candidate_timing_empirically_calibrated": True,
            "candidate_manifest_sha256": file_sha256(tmp_path / "background-calibrated.jsonl"),
            "background_manifest_sha256": plan["manifests"]["background"]["sha256"],
            "manifest_path": str(search_manifest),
            "manifest_sha256": file_sha256(search_manifest),
            "equivalent_live_time_years": 1.0,
            "required_detector_subsets": detector_subsets,
            "pairwise_light_travel_time_seconds": physical_limits,
            "pairwise_allowed_peak_separation_seconds": allowed_separations,
            **common_identity,
        },
    )
    ranking_manifest = tmp_path / "ranking.jsonl"
    ranking_manifest.write_text(json.dumps({"injection_id": "i"}) + "\n")
    injection_ranking = _write_json(
        tmp_path / "ranking-report.json",
        {
            "status": (
                "physical_variable_detector_set_injection_candidate_rankings"
            ),
            "split": "val",
            "timing_calibration_consistent": True,
            "candidate_scoring_provenance_consistent": True,
            "candidate_manifest_sha256": file_sha256(tmp_path / "injection-calibrated.jsonl"),
            "injection_trigger_manifest_sha256": file_sha256(trigger_paths["injection"]),
            "manifest_path": str(ranking_manifest),
            "manifest_sha256": file_sha256(ranking_manifest),
            "required_detector_subsets": detector_subsets,
            "pairwise_light_travel_time_seconds": physical_limits,
            "pairwise_allowed_peak_separation_seconds": allowed_separations,
            **common_identity,
        },
    )

    receipt = freeze_calibration_perturbation_scenario_result(
        plan_path,
        score_reports["background"],
        score_reports["injection"],
        timing_reports["background"],
        timing_reports["injection"],
        background_search,
        injection_ranking,
        tmp_path / "receipt.json",
    )

    assert receipt["passed"] is True
    assert receipt["scenario_id"] == scenario_id
    assert receipt["threshold_fitted_or_selected"] is False
    assert (
        receipt["model_identity"]["detector_set_policy"]
        == "single_model_explicit_missing_ifo_validity_v1"
    )
    assert receipt["model_identity"]["network_coherence_policy"][
        "required_detector_subsets"
    ] == detector_subsets


def test_calibration_robustness_rejects_missing_required_detector_subset(
    tmp_path: Path,
) -> None:
    plan, baseline, receipts, config = _make_robustness_inputs(tmp_path)
    content = config.read_text(encoding="utf-8")
    config.write_text(
        content.replace(
            "  seed: 1234\n",
            "  required_detector_subsets: [H1+L1, H1+V1]\n"
            "  seed: 1234\n",
        ),
        encoding="utf-8",
    )

    result = evaluate_calibration_perturbation_robustness(
        plan,
        baseline,
        receipts,
        config,
        tmp_path / "missing-detector-subset.json",
    )

    assert result["passed"] is False
    assert result["required_detector_subsets_covered"] is False
    assert result["required_detector_subsets"] == ["H1+L1", "H1+V1"]


def test_calibration_robustness_rejects_missing_or_refitted_scenario(
    tmp_path: Path,
) -> None:
    plan, baseline, receipts, config = _make_robustness_inputs(tmp_path)
    with pytest.raises(ValueError, match="every frozen scenario"):
        evaluate_calibration_perturbation_robustness(
            plan, baseline, receipts[:-1], config, tmp_path / "missing.json"
        )
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    receipt["threshold_fitted_or_selected"] = True
    receipts[0].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the baseline"):
        evaluate_calibration_perturbation_robustness(
            plan, baseline, receipts, config, tmp_path / "refit.json"
        )
