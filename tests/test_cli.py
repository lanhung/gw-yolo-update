from __future__ import annotations

from unittest.mock import patch

from gwyolo.cli import main


def test_trigger_cli_routes_only_declared_arguments() -> None:
    with patch("gwyolo.trigger.score_background_manifest", return_value={}) as score:
        assert (
            main(
                [
                    "trigger-score",
                    "--manifest",
                    "manifest.jsonl",
                    "--checkpoint",
                    "model.pt",
                    "--config",
                    "config.yaml",
                    "--output-dir",
                    "output",
                ]
            )
            == 0
        )
    score.assert_called_once_with(
        manifest_path="manifest.jsonl",
        checkpoint_path="model.pt",
        config_path="config.yaml",
        output_dir="output",
        model_ifos=("H1", "L1", "V1"),
        q_values=(4, 8, 16),
        target_sample_rate=1024,
        context_duration=64.0,
        save_probabilities=False,
        required_split=None,
        enabled_ifos=None,
        coherence_config_path=None,
    )


def test_physical_cli_forwards_storage_and_probability_flags() -> None:
    with patch("gwyolo.waveforms.run_injection_materialization", return_value={}) as materialize:
        assert (
            main(
                [
                    "injection-materialize",
                    "--recipes",
                    "recipes.jsonl",
                    "--background-manifest",
                    "background.jsonl",
                    "--output-dir",
                    "materialized",
                    "--storage-mode",
                    "full",
                ]
            )
            == 0
        )
    assert materialize.call_args.kwargs["storage_mode"] == "full"

    with patch("gwyolo.injection_score.score_materialized_injections", return_value={}) as score:
        assert (
            main(
                [
                    "injection-score",
                    "--manifest",
                    "materialized.jsonl",
                    "--checkpoint",
                    "model.pt",
                    "--config",
                    "config.yaml",
                    "--output-dir",
                    "scores",
                    "--save-probabilities",
                ]
            )
            == 0
        )
    assert score.call_args.kwargs["save_probabilities"] is True
    assert score.call_args.kwargs["required_split"] is None
    assert score.call_args.kwargs["enabled_ifos"] is None
    assert score.call_args.kwargs["coherence_config_path"] is None


def test_detector_arrival_validation_stratification_cli_routes_inputs() -> None:
    target = "gwyolo.arrival_timing.run_detector_arrival_timing_validation_stratification"
    with patch(target, return_value={}) as stratify:
        assert (
            main(
                [
                    "detector-arrival-timing-validation-stratify",
                    "--config",
                    "timing.yaml",
                    "--validation-manifest",
                    "validation.jsonl",
                    "--checkpoint",
                    "timing.pt",
                    "--output",
                    "strata.json",
                    "--predictions-output",
                    "predictions.jsonl",
                ]
            )
            == 0
        )
    stratify.assert_called_once_with(
        "timing.yaml",
        "validation.jsonl",
        "timing.pt",
        "strata.json",
        "predictions.jsonl",
    )

    target = "gwyolo.arrival_timing.run_detector_arrival_timing_validation_comparison"
    with patch(target, return_value={}) as compare:
        assert (
            main(
                [
                    "detector-arrival-timing-validation-compare",
                    "--config",
                    "promotion.yaml",
                    "--reference-predictions",
                    "v1.jsonl",
                    "--candidate-predictions",
                    "v2.jsonl",
                    "--output",
                    "comparison.json",
                ]
            )
            == 0
        )
    compare.assert_called_once_with(
        "promotion.yaml", "v1.jsonl", "v2.jsonl", "comparison.json"
    )


def test_candidate_proposal_audit_cli_routes_all_instance_inputs() -> None:
    with patch(
        "gwyolo.candidates.run_candidate_proposal_coverage_audit", return_value={}
    ) as audit:
        assert (
            main(
                [
                    "candidate-proposal-audit",
                    "--injection-manifest",
                    "injections.jsonl",
                    "--candidate-manifest",
                    "candidates.jsonl",
                    "--output",
                    "coverage.json",
                    "--padding-seconds",
                    "0.5",
                ]
            )
            == 0
        )
    audit.assert_called_once_with(
        "injections.jsonl", "candidates.jsonl", "coverage.json", 0.5
    )

    with patch(
        "gwyolo.candidates.run_candidate_proposal_threshold_selection",
        return_value={},
    ) as select:
        assert (
            main(
                [
                    "candidate-proposal-sweep-select",
                    "--config",
                    "proposal.yaml",
                    "--audit-report",
                    "audit-03.json",
                    "--audit-report",
                    "audit-05.json",
                    "--output",
                    "selection.json",
                ]
            )
            == 0
        )
    select.assert_called_once_with(
        "proposal.yaml", ["audit-03.json", "audit-05.json"], "selection.json"
    )
