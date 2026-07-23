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
        calibration_plan_path=None,
        calibration_scenario_id=None,
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
    assert score.call_args.kwargs["calibration_plan_path"] is None
    assert score.call_args.kwargs["calibration_scenario_id"] is None


def test_locked_candidate_cli_forwards_suite_access_and_background() -> None:
    target = "gwyolo.cli.run_frozen_candidate_search_evaluation"
    with patch(target, return_value={}) as evaluate:
        assert (
            main(
                [
                    "candidate-search-evaluate-frozen",
                    "--calibration-report",
                    "calibration.json",
                    "--test-time-slide-report",
                    "slides.json",
                    "--test-background-manifest",
                    "background.jsonl",
                    "--test-injection-ranking-report",
                    "injections.json",
                    "--minimum-test-live-time-years",
                    "23.02585093",
                    "--minimum-test-injections",
                    "3000",
                    "--bootstrap-replicates",
                    "10000",
                    "--seed",
                    "20260722",
                    "--locked-suite-plan",
                    "suite.json",
                    "--access-log",
                    "access.json",
                    "--output-key",
                    "raw_candidate_search",
                    "--output",
                    "result.json",
                ]
            )
            == 0
        )
    evaluate.assert_called_once_with(
        "calibration.json",
        "slides.json",
        "injections.json",
        "result.json",
        23.02585093,
        3000,
        10000,
        20260722,
        "suite.json",
        "access.json",
        "raw_candidate_search",
        "background.jsonl",
    )


def test_locked_streaming_publication_and_merge_cli_route_exact_inputs() -> None:
    publish_target = (
        "gwyolo.locked_streaming.publish_locked_o4b_streaming_shard_artifacts"
    )
    with patch(publish_target, return_value={}) as publish:
        assert (
            main(
                [
                    "locked-o4b-streaming-shard-publish",
                    "--execution-plan",
                    "execution.json",
                    "--access-log",
                    "access.json",
                    "--shard-index",
                    "7",
                    "--raw-background-candidates",
                    "raw-bg.jsonl",
                    "--raw-injection-candidates",
                    "raw-inj.jsonl",
                    "--mask-background-candidates",
                    "mask-bg.jsonl",
                    "--mask-injection-candidates",
                    "mask-inj.jsonl",
                    "--ood-source-manifest",
                    "ood.jsonl",
                    "--injection-trigger-manifest",
                    "triggers.jsonl",
                    "--pe-input-manifest",
                    "pe.jsonl",
                    "--code-commit",
                    "abc123",
                ]
            )
            == 0
        )
    publish.assert_called_once_with(
        "execution.json",
        "access.json",
        7,
        "raw-bg.jsonl",
        "raw-inj.jsonl",
        "mask-bg.jsonl",
        "mask-inj.jsonl",
        "ood.jsonl",
        "triggers.jsonl",
        "pe.jsonl",
        "abc123",
    )

    merge_target = (
        "gwyolo.locked_streaming.merge_locked_o4b_streaming_suite_input_sources"
    )
    with patch(merge_target, return_value={}) as merge:
        assert (
            main(
                [
                    "locked-o4b-streaming-suite-inputs-merge",
                    "--suite-plan",
                    "suite.json",
                    "--execution-plan",
                    "execution.json",
                    "--access-log",
                    "access.json",
                    "--streaming-completion-audit",
                    "completion.json",
                    "--post-dq-weight-report",
                    "weights.json",
                    "--code-commit",
                    "abc123",
                ]
            )
            == 0
        )
    merge.assert_called_once_with(
        "suite.json",
        "execution.json",
        "access.json",
        "completion.json",
        "weights.json",
        "abc123",
    )


def test_automatic_mask_cli_routes_exact_inputs() -> None:
    audit_target = "gwyolo.automatic_mask.audit_automatic_mask_policy"
    with patch(audit_target, return_value={}) as audit:
        assert (
            main(
                [
                    "automatic-mask-policy-audit",
                    "--overlap-manifest",
                    "validation.jsonl",
                    "--overlap-config",
                    "overlap.yaml",
                    "--output",
                    "audit.json",
                ]
            )
            == 0
        )
    audit.assert_called_once_with(
        "validation.jsonl", "overlap.yaml", "audit.json"
    )

    bind_target = (
        "gwyolo.automatic_mask.bind_raw_mask_automatic_publication_evidence"
    )
    with patch(bind_target, return_value={}) as bind:
        assert (
            main(
                [
                    "candidate-search-raw-mask-automatic-endpoint-bind",
                    "--raw-mask-endpoint",
                    "raw-mask.json",
                    "--automatic-mask-audit",
                    "audit.json",
                    "--gate-config",
                    "gate.yaml",
                    "--output",
                    "endpoint.json",
                ]
            )
            == 0
        )
    bind.assert_called_once_with(
        "raw-mask.json", "audit.json", "gate.yaml", "endpoint.json"
    )


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


def test_candidate_refiner_train_cli_routes_endpoint_warm_start() -> None:
    target = "gwyolo.candidate_refiner.run_candidate_local_refiner_training"
    with patch(target, return_value={}) as train:
        assert (
            main(
                [
                    "candidate-refiner-train",
                    "--config",
                    "refiner.yaml",
                    "--train-injection-manifest",
                    "train-injections.jsonl",
                    "--train-candidate-manifest",
                    "train-candidates.jsonl",
                    "--validation-injection-manifest",
                    "val-injections.jsonl",
                    "--validation-selection-candidate-manifest",
                    "selection.jsonl",
                    "--validation-calibration-candidate-manifest",
                    "calibration.jsonl",
                    "--output-dir",
                    "output",
                    "--seed",
                    "7",
                    "--pretrained-endpoint-checkpoint",
                    "endpoint.pt",
                ]
            )
            == 0
        )
    train.assert_called_once_with(
        "refiner.yaml",
        "train-injections.jsonl",
        "train-candidates.jsonl",
        "val-injections.jsonl",
        "selection.jsonl",
        "calibration.jsonl",
        "output",
        7,
        "endpoint.pt",
    )


def test_candidate_pair_ranker_cli_routes_grouped_manifests() -> None:
    target = "gwyolo.candidate_set_training.run_candidate_pair_ranker_training"
    with patch(target, return_value={}) as train:
        assert (
            main(
                [
                    "candidate-pair-ranker-train",
                    "--config",
                    "pair.yaml",
                    "--train-injection-manifest",
                    "train-injections.jsonl",
                    "--train-candidate-manifest",
                    "train-candidates.jsonl",
                    "--validation-injection-manifest",
                    "val-injections.jsonl",
                    "--validation-selection-candidate-manifest",
                    "selection.jsonl",
                    "--output-dir",
                    "output",
                    "--seed",
                    "9",
                ]
            )
            == 0
        )
    train.assert_called_once_with(
        "pair.yaml",
        "train-injections.jsonl",
        "train-candidates.jsonl",
        "val-injections.jsonl",
        "selection.jsonl",
        "output",
        9,
    )
