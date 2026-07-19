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
