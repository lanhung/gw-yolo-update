from __future__ import annotations

from gwyolo import candidate_pipeline
from gwyolo.cli import main


def test_candidate_pipeline_cli_forwards_model_selection_report(
    monkeypatch,
) -> None:
    captured = {}

    def fake_pipeline(*args):
        captured["args"] = args
        return {"status": "ok"}

    monkeypatch.setattr(candidate_pipeline, "run_candidate_validation_pipeline", fake_pipeline)
    assert (
        main(
            [
                "candidate-search-validation-pipeline",
                "--background-manifest",
                "background.jsonl",
                "--injection-manifest",
                "injections.jsonl",
                "--checkpoint",
                "model.pt",
                "--config",
                "model.yaml",
                "--coherence-config",
                "coherence.yaml",
                "--output-dir",
                "output",
                "--model-selection-report",
                "five-seed.json",
            ]
        )
        == 0
    )
    assert captured["args"][-1] == "five-seed.json"
