from __future__ import annotations

from pathlib import Path

from gwyolo.io import file_sha256
from gwyolo.mask_pipeline import run_mask_search_validation_pipeline


def test_mask_pipeline_runs_six_paired_score_arms(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    config = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("overlap_training: {tensor: {}}\n")
    calls = {"background_score": [], "injection_score": [], "clean": []}

    def artifact(name: str, status: str) -> dict:
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n")
        return {
            "status": status,
            "manifest_path": str(path),
            "manifest_sha256": file_sha256(path),
            "triggers_path": str(path),
        }

    def background_score(*args, **kwargs):
        calls["background_score"].append(kwargs.get("save_probabilities", False))
        return artifact(f"background-score-{len(calls['background_score'])}", "background")

    def injection_score(*args, **kwargs):
        calls["injection_score"].append(kwargs.get("save_probabilities", False))
        return artifact(f"injection-score-{len(calls['injection_score'])}", "injection")

    def background_clean(*args, **kwargs):
        calls["clean"].append("background")
        return artifact("background-clean", "background_clean")

    def injection_clean(*args, **kwargs):
        calls["clean"].append("injection")
        return artifact(f"injection-clean-{calls['clean'].count('injection')}", "clean")

    def compare(*args, **kwargs):
        output = Path(args[6])
        output.write_text('{"development_gates_passed": false}\n')
        return {
            "status": "comparison",
            "development_gates_passed": False,
        }

    monkeypatch.setattr("gwyolo.mask_pipeline.score_background_manifest", background_score)
    monkeypatch.setattr("gwyolo.mask_pipeline.score_materialized_injections", injection_score)
    monkeypatch.setattr(
        "gwyolo.mask_pipeline.run_learned_background_deglitch", background_clean
    )
    monkeypatch.setattr("gwyolo.mask_pipeline.run_learned_deglitch", injection_clean)
    monkeypatch.setattr("gwyolo.mask_pipeline.run_mask_search_validation", compare)
    result = run_mask_search_validation_pipeline(
        "background.jsonl",
        "clean.jsonl",
        "contaminated.jsonl",
        checkpoint,
        config,
        tmp_path / "pipeline",
        maximum_validation_false_alarms=8,
    )
    assert calls["background_score"] == [True, False]
    assert calls["injection_score"] == [True, False, True, False]
    assert calls["clean"] == ["background", "injection", "injection"]
    assert result["development_gates_passed"] is False
    assert result["test_evaluation"] is None
