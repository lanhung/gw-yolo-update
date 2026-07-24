from __future__ import annotations

import re
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_detector_set_overlap_materialization.sh"
)


def test_detector_set_overlap_materialization_is_audited_and_not_scaling() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("physical-overlap-materialize") == 2
    assert "physical-overlap-audit" in source
    assert "EXPANSION_CAPACITY_REPORT" in source
    assert "EXPANDED_TRAIN_REPORT" in source
    assert "EXPANDED_VALIDATION_REPORT" in source
    assert "EXPANSION_READINESS_AUDIT" in source
    assert "detector_set_robustness_ablation_ready" in source
    assert "signal_overlap_materialization_authorized" in source
    assert "detector_complete_clean_training_authorized" in source
    assert "same_distribution_data_scaling_claim_allowed" in source
    assert '"test_rows_read": 0' in source
    assert "next_scale_training_authorized" in source
    assert "verified_detector_set_overlap_robustness_corpus" in source
    assert "MINIMUM_FREE_KB" in source
    assert "insufficient free space" in source


def test_detector_set_overlap_materialization_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")
