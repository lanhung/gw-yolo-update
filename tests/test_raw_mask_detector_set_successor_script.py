from __future__ import annotations

import re
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_raw_mask_detector_set_successor.sh"
)


def test_raw_mask_detector_set_successor_is_validation_gate_bound() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "SOURCE_RAW_MASK_RECEIPT",
        "NETWORK_CONFIG",
        "completed_validation_only_raw_mask_continuous_background",
        "verified_merged_streamed_raw_mask_candidate_background",
        "authorized_validation_candidate_continuous_background_plan",
        "detector-set-block-permutation-schedule-freeze",
        "detector-set-block-permutations",
        "detector-set-injection-candidate-rank",
        "raw-mask-detector-set-ranking-successor-freeze",
        "candidate-search-calibrate",
        "candidate-search-raw-mask-compare",
        "--detector-set-ranking-successor",
        "candidate-search-raw-mask-endpoint-bind",
        "detector_set_candidate_background_dependence_audit_v1",
    ):
        assert token in source
    assert "evaluation-corpus-open-once" not in source
    assert "--split test" not in source


def test_raw_mask_detector_set_successor_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) >= 2
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")
