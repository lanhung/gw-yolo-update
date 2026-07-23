from __future__ import annotations

import re
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_numeric_raw_mask_detector_set_successor.sh"
)


def test_numeric_raw_mask_detector_set_successor_is_gate_bound() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "DETECTOR_BASELINE_RECEIPT",
        "MASK_VALIDATION_RECEIPT",
        "MASK_TIMING_RECEIPT",
        "--save-probabilities",
        "learned-background-deglitch",
        "learned-deglitch",
        "detector-set-block-permutations",
        "detector-set-injection-candidate-rank",
        "numeric-raw-mask-detector-set-ranking-successor-freeze",
        "candidate-search-raw-mask-compare",
        "completed_validation_only_numeric_detector_set_raw_mask_successor",
        "final_search_far_claim_allowed",
    ):
        assert token in source
    assert "--split test" not in source
    assert "evaluation-corpus-open-once" not in source


def test_numeric_raw_mask_detector_set_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) >= 3
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")
