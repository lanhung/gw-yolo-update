from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_mask_conditioned_background_range.sh"


def test_mask_conditioned_background_runner_is_gate_bound_and_validation_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "MASK_VALIDATION_RECEIPT",
        "MASK_TIMING_RECEIPT",
        "candidate-scoring-compatibility-audit",
        "background-raw-mask-stream-shard",
        "background-raw-mask-stream-merge",
        "candidate-block-permutation-schedule-freeze",
        "candidate-block-permutations",
        "candidate-search-calibrate",
        '"$OUTPUT_ROOT/raw/frozen_validation_candidate_search_calibration.json"',
        '"$OUTPUT_ROOT/mask/frozen_validation_candidate_search_calibration.json"',
        '"test_rows_read": 0',
        '"locked_test_allowed": False',
        '"locked_test_open_allowed": False',
        '"locked_test_prerequisites_satisfied": False',
        "raw/mask background validation receipts are immutable",
        "must cover the complete parent plan",
    ):
        assert token in source
    assert '"scale_locked_test_allowed": True' not in source
    assert "--test-fraction" not in source
    assert "evaluation-corpus-open-once" not in source
