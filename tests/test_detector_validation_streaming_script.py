from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_detector_validation_streaming_acquisition.sh"
)


def test_detector_validation_stream_is_hash_stable_test_free_and_evicts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    ordered = (
        "gwosc-batch-download",
        "background-batch-plan",
        "background-bank-materialize",
        "detector-validation-source-evict",
        "detector-validation-shard-seal",
    )
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    for token in (
        "--split-strategy hash_threshold_v1",
        "--test-fraction 0",
        "--maximum-windows-per-gps-block 1",
        "MINIMUM_FREE_BYTES",
        "detector-validation-background-merge",
        "detector-validation-injection-plan",
        '"candidate_scores_inspected"',
        '"test_rows_read"',
    ):
        assert token in source
    assert "rm -rf" not in source
    assert "evaluation-corpus-open-once" not in source
    assert "trigger-score" not in source
    assert "O4b" not in source
