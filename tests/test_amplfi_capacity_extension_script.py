from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_amplfi_background_capacity_extension.sh"
)


def test_amplfi_capacity_extension_preserves_frozen_policy_and_evicts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    ordered = (
        "gwosc-plan-disjoint",
        "gwosc-batch-download",
        "background-batch-plan",
        "amplfi-background-export",
        "amplfi-background-source-evict",
        "amplfi-background-extension-merge",
        "amplfi-background-capacity-audit",
    )
    positions = [source.index(token) for token in ordered]
    assert positions == sorted(positions)
    for token in (
        "--split-strategy hash_threshold_v1",
        "--test-fraction 0",
        "BACKGROUND_SEED:-20260727",
        "BASE_STREAM_MERGE_REPORT",
        "CAPACITY_POLICY",
        "MINIMUM_FREE_BYTES",
        "from scipy.signal import resample_poly",
        "requires h5py, NumPy and scipy.signal",
        '"candidate_scores_inspected"',
        '"test_rows_read"',
    ):
        assert token in source
    assert "rm -rf" not in source
    assert "evaluation-corpus-open-once" not in source
    assert "O4b" not in source
