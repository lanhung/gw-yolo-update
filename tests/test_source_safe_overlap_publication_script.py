from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_source_safe_overlap_publication.sh"
)


def test_source_safe_overlap_publication_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_source_safe_overlap_publication_binds_complete_validation_chain() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert "verified_group_safe_gravityspy_aligned_network_corpus" in source
    assert "cross_split_overlaps" in source
    assert "run_recovered_overlap_ablation.sh" in source
    assert "physical-overlap-sampling-promote" in source
    assert "run_overlap_five_seed_promotion.sh" in source
    assert "completed_source_safe_overlap_negative_promotion" in source
    assert '"test_rows_read": 0' in source
    assert "--required-split test" not in source
