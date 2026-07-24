from __future__ import annotations

import re
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_glitch_head_only_overlap_fallback.sh"
)


def test_glitch_head_fallback_requires_teacher_failure_and_is_test_blind() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "FAILED_TEACHER_CHAIN_ROOT",
        "teacher_anchor_fallback_authorization.json",
        "authorized_validation_only_teacher_anchored_overlap_fallback",
        "completed_source_safe_overlap_negative_promotion",
        "completed_source_safe_overlap_negative_five_seed",
        "20-epoch clean-retention failure",
        "physical_overlap_finetune_glitch_head_only.yaml",
        "physical_overlap_finetune_family_balanced_glitch_head_only.yaml",
        "checkpoint_selection_metric",
        "validation_loss",
        "authorized_validation_only_glitch_head_only_overlap_fallback",
        "backbone_and_chirp_head_bit_exact",
        "physical_overlap_report.json",
        "ln -s",
        "test_rows_read",
    ):
        assert token in source
    assert "--split test" not in source
    assert "evaluation-corpus-open-once" not in source


def test_glitch_head_fallback_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 1
    compile(snippets[0], SCRIPT.name, "exec")
