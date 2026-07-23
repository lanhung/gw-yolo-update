from __future__ import annotations

import re
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_teacher_anchored_overlap_fallback.sh"
)


def test_teacher_anchor_fallback_requires_a_completed_negative_gate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    for token in (
        "FAILED_CHAIN_ROOT",
        "completed_source_safe_overlap_negative_promotion",
        "completed_source_safe_overlap_negative_five_seed",
        "20-epoch clean-retention failure",
        "physical_overlap_finetune_teacher_anchor.yaml",
        "physical_overlap_finetune_family_balanced_teacher_anchor.yaml",
        "authorized_validation_only_teacher_anchored_overlap_fallback",
        "clean_chirp_distillation_weight",
        "physical_overlap_report.json",
        "ln -s",
        "test_rows_read",
    ):
        assert token in source
    assert "--split test" not in source
    assert "evaluation-corpus-open-once" not in source


def test_teacher_anchor_fallback_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 1
    compile(snippets[0], SCRIPT.name, "exec")
