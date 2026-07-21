from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_mask_deglitch_validation.sh"


def test_mask_deglitch_validation_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_mask_deglitch_validation_binds_independent_six_arm_protocol() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert "completed_five_seed_source_safe_overlap_validation" in source
    assert 'selection_commit = str(summary.get("code_commit", ""))' in source
    assert 'model_report.get("code_commit") != selection_commit' in source
    assert 'summary.get("code_commit") != commit' not in source
    assert '"model_selection_code_commit"' in source
    assert '"scale_mask_conditioned_morphology_background"' in source
    assert '"coherent_background_scale_allowed": False' in source
    assert "frozen_gps_and_purpose_disjoint_validation_endpoint" in source
    assert "verified_independent_validation_pe_overlap" in source
    assert "passed_physical_overlap_group_audit" in source
    assert "physical-overlap-contamination" in source
    assert "mask-search-validation-pipeline" in source
    assert "clean_noninferiority_margin" in source
    assert "minimum_contaminated_efficiency_gain" in source
    assert '"test_rows_read": 0' in source
    assert "--required-split test" not in source
