from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_promoted_paired_pe_smoke_fails_closed_when_inputs_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_promoted_paired_pe_smoke_binds_every_selected_artifact() -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    source = script.read_text(encoding="utf-8")
    for field in (
        "selected_checkpoint_sha256",
        "finetune_reports",
        "config_file_sha256",
        "overlap_validation_manifest_sha256",
        "clean_validation_manifest_sha256",
        "test_data_opened",
    ):
        assert field in source
    assert 'export GWYOLO_MODEL_CONFIG="${selection[1]}"' in source
