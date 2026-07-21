from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_paired_pe_smoke_fails_before_work_when_paths_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/run_paired_pe_smoke.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "GWYOLO_PYTHON" in completed.stderr


def test_paired_pe_smoke_binds_the_selected_model_configuration() -> None:
    script = Path(__file__).parents[1] / "scripts/run_paired_pe_smoke.sh"
    source = script.read_text(encoding="utf-8")
    assert "GWYOLO_MODEL_CONFIG" in source
    assert '"config_file_sha256"' in source
    assert '--config "$GWYOLO_MODEL_CONFIG"' in source
    assert "--config configs/physical_overlap_finetune.yaml" not in source
