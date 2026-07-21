from __future__ import annotations

import os
import subprocess
import sys
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
    assert "promoted paired PE model selection failed" in source
    assert "readarray -t selection < <(" not in source


def test_promoted_paired_pe_smoke_propagates_selector_failure(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/run_promoted_paired_pe_smoke.sh"
    code = tmp_path / "code"
    (code / "src" / "gwyolo").mkdir(parents=True)
    inputs = {}
    for name in (
        "five_seed_summary",
        "uniform_config",
        "family_balanced_config",
        "overlap_manifest",
        "injection_manifest",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        inputs[name.upper()] = str(path)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "TASK_CODE_DIR": str(code),
            "GWYOLO_CODE_COMMIT": "commit",
            "OUTPUT_ROOT": str(tmp_path / "output"),
            **inputs,
        }
    )
    completed = subprocess.run(
        ["bash", str(script)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "promoted paired PE model selection failed" in completed.stderr
    assert "unbound variable" not in completed.stderr
