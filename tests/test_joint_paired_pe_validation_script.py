from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_joint_paired_pe_validation_fails_closed_when_inputs_are_unset() -> None:
    script = Path(__file__).parents[1] / "scripts/run_joint_paired_pe_validation.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_joint_paired_pe_validation_uses_strict_joint_cli_and_validation_split() -> None:
    script = Path(__file__).parents[1] / "scripts/run_joint_paired_pe_validation.sh"
    source = script.read_text(encoding="utf-8")
    assert "pe-robustness-joint-evaluate" in source
    assert "pe-robustness-promote" in source
    assert "configs/pe_robustness_promotion.yaml" in source
    assert source.count("--required-split val") == 2
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
    assert "paired_pe_native_inputs_smoke_complete" in source
    assert "cross_backend_matched_input_gate" in source
    assert "DINGO_NATIVE_PRIOR" in source
    assert source.count("--native-prior") == 2
    assert "--required-split test" not in source
