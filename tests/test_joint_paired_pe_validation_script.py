from __future__ import annotations

import os
import subprocess
import sys
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
    assert "readarray -t native_manifests < <(" not in source


def test_joint_paired_pe_validation_propagates_manifest_resolution_failure(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    pe_input = tmp_path / "pe-input"
    (pe_input / "dingo-native").mkdir(parents=True)
    (pe_input / "amplfi-native").mkdir(parents=True)
    for path in (
        pe_input / "dingo-native" / "native_conditioning_report.json",
        pe_input / "amplfi-native" / "native_conditioning_report.json",
        pe_input / "paired_pe_smoke_summary.json",
    ):
        path.write_text("{}\n", encoding="utf-8")
    inputs = {}
    for name in (
        "dingo_model_metadata",
        "dingo_native_prior",
        "dingo_model_init",
        "amplfi_model_metadata",
        "amplfi_native_prior",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        inputs[name.upper()] = str(path)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "TASK_CODE_DIR": str(repository),
            "GWYOLO_CODE_COMMIT": commit,
            "PE_INPUT_ROOT": str(pe_input),
            "DINGO_PYTHON": sys.executable,
            "AMPLFI_PYTHON": sys.executable,
            "OUTPUT_ROOT": str(tmp_path / "output"),
            **inputs,
        }
    )
    completed = subprocess.run(
        ["bash", str(repository / "scripts/run_joint_paired_pe_validation.sh")],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 4
    assert "joint PE native manifest resolution failed" in completed.stderr
    assert "unbound variable" not in completed.stderr
