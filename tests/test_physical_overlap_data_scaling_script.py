from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts/run_physical_overlap_data_scaling.sh"


def test_overlap_data_scaling_script_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_overlap_data_scaling_script_runs_both_controls_and_never_test() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "physical-overlap-scale-subsets" in source
    assert "physical-overlap-scale-summarize" in source
    assert "fixed_epochs fixed_optimizer_updates" in source
    assert "exactly five seeds" in source
    assert "--include-full" in source
    assert "--required-split test" not in source
    assert "test_rows_read" in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source


def test_teacher_anchor_scaling_configs_preserve_both_compute_controls() -> None:
    configs = Path(__file__).parents[1] / "configs"
    fixed_epochs = yaml.safe_load(
        (configs / "physical_overlap_scale_fixed_epochs_teacher_anchor.yaml").read_text(
            encoding="utf-8"
        )
    )["overlap_training"]
    fixed_updates = yaml.safe_load(
        (
            configs / "physical_overlap_scale_fixed_updates_teacher_anchor.yaml"
        ).read_text(encoding="utf-8")
    )["overlap_training"]

    for settings in (fixed_epochs, fixed_updates):
        assert settings["learning_rate"] == 1e-5
        assert settings["clean_chirp_weight"] == 0.25
        assert settings["clean_chirp_distillation_weight"] == 4.0
        assert settings["minimum_clean_chirp_iou_retention"] == 0.95
        assert settings["glitch_family_sampling"]["enabled"] is True
    assert fixed_epochs["training_control"] == "fixed_epochs"
    assert fixed_epochs["epochs"] == 20
    assert "max_optimizer_updates" not in fixed_epochs
    assert fixed_updates["training_control"] == "fixed_optimizer_updates"
    assert fixed_updates["max_optimizer_updates"] == 4000


def test_glitch_head_scaling_configs_preserve_scope_and_compute_controls() -> None:
    configs = Path(__file__).parents[1] / "configs"
    fixed_epochs = yaml.safe_load(
        (
            configs
            / "physical_overlap_scale_fixed_epochs_glitch_head_only.yaml"
        ).read_text(encoding="utf-8")
    )["overlap_training"]
    fixed_updates = yaml.safe_load(
        (
            configs
            / "physical_overlap_scale_fixed_updates_glitch_head_only.yaml"
        ).read_text(encoding="utf-8")
    )["overlap_training"]

    for settings in (fixed_epochs, fixed_updates):
        assert settings["training_scope"] == "glitch_head_only"
        assert settings["learning_rate"] == 1e-4
        assert settings["weight_decay"] == 0.0
        assert settings["clean_chirp_weight"] == 0.25
        assert settings["clean_chirp_distillation_weight"] == 0.0
        assert settings["minimum_clean_chirp_iou_retention"] == 0.95
        assert settings["glitch_family_sampling"]["enabled"] is True
    assert fixed_epochs["training_control"] == "fixed_epochs"
    assert fixed_epochs["epochs"] == 20
    assert "max_optimizer_updates" not in fixed_epochs
    assert fixed_updates["training_control"] == "fixed_optimizer_updates"
    assert fixed_updates["max_optimizer_updates"] == 4000
