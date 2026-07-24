from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/run_glitch_adapter_overlap_fallback.sh"
CONFIG = ROOT / "configs/physical_overlap_finetune_glitch_adapter.yaml"


def test_glitch_adapter_fallback_is_negative_head_gated() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "completed_source_safe_overlap_negative_promotion",
        "completed_source_safe_overlap_negative_five_seed",
        "checkpoint_selection_metric",
        "validation_loss",
        "glitch_head_only",
        "non_glitch_state_preserved_bit_exact",
        "authorized_validation_only_glitch_adapter_overlap_fallback",
        "zero_initialized_residual_glitch_decoder_v1",
        "test_rows_read",
        "GWYOLO_ASSIGNED_GPU_INDEX",
        "--query-compute-apps=pid",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert token in source


def test_glitch_adapter_fallback_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 1
    compile(snippets[0], f"{SCRIPT.name}:heredoc", "exec")


def test_glitch_adapter_config_freezes_predeclared_capacity() -> None:
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
        "overlap_training"
    ]
    assert settings["training_scope"] == "glitch_adapter_only"
    assert settings["glitch_adapter_channels"] == 16
    assert settings["checkpoint_selection_metric"] == "validation_loss"
    assert settings["epochs"] == 20
    assert settings["learning_rate"] == 0.0003
    assert settings["weight_decay"] == 0.0001
    assert settings["minimum_clean_chirp_iou_retention"] == 0.95
    assert settings["threshold_grid"] == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_glitch_adapter_fallback_rejects_missing_contract(tmp_path: Path) -> None:
    environment = {
        **os.environ,
        "TASK_PYTHON": str(tmp_path / "python"),
        "TASK_CODE_DIR": str(tmp_path / "code"),
        "GWYOLO_CODE_COMMIT": "fallback",
        "FAILED_HEAD_CODE_COMMIT": "head",
        "FAILED_HEAD_CHAIN_ROOT": str(tmp_path / "failed"),
        "CLEAN_TRAIN_MANIFEST": str(tmp_path / "train"),
        "CLEAN_VALIDATION_MANIFEST": str(tmp_path / "validation"),
        "PRETRAINED_CHECKPOINT": str(tmp_path / "checkpoint"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "SEED": "0",
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "seed must be a positive integer" in completed.stderr
