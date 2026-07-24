from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/run_glitch_adapter_validation_scaling_successor.sh"
)


def test_glitch_adapter_successor_fails_closed_without_inputs() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_glitch_adapter_successor_orders_all_validation_gates() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    one_seed = source.index("physical-overlap-single-arm-promote")
    five_seed = source.index("physical-overlap-five-seed-summarize")
    scaling = source.index("run_physical_overlap_data_scaling.sh")
    hard_endpoint = source.index("run_physical_overlap_scaling_hard_endpoint.sh")
    assert one_seed < five_seed < scaling < hard_endpoint
    for token in (
        "completed_glitch_adapter_negative_one_seed",
        "completed_glitch_adapter_negative_five_seed",
        "not_authorized_by_one_seed_gate",
        "not_authorized_by_five_seed_gate",
        "scale_promotion_authorized",
        "physical_overlap_scale_fixed_epochs_glitch_adapter.yaml",
        "physical_overlap_scale_fixed_updates_glitch_adapter.yaml",
        "GWYOLO_ASSIGNED_GPU_INDEX",
        'nvidia-smi -i "$assigned_gpu"',
        "FAR/IFAR/<VT>",
        "test_rows_read",
    ):
        assert token in source
    assert "--required-split test" not in source


def test_glitch_adapter_successor_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) == 5
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:{index}", "exec")
