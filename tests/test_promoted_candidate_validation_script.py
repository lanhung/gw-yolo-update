from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_promoted_candidate_validation.sh"
)


def test_promoted_candidate_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_promoted_candidate_propagates_five_seed_selector_failure(
    tmp_path: Path,
) -> None:
    inputs = {}
    for name in (
        "five_seed_summary",
        "background_manifest",
        "injection_manifest",
        "uniform_config",
        "family_balanced_config",
        "coherence_config",
    ):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        inputs[name.upper()] = str(path)
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "WAVEFORM_PYTHON": sys.executable,
            "OUTPUT_ROOT": str(tmp_path / "output"),
            "GWYOLO_CODE_COMMIT": "commit",
            **inputs,
        }
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "failed to resolve checkpoint/config" in completed.stderr
    assert "unbound variable" not in completed.stderr
