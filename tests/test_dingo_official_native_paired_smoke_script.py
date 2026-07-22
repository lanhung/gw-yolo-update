from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "run_dingo_official_native_paired_smoke.sh"
)


def test_dingo_official_native_smoke_fails_closed_when_inputs_are_unset() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 2
    assert "TASK_PYTHON" in completed.stderr


def test_dingo_official_native_smoke_is_validation_only_and_not_joint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "dingo-official-native-model-freeze" in source
    assert "DINGO_NATIVE_RUNTIME_RECEIPT" in source
    assert "DINGO_NATIVE_EVENT_SMOKE_SUMMARY" in source
    assert "--native-runtime-receipt" in source
    assert "--native-event-smoke-summary" in source
    assert "--comparison-mode official_native" in source
    assert source.count("--required-split val") == 1
    assert "pe-robustness-evaluate" in source
    assert "--within-backend-only" in source
    assert "within_backend_provenance_gate" in source
    assert "pe-robustness-joint-evaluate" not in source
    assert "amplfi-common-batch" not in source
    assert "AMPLFI_MODEL_METADATA" not in source
    assert "--required-split test" not in source
    assert "cross_backend_absolute_comparison_allowed" in source
    assert "evaluation_tier" in source
    assert "minimum_publication_validation_injections" in source
    assert "bootstrap_replicates" in source
    assert "three-event smoke" not in source
    assert 'git -C "$TASK_CODE_DIR" rev-parse HEAD' in source
