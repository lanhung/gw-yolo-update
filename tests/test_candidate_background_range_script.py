from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_candidate_background_range.sh"


def _minimum_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "TASK_PYTHON",
        "TASK_CODE_DIR",
        "GWYOLO_CODE_COMMIT",
        "SCORING_CODE_DIR",
        "SCORING_CODE_COMMIT",
        "PROMOTION_REPORT",
        "PROMOTED_PIPELINE_REPORT",
        "PARENT_PLAN",
        "EVENT_EXCLUSIONS",
        "COHERENCE_CONFIG",
        "TIMING_CALIBRATION_REPORT",
        "VALIDATION_INJECTION_RANKING_REPORT",
        "CACHE_ROOT",
        "OUTPUT_ROOT",
        "SHARD_STOP_EXCLUSIVE",
    ):
        environment[name] = str(tmp_path / name.lower())
    environment["SHARD_STOP_EXCLUSIVE"] = "2"
    return environment


def test_candidate_background_extension_requires_separate_base_root(
    tmp_path: Path,
) -> None:
    environment = _minimum_environment(tmp_path)
    environment["SHARD_START"] = "1"
    missing = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "requires BASE_OUTPUT_ROOT" in missing.stderr

    environment["BASE_OUTPUT_ROOT"] = environment["OUTPUT_ROOT"]
    aliased = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert aliased.returncode == 2
    assert "separate from the immutable base output" in aliased.stderr

    environment["BASE_OUTPUT_ROOT"] = str(tmp_path / "base-output")
    missing_decision = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_decision.returncode == 2
    assert "requires CAPACITY_EXTENSION_DECISION" in missing_decision.stderr


def test_candidate_background_extension_binds_authoritative_parent() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '--parent-plan "$PARENT_PLAN"' in source
    assert '"$CAPACITY_EXTENSION_DECISION" "$PARENT_PLAN"' in source
    assert '"$BASE_OUTPUT_ROOT/shard-$shard/streamed_background_shard_report.json"' in source
    assert 'get("parent_plan_sha256") != digest' in source
