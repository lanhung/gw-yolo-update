from __future__ import annotations

import os
import re
import subprocess
import sys
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


def test_candidate_background_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) >= 4
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_candidate_background_propagates_five_seed_selector_failure(
    tmp_path: Path,
) -> None:
    environment = _minimum_environment(tmp_path)
    for name in ("task_code_dir", "scoring_code_dir"):
        (tmp_path / name / "src" / "gwyolo").mkdir(parents=True)
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "FIVE_SEED_SUMMARY": str(tmp_path / "missing-five-seed.json"),
            "UNIFORM_CONFIG": str(tmp_path / "uniform.yaml"),
            "FAMILY_BALANCED_CONFIG": str(tmp_path / "balanced.yaml"),
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
