from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "run_post_five_seed_candidate_publication_chain.sh"
)


def test_post_five_seed_candidate_chain_is_fail_closed_and_ordered() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "SOURCE_SAFE_CHAIN_RECEIPT" in source
    assert 'receipt.get("five_seed_promoted") is not True' in source
    assert "write_negative_receipt" in source
    assert "continuous_background_started" in source
    assert "run_promoted_candidate_validation.sh" in source
    assert "run_candidate_validation_comparison.sh" in source
    assert "run_candidate_background_range.sh" in source
    assert "run_candidate_validation_detector_set_successor.sh" not in source
    assert source.index("run_promoted_candidate_validation.sh") < source.index(
        "run_candidate_validation_comparison.sh"
    )
    assert source.index("run_candidate_validation_comparison.sh") < source.index(
        "run_candidate_background_range.sh"
    )
    assert "completed_post_five_seed_h1l1_candidate_publication_chain" in source
    assert '"variable_detector_calibration_frozen": False' in source


def test_post_five_seed_candidate_chain_binds_locked_inputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for value in (
        "INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
        "PARENT_PLAN",
        "VALIDATION_PURPOSE_AUDIT",
        "CAPACITY_FORECAST",
        "EVENT_EXCLUSIONS",
        "FIVE_SEED_SUMMARY",
    ):
        assert value in source
    assert "purpose_gps_block_overlap" in source
    assert "candidate_calibration_background_manifest_sha256" in source
    assert "injection_arrival_manifest_sha256" in source
    assert 'int(endpoint.get("test_rows_read", -1)) != 0' in source
    assert "publication_calibration_eligible" in source


def test_post_five_seed_candidate_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) >= 6
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_post_five_seed_candidate_chain_retains_negative_gate(
    tmp_path: Path,
) -> None:
    repository = SCRIPT.parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_receipt = tmp_path / "source-safe-receipt.json"
    source_receipt.write_text(
        json.dumps(
            {
                "status": "completed_source_safe_overlap_negative_promotion",
                "execution_passed": True,
                "five_seed_executed": False,
                "five_seed_promoted": False,
                "scientific_claim_allowed": False,
                "search_claim_allowed": False,
                "test_rows_read": 0,
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "TASK_CODE_DIR": str(repository),
            "GWYOLO_CODE_COMMIT": commit,
            "SOURCE_SAFE_CHAIN_RECEIPT": str(source_receipt),
            "FIVE_SEED_SUMMARY": str(tmp_path / "absent-summary.json"),
            "CACHE_ROOT": str(tmp_path / "cache"),
            "OUTPUT_ROOT": str(tmp_path / "output"),
        }
    )
    for variable in (
        "INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
        "BASELINE_CHECKPOINT",
        "BASELINE_CONFIG",
        "UNIFORM_CONFIG",
        "FAMILY_BALANCED_CONFIG",
        "COHERENCE_CONFIG",
        "PROMOTION_CONFIG",
        "PARENT_PLAN",
        "VALIDATION_PURPOSE_AUDIT",
        "CAPACITY_FORECAST",
        "EVENT_EXCLUSIONS",
    ):
        path = tmp_path / variable.lower()
        path.write_text("{}\n", encoding="utf-8")
        environment[variable] = str(path)

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(
        (tmp_path / "output" / "post_five_seed_candidate_publication_receipt.json")
        .read_text(encoding="utf-8")
    )
    assert (
        receipt["status"]
        == "completed_post_five_seed_candidate_negative_promotion"
    )
    assert receipt["continuous_background_started"] is False
    assert receipt["test_rows_read"] == 0
