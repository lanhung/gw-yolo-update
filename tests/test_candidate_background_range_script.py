from __future__ import annotations

import hashlib
import json
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
        "INDEPENDENT_VALIDATION_ENDPOINT_REPORT",
        "PARENT_PLAN",
        "VALIDATION_PURPOSE_AUDIT",
        "CAPACITY_FORECAST",
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
    assert '"$INDEPENDENT_VALIDATION_ENDPOINT_REPORT"' in source
    assert '"$VALIDATION_PURPOSE_AUDIT"' in source
    assert '"$CAPACITY_FORECAST"' in source
    assert "candidate-background-plan-authorize" in source
    assert "publication_background_plan_authorization.json" in source
    assert '"$CAPACITY_EXTENSION_DECISION" "$PARENT_PLAN"' in source
    assert '"$BASE_OUTPUT_ROOT/shard-$shard/streamed_background_shard_report.json"' in source
    assert 'get("parent_plan_sha256") != digest' in source
    assert "background shard $shard exhausted bounded retries" in source
    assert "MAX_ATTEMPTS" in source
    assert "--verified-source-inventory" in source


def test_candidate_background_embedded_python_compiles() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    snippets = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(snippets) >= 4
    for index, snippet in enumerate(snippets):
        compile(snippet, f"{SCRIPT.name}:heredoc-{index}", "exec")


def test_candidate_background_rejects_audit_for_another_plan(tmp_path: Path) -> None:
    environment = _minimum_environment(tmp_path)
    environment.update(
        {
            "TASK_PYTHON": sys.executable,
            "TASK_CODE_DIR": str(SCRIPT.parents[1]),
            "SCORING_CODE_DIR": str(SCRIPT.parents[1]),
            "CHECKPOINT": str(tmp_path / "checkpoint.pt"),
            "CONFIG": str(tmp_path / "config.yaml"),
            "SHARD_STOP_EXCLUSIVE": "1",
        }
    )
    for name in (
        "promotion_report",
        "promoted_pipeline_report",
        "event_exclusions",
        "checkpoint.pt",
        "config.yaml",
        "coherence_config",
        "timing_calibration_report",
        "validation_injection_ranking_report",
    ):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    purpose_path = tmp_path / "purpose.json"
    purpose_path.write_text("{}\n", encoding="utf-8")
    purpose_hash = hashlib.sha256(purpose_path.read_bytes()).hexdigest()
    endpoint = {
        "status": "frozen_gps_and_purpose_disjoint_validation_endpoint",
        "passed": True,
        "scientific_claim_allowed": False,
        "rows": 3000,
        "candidate_calibration_unique_gps_blocks": 25,
        "injection_validation_unique_gps_blocks": 25,
        "purpose_gps_block_overlap": 0,
        "test_rows_read": 0,
        "test_evaluation": None,
        "component_reports": {
            "purpose_partition": {
                "path": str(purpose_path),
                "sha256": purpose_hash,
            }
        },
    }
    plan = {
        "status": "development_acquisition_plan",
        "run": "O4a",
        "locked_evaluation_data": False,
        "selected_pairs": 4,
        "candidate_scores_inspected": False,
        "test_data_opened": False,
        "pairs": [{"pair_id": f"pair-{index}"} for index in range(4)],
    }
    endpoint_path = Path(environment["INDEPENDENT_VALIDATION_ENDPOINT_REPORT"])
    plan_path = Path(environment["PARENT_PLAN"])
    endpoint_path.write_text(json.dumps(endpoint), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    audit = {
        "status": "verified_gwosc_plan_validation_purpose_disjointness",
        "passed": True,
        "scientific_claim_allowed": False,
        "candidate_scores_inspected": False,
        "test_rows_read": 0,
        "overlap_pair_ids": [],
        "overlap_gps_blocks": [],
        "plan": {"sha256": "b" * 64},
        "purpose_partition": {"sha256": purpose_hash},
        "roles": {
            role: {
                "gps_interval_overlap_count": 0,
                "direct_pair_id_overlaps": [],
            }
            for role in ("candidate_calibration", "injection_validation")
        },
    }
    forecast = {
        "status": "score_blind_candidate_block_capacity_forecast",
        "scientific_claim_allowed": False,
        "forecast_only": True,
        "candidate_scores_inspected": False,
        "planned_pairs_satisfy_safety_forecast": True,
        "recommendation_fits_available_pairs": True,
        "planned_parent_plan_sha256": plan_hash,
        "planned_source_pairs": 4,
        "recommended_minimum_source_pairs": 4,
        "safety_factor": 1.5,
        "target_far_per_year": 0.1,
        "zero_count_confidence": 0.9,
    }
    Path(environment["VALIDATION_PURPOSE_AUDIT"]).write_text(
        json.dumps(audit), encoding="utf-8"
    )
    Path(environment["CAPACITY_FORECAST"]).write_text(
        json.dumps(forecast), encoding="utf-8"
    )

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "validation-purpose audit does not authorize" in completed.stderr


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
