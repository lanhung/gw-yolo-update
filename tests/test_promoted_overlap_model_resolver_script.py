from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gwyolo.io import file_sha256


SCRIPT = (
    Path(__file__).parents[1] / "scripts/resolve_promoted_overlap_model.sh"
)


@pytest.mark.parametrize(
    ("arm", "config_index"),
    [("uniform", 0), ("family_balanced", 1), ("glitch_adapter", 2)],
)
def test_promoted_model_resolver_supports_all_frozen_arms(
    tmp_path: Path, arm: str, config_index: int
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    configs = []
    for name in ("uniform", "family", "adapter"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"name: {name}\n")
        configs.append(path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "completed_five_seed_source_safe_overlap_validation",
                "passed": True,
                "test_data_opened": False,
                "promoted_arm": arm,
                "five_seed_stability": {
                    "status": "five_seed_reproducibility_gate_v1",
                    "passed": True,
                },
                "selected_checkpoint_path": str(checkpoint),
                "selected_checkpoint_sha256": file_sha256(checkpoint),
                "common_artifact_hashes": {
                    "config_file_sha256": file_sha256(configs[config_index])
                },
            }
        )
    )
    completed = subprocess.run(
        ["bash", str(SCRIPT), str(summary), *(str(path) for path in configs)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "TASK_PYTHON": sys.executable},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        arm,
        str(checkpoint.resolve()),
        str(configs[config_index].resolve()),
    ]


def test_promoted_model_resolver_rejects_unbound_adapter_config(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}")
    completed = subprocess.run(
        ["bash", str(SCRIPT), str(summary), str(summary), str(summary), str(summary)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "TASK_PYTHON": sys.executable},
    )
    assert completed.returncode != 0


def test_promoted_model_resolver_requires_audit_for_mixed_training_commits(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    configs = []
    for name in ("uniform", "family", "adapter"):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"name: {name}\n")
        configs.append(path)
    reports = []
    for index, commit in enumerate(("a" * 40, "b" * 40)):
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps({"code_commit": commit}))
        reports.append({"path": str(path), "sha256": file_sha256(path)})
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "completed_five_seed_source_safe_overlap_validation",
                "passed": True,
                "test_data_opened": False,
                "promoted_arm": "glitch_adapter",
                "five_seed_stability": {
                    "status": "five_seed_reproducibility_gate_v1",
                    "passed": True,
                },
                "selected_checkpoint_path": str(checkpoint),
                "selected_checkpoint_sha256": file_sha256(checkpoint),
                "common_artifact_hashes": {
                    "config_file_sha256": file_sha256(configs[2])
                },
                "finetune_reports": reports,
            }
        )
    )
    command = [
        "bash",
        str(SCRIPT),
        str(summary),
        *(str(path) for path in configs),
    ]
    environment = {**os.environ, "TASK_PYTHON": sys.executable}
    missing = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert missing.returncode != 0
    assert "lack a compatibility audit" in missing.stderr

    compatibility = tmp_path / "compatibility.json"
    compatibility.write_text(
        json.dumps(
            {
                "status": "audited_overlap_training_code_compatibility",
                "passed": True,
                "test_data_opened": False,
                "audited_commits": ["a" * 40, "b" * 40],
                "checks": {"training_surface_identical": True},
            }
        )
    )
    accepted = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env={
            **environment,
            "MODEL_TRAINING_COMPATIBILITY_REPORT": str(compatibility),
        },
    )
    assert accepted.returncode == 0, accepted.stderr
