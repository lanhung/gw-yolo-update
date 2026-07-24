from pathlib import Path

import pytest

from gwyolo.training_compatibility import (
    audit_overlap_training_code_compatibility,
)


ROOT = Path(__file__).parents[1]


def test_adapter_training_surface_is_identical_across_successor_commits(
    tmp_path: Path,
) -> None:
    result = audit_overlap_training_code_compatibility(
        ROOT,
        [
            "cf877fc716b5df51a8518a52a1aa78dbaefc107f",
            "790e9dc9c339212dedaee6475de033457a249d4f",
        ],
        "configs/physical_overlap_finetune_glitch_adapter.yaml",
        tmp_path / "compatibility.json",
    )
    assert result["passed"] is True
    assert len(result["audited_commits"]) == 2
    assert result["checks"]["overlap_training_surface_identical"] is True
    assert all(result["checks"].values())
    hashes = {
        row["overlap_training_surface_sha256"]
        for row in result["revisions"].values()
    }
    assert len(hashes) == 1


def test_training_compatibility_rejects_one_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least two commits"):
        audit_overlap_training_code_compatibility(
            ROOT,
            ["HEAD"],
            "configs/physical_overlap_finetune_glitch_adapter.yaml",
            tmp_path / "compatibility.json",
        )
