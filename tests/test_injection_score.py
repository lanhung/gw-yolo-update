from __future__ import annotations

import json
from pathlib import Path

import pytest

from gwyolo.injection_score import _load_resumable_rows, _save_progress
from gwyolo.io import file_sha256


def test_injection_score_resume_verifies_identity_and_probability_hash(tmp_path: Path) -> None:
    probability = tmp_path / "probability.npz"
    probability.write_bytes(b"probability")
    identity = {"save_probabilities": True, "manifest_sha256": "manifest"}
    manifest = [{"injection_id": "one"}, {"injection_id": "two"}]
    rows = [
        {
            "injection_id": "one",
            "probability_path": str(probability),
            "probability_sha256": file_sha256(probability),
        }
    ]
    _save_progress(tmp_path, rows, identity, requested=2)
    assert _load_resumable_rows(tmp_path, identity, manifest) == rows

    probability.write_bytes(b"changed")
    with pytest.raises(ValueError, match="probability hash mismatch"):
        _load_resumable_rows(tmp_path, identity, manifest)


def test_injection_score_resume_rejects_changed_run_identity(tmp_path: Path) -> None:
    (tmp_path / "injection_score_state.json").write_text(
        json.dumps(
            {
                "status": "in_progress",
                "run_identity": {"save_probabilities": False, "manifest_sha256": "old"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different run"):
        _load_resumable_rows(
            tmp_path,
            {"save_probabilities": False, "manifest_sha256": "new"},
            [{"injection_id": "one"}],
        )
