from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from gwyolo.injection_score import (
    _load_resumable_rows,
    _save_progress,
    apply_analysis_override,
)
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


def test_analysis_override_replaces_only_locked_crop(tmp_path: Path) -> None:
    override = tmp_path / "cleaned.npz"
    np.savez(
        override,
        cleaned_strain=np.asarray([[10.0, 11.0], [20.0, 21.0]]),
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(2),
        analysis_gps_start=np.asarray(101.0),
    )
    row = {
        "analysis_override_path": str(override),
        "analysis_override_sha256": file_sha256(override),
    }
    original = np.arange(8, dtype=np.float64).reshape(2, 4)
    context = {
        "ifos": ["H1", "L1"],
        "sample_rate": 2,
        "analysis_gps_start": 101.0,
        "analysis_start_index": 1,
        "analysis_stop_index": 3,
        "mixture": original,
    }
    updated, record = apply_analysis_override(row, context)
    assert updated["mixture"].tolist() == [[0.0, 10.0, 11.0, 3.0], [4.0, 20.0, 21.0, 7.0]]
    assert np.array_equal(context["mixture"], original)
    assert record["analysis_override_sha256"] == file_sha256(override)
