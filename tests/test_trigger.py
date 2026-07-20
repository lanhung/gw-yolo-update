from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256

from gwyolo.trigger import (
    _load_resumable_trigger_rows,
    _save_trigger_progress,
    _window_strain,
    coherence_assisted_summary,
    network_ranking,
    probability_summaries,
)


def test_coherence_assisted_score_uses_physical_lag_roi_by_hand() -> None:
    strain = np.zeros((2, 200), dtype=np.float64)
    strain[0, 100] = 1.0
    strain[1, 101] = -1.0
    peaks = {
        "H1": {"offset_seconds": 1.0, "gps": 1001.0},
        "L1": {"offset_seconds": 1.0, "gps": 1001.0},
    }
    result = coherence_assisted_summary(
        strain,
        ("H1", "L1"),
        ["H1", "L1"],
        100,
        peaks,
        0.8,
        {"H1-L1": 0.01},
        0.0,
        1.0,
    )
    assert result["coherence_evaluable"] is True
    assert result["coherence_mean_absolute_correlation"] == 1.0
    assert result["coherence_features"][0]["lag_samples"] == 1
    assert result["coherence_assisted_score"] == pytest.approx(0.8)
    assert result["strain_envelope_peak_times"]["H1"]["sample_index"] == 100
    assert result["strain_envelope_peak_times"]["L1"]["sample_index"] == 101
    assert result["strain_envelope_arrival_gate"]["passed"] is True
    assert json.loads(json.dumps(result)) == result


def test_network_ranking_uses_second_loudest_valid_ifo() -> None:
    result = network_ranking(
        {"H1": 0.9, "L1": 0.8, "V1": 0.2},
        {"H1": 0.1, "L1": 0.3, "V1": 0.7},
        ["H1", "L1"],
    )
    assert result["ranking_score"] == 0.8
    assert result["maximum_glitch_score"] == 0.3
    assert result["chirp_glitch_margin"] == 0.5
    assert result["network_mode"] == "coincident"


def test_single_ifo_ranking_is_explicitly_diagnostic() -> None:
    result = network_ranking({"H1": 0.7}, {"H1": 0.2}, ["H1"])
    assert result["ranking_score"] == 0.7
    assert result["network_mode"] == "single_ifo_diagnostic"


def test_probability_summary_peak_time_and_network_score_by_hand() -> None:
    probabilities = np.zeros((2, 2, 1, 1, 4), dtype=np.float32)
    probabilities[0, 0, 0, 0] = [0.1, 0.8, 0.2, 0.1]
    probabilities[0, 1, 0, 0] = [0.1, 0.2, 0.7, 0.1]
    probabilities[1, 0, 0, 0] = [0.3, 0.1, 0.1, 0.1]
    probabilities[1, 1, 0, 0] = [0.1, 0.4, 0.1, 0.1]
    result = probability_summaries(
        probabilities, ("H1", "L1"), ["H1", "L1"], 1000.0, 8.0
    )
    assert result["ranking_score"] == np.float32(0.7)
    assert result["peak_times"]["chirp"]["H1"]["gps"] == 1003.0
    assert result["peak_times"]["chirp"]["L1"]["gps"] == 1005.0


def test_trigger_resume_reuses_only_matching_window_ids(tmp_path: Path) -> None:
    identity = {"manifest_sha256": "same"}
    rows = [{"window_id": "one", "ranking_score": 0.5}]
    manifest = [{"window_id": "one"}, {"window_id": "two"}]
    _save_trigger_progress(tmp_path, rows, identity, requested=2)
    assert _load_resumable_trigger_rows(tmp_path, identity, manifest) == rows

    partial = tmp_path / "background_triggers.partial.jsonl"
    partial.write_text(
        json.dumps({"window_id": "not-requested", "ranking_score": 0.5}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        _load_resumable_trigger_rows(tmp_path, identity, manifest)


def test_trigger_resume_rejects_changed_identity(tmp_path: Path) -> None:
    _save_trigger_progress(tmp_path, [], {"manifest_sha256": "old"}, requested=1)
    with pytest.raises(ValueError, match="different run"):
        _load_resumable_trigger_rows(
            tmp_path,
            {"manifest_sha256": "new"},
            [{"window_id": "one"}],
        )


def test_window_override_is_inserted_before_full_context_whitening(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.hdf5"
    source.write_bytes(b"source")
    override = tmp_path / "cleaned.npz"
    np.savez(
        override,
        cleaned_strain=np.asarray([[10.0, 11.0], [0.0, 0.0]]),
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(2),
        analysis_gps_start=np.asarray(101.0),
    )
    monkeypatch.setattr(
        "gwyolo.trigger.read_hdf5_segment",
        lambda *args: {
            "strain": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            "sample_rate": 2,
        },
    )
    monkeypatch.setattr("gwyolo.trigger._whiten", lambda values: values)
    row = {
        "gps_start": 101.0,
        "gps_end": 102.0,
        "duration": 1.0,
        "ifos": ["H1"],
        "source_files": {"H1": {"path": str(source)}},
        "analysis_override_path": str(override),
        "analysis_override_sha256": file_sha256(override),
    }
    strain, valid_ifos, record = _window_strain(
        row, ("H1", "L1"), 2, context_duration=4.0
    )
    assert valid_ifos == ["H1"]
    assert strain.tolist() == [[10.0, 11.0], [0.0, 0.0]]
    assert record["analysis_override_sha256"] == file_sha256(override)
