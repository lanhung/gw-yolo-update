from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

from gwyolo.io import file_sha256
from gwyolo.learned_deglitch import (
    run_learned_background_deglitch,
    signal_retention_metrics,
)


def test_signal_retention_metrics_match_half_signal_by_hand() -> None:
    noise = np.asarray([[10.0, 20.0]])
    signal = np.asarray([[2.0, 4.0]])
    mixture = noise + signal
    cleaned = noise + 0.5 * signal
    metrics = signal_retention_metrics(mixture, cleaned, noise, signal)
    assert metrics["network_signal_projection_retention"] == 0.5
    assert metrics["signal_projection_retention_by_ifo"] == [0.5]
    assert math.isclose(metrics["waveform_change_rms"], math.sqrt(2.5))
    assert math.isclose(metrics["postclean_signal_error_rms"], math.sqrt(2.5))


def test_background_deglitch_writes_rescorable_override(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.hdf5"
    source.write_bytes(b"verified source")
    background = tmp_path / "background.jsonl"
    background.write_text(
        json.dumps(
            {
                "window_id": "window-one",
                "split": "val",
                "gps_start": 100.0,
                "gps_end": 102.0,
                "duration": 2.0,
                "gps_block": "block-one",
                "ifos": ["H1"],
                "source_files": {
                    "H1": {"path": str(source), "sha256": file_sha256(source)}
                },
            }
        )
        + "\n"
    )
    probability = tmp_path / "probability.npz"
    chirp = np.zeros((3, 1, 8, 8), dtype=np.float32)
    glitch = np.zeros_like(chirp)
    glitch[0, 0, 2:5, 2:6] = 1.0
    np.savez(
        probability,
        chirp_probability=chirp,
        glitch_probability=glitch,
        ifos=np.asarray(["H1", "L1", "V1"]),
        q_values=np.asarray([4]),
    )
    scored = tmp_path / "scores.jsonl"
    scored.write_text(
        json.dumps(
            {
                "window_id": "window-one",
                "probability_path": str(probability),
                "probability_sha256": file_sha256(probability),
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        "gwyolo.learned_deglitch.read_hdf5_segment",
        lambda *args: {
            "strain": np.sin(np.linspace(0, 30, 256)) * 1e-21,
            "sample_rate": 64,
        },
    )
    report = run_learned_background_deglitch(
        background,
        scored,
        tmp_path / "cleaned",
        model_ifos=("H1", "L1", "V1"),
        target_sample_rate=64,
        context_duration=4.0,
        required_split="val",
    )
    assert report["windows"] == 1
    row = json.loads(Path(report["manifest_path"]).read_text().strip())
    assert row["window_id"] == "window-one"
    assert row["analysis_override_sha256"] == file_sha256(
        row["analysis_override_path"]
    )
    with np.load(row["analysis_override_path"], allow_pickle=False) as arrays:
        assert arrays["cleaned_strain"].shape == (3, 128)
        assert np.count_nonzero(arrays["cleaned_strain"][1:]) == 0
