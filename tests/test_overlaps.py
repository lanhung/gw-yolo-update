from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.overlaps import (
    audit_physical_overlap_manifests,
    materialize_physical_overlaps,
    pair_overlap_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_overlap_pairing_is_unique_detector_compatible_and_deterministic() -> None:
    glitches = [
        {"split": "train", "glitch_id": f"g{i}", "network_gps_block": f"gb{i}", "ifo": ifo}
        for i, ifo in enumerate(("H1", "L1", "V1"))
    ]
    injections = [
        {"split": "train", "injection_id": f"i{i}", "waveform_id": f"w{i}", "ifos": ifos}
        for i, ifos in enumerate((["H1"], ["L1", "V1"], ["H1", "V1"]))
    ]
    first = pair_overlap_rows(glitches, injections, "train", seed=12, limit=3)
    second = pair_overlap_rows(glitches, injections, "train", seed=12, limit=3)
    assert [(g["glitch_id"], i["injection_id"]) for g, i in first] == [
        (g["glitch_id"], i["injection_id"]) for g, i in second
    ]
    assert len({i["injection_id"] for _, i in first}) == 3
    assert all(g["ifo"] in i["ifos"] for g, i in first)


def test_physical_overlap_materializes_fresh_transform_and_explicit_availability(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """overlap_factory:
  model_ifos: [H1, L1, V1]
  q_values: [4]
  target_sample_rate: 64
  tensor:
    whitening: self
    target_whitening: morphology
    mask_fraction: 0.08
    frequency_bins: 8
    time_bins: 8
    fmin: 4
    fmax: 30
""",
        encoding="utf-8",
    )
    raw = np.sin(np.linspace(0, 12 * np.pi, 128, endpoint=False)) * 1e-21
    weak_mask = np.zeros((3, 1, 8, 8), dtype=np.uint8)
    weak_mask[1, 0, 2:5, 3:6] = 1
    glitch_sample = tmp_path / "glitch.npz"
    np.savez(
        glitch_sample,
        raw_strain=raw,
        glitch_mask=weak_mask,
        ifos=np.asarray(["H1", "L1", "V1"]),
        q_values=np.asarray([4], dtype=np.float32),
        sample_rate=np.asarray(64),
    )
    glitch_row = {
        "split": "train",
        "glitch_id": "g1",
        "network_gps_block": "O3a:100:8",
        "ifo": "L1",
        "event_time": 100.0,
        "observing_run": "O3a",
        "ml_label": "Blip",
        "path": str(glitch_sample),
        "sha256": file_sha256(glitch_sample),
        "mask_provenance": "weak-test",
        "human_pixel_mask": False,
    }
    gravity_manifest = tmp_path / "gravity.jsonl"
    _write_jsonl(gravity_manifest, [glitch_row])

    signal = np.zeros((2, 256), dtype=np.float64)
    phase = np.linspace(0, 20 * np.pi, 128, endpoint=False)
    signal[1, 64:192] = np.sin(phase) * np.linspace(0.1, 1.0, 128) * 2e-22
    injection_sample = tmp_path / "injection.npz"
    np.savez(
        injection_sample,
        signal=signal,
        noise=np.zeros_like(signal),
        strain=signal,
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(64),
        context_gps_start=np.asarray(90.0),
        analysis_gps_start=np.asarray(91.0),
        analysis_start_index=np.asarray(64),
        analysis_stop_index=np.asarray(192),
    )
    injection_row = {
        "split": "train",
        "injection_id": "i1",
        "waveform_id": "w1",
        "gps_block": "O3a:200:8",
        "ifos": ["H1", "L1"],
        "source_family": "BBH",
        "materialized_path": str(injection_sample),
        "materialized_sha256": file_sha256(injection_sample),
    }
    injection_manifest = tmp_path / "injections.jsonl"
    _write_jsonl(injection_manifest, [injection_row])

    report = materialize_physical_overlaps(
        gravity_manifest,
        injection_manifest,
        config,
        tmp_path / "output",
        "train",
        seed=7,
    )
    assert report["rows"] == 1
    assert report["rendered_image_count"] == 0
    assert report["network_coherence_claim_allowed"] is False
    row = json.loads(Path(report["manifest_path"]).read_text().strip())
    with np.load(row["path"], allow_pickle=False) as arrays:
        assert arrays["features"].shape == (3, 1, 8, 8)
        assert arrays["chirp_mask"].shape == (3, 1, 8, 8)
        assert arrays["detector_availability"].tolist() == [0, 1, 0]
        assert np.count_nonzero(arrays["features"][[0, 2]]) == 0
        assert np.count_nonzero(arrays["chirp_mask"][1]) > 0
        assert np.array_equal(arrays["glitch_mask"], weak_mask)
        assert arrays["mixture_strain"][1] == pytest.approx(
            arrays["raw_glitch_strain"][1] + arrays["signal_strain"][1]
        )


def test_overlap_cross_split_audit_rejects_reused_waveform_or_glitch(tmp_path) -> None:
    base = {
        "mixture_id": "m-train",
        "injection_id": "i-train",
        "waveform_id": "shared-waveform",
        "glitch_id": "g-train",
        "injection_gps_block": "injection-block-train",
        "gps_block": "block-train",
        "network_gps_block": "block-train",
        "split": "train",
    }
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, [base])
    _write_jsonl(
        val,
        [
            {
                **base,
                "mixture_id": "m-val",
                "injection_id": "i-val",
                "glitch_id": "g-val",
                "injection_gps_block": "injection-block-val",
                "gps_block": "block-val",
                "network_gps_block": "block-val",
                "split": "val",
            }
        ],
    )
    with pytest.raises(ValueError, match="split leakage"):
        audit_physical_overlap_manifests([train, val], tmp_path / "audit.json")
