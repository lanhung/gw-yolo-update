from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.overlaps import (
    _fft_upsample,
    audit_physical_overlap_manifests,
    build_contaminated_injection_overrides,
    freeze_physical_overlap_scaling_hard_subset,
    freeze_physical_overlap_scaling_subsets,
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
    assert all(set(g.get("available_ifos", [g["ifo"]])) <= set(i["ifos"]) for g, i in first)


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

    corpus_audit = tmp_path / "corpus-audit.json"
    corpus_audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
                "train_manifest_sha256": file_sha256(gravity_manifest),
                "validation_manifest_sha256": "unused-in-this-test",
                "split_audit": {
                    "cross_split_overlaps": {
                        "glitch_id": [],
                        "network_gps_block": [],
                        "source_hdf5_url": [],
                        "numeric_sample_sha256": [],
                    }
                },
            }
        )
    )

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
        gravityspy_corpus_audit=corpus_audit,
    )
    assert report["rows"] == 1
    assert report["rendered_image_count"] == 0
    assert report["network_coherence_claim_allowed"] is False
    assert report["gravityspy_corpus_audit_sha256"] == file_sha256(corpus_audit)
    bad_audit = tmp_path / "bad-corpus-audit.json"
    bad_payload = json.loads(corpus_audit.read_text())
    bad_payload["train_manifest_sha256"] = "0" * 64
    bad_audit.write_text(json.dumps(bad_payload))
    with pytest.raises(ValueError, match="does not bind"):
        materialize_physical_overlaps(
            gravity_manifest,
            injection_manifest,
            config,
            tmp_path / "bad-output",
            "train",
            seed=7,
            gravityspy_corpus_audit=bad_audit,
        )
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


def test_network_overlap_adds_coherent_signal_to_every_available_ifo(tmp_path) -> None:
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
"""
    )
    samples = 128
    raw = np.zeros((3, samples), dtype=np.float64)
    raw[0] = np.sin(np.linspace(0, 15, samples)) * 1e-21
    raw[1] = np.cos(np.linspace(0, 15, samples)) * 1e-21
    gravity_path = tmp_path / "network-glitch.npz"
    weak = np.zeros((3, 1, 8, 8), dtype=np.uint8)
    weak[0, 0, 2:4, 3:5] = 1
    np.savez(
        gravity_path,
        raw_strain=raw,
        glitch_mask=weak,
        detector_availability=np.asarray([1, 1, 0]),
        ifos=np.asarray(["H1", "L1", "V1"]),
        q_values=np.asarray([4]),
        sample_rate=np.asarray(64),
    )
    gravity_manifest = tmp_path / "gravity.jsonl"
    _write_jsonl(
        gravity_manifest,
        [
            {
                "split": "val",
                "glitch_id": "network-g",
                "network_gps_block": "O2:network",
                "ifo": "H1",
                "available_ifos": ["H1", "L1"],
                "detector_availability": [1, 1, 0],
                "event_time": 100.0,
                "path": str(gravity_path),
                "sha256": file_sha256(gravity_path),
            }
        ],
    )
    signal = np.zeros((2, 256), dtype=np.float64)
    signal[0, 64:192] = np.sin(np.linspace(0, 20, samples)) * 2e-22
    signal[1, 64:192] = np.sin(np.linspace(0.2, 20.2, samples)) * 3e-22
    injection_path = tmp_path / "network-injection.npz"
    np.savez(
        injection_path,
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
    injection_manifest = tmp_path / "injection.jsonl"
    _write_jsonl(
        injection_manifest,
        [
            {
                "split": "val",
                "injection_id": "network-i",
                "waveform_id": "network-w",
                "gps_block": "O4a:network",
                "ifos": ["H1", "L1"],
                "materialized_path": str(injection_path),
                "materialized_sha256": file_sha256(injection_path),
            }
        ],
    )
    report = materialize_physical_overlaps(
        gravity_manifest, injection_manifest, config, tmp_path / "output", "val"
    )
    row = json.loads(Path(report["manifest_path"]).read_text().strip())
    assert row["available_ifos"] == ["H1", "L1"]
    assert report["detector_subset_counts"] == {"H1L1": 1}
    with np.load(row["path"], allow_pickle=False) as arrays:
        assert arrays["detector_availability"].tolist() == [1, 1, 0]
        assert np.count_nonzero(arrays["signal_strain"][0]) > 0
        assert np.count_nonzero(arrays["signal_strain"][1]) > 0
        assert np.count_nonzero(arrays["signal_strain"][2]) == 0
        assert np.count_nonzero(arrays["chirp_mask"][:2]) > 0
    contaminated = build_contaminated_injection_overrides(
        report["manifest_path"],
        injection_manifest,
        tmp_path / "contaminated",
        "val",
    )
    contaminated_row = json.loads(
        Path(contaminated["manifest_path"]).read_text().strip()
    )
    clean_row = json.loads(
        Path(contaminated["paired_clean_manifest_path"]).read_text().strip()
    )
    assert clean_row["injection_id"] == contaminated_row["injection_id"]
    assert clean_row["waveform_id"] == contaminated_row["waveform_id"]
    assert contaminated_row["analysis_override_kind"] == "real_glitch_contaminated"
    assert contaminated_row["glitch_id"] == "network-g"
    with np.load(contaminated_row["analysis_override_path"], allow_pickle=False) as arrays:
        assert arrays["analysis_strain"].shape == (2, samples)
        assert arrays["ifos"].tolist() == ["H1", "L1"]


def test_fft_upsample_preserves_bandlimited_amplitude_and_samples() -> None:
    time = np.arange(64) / 64.0
    signal = np.sin(2 * np.pi * 5 * time)
    upsampled = _fft_upsample(signal, 64, 256)
    assert upsampled.shape == (256,)
    assert upsampled[::4] == pytest.approx(signal, abs=1e-12)
    assert np.max(np.abs(upsampled)) == pytest.approx(1.0, abs=1e-12)


def test_fft_upsample_splits_even_source_nyquist_bin() -> None:
    signal = (-1.0) ** np.arange(16)
    upsampled = _fft_upsample(signal, 16, 32)
    assert upsampled[::2] == pytest.approx(signal, abs=1e-12)
    assert upsampled[1::2] == pytest.approx(0.0, abs=1e-12)


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


def test_overlap_scaling_subsets_are_nested_group_safe_and_hash_verified(
    tmp_path: Path,
) -> None:
    corpus_audit = tmp_path / "corpus-audit.json"
    corpus_audit.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
            }
        )
    )
    corpus_audit_sha256 = file_sha256(corpus_audit)
    manifests = {}
    for split, count in (("train", 6), ("val", 2)):
        rows = []
        for index in range(count):
            suffix = f"{split}-{index}"
            artifact = tmp_path / f"{suffix}.npz"
            artifact.write_bytes(suffix.encode())
            rows.append(
                {
                    "split": split,
                    "mixture_id": f"m-{suffix}",
                    "injection_id": f"i-{suffix}",
                    "waveform_id": f"w-{suffix}",
                    "glitch_id": f"g-{suffix}",
                    "injection_gps_block": f"ib-{suffix}",
                    "gps_block": f"gb-{suffix}",
                    "network_gps_block": f"gb-{suffix}",
                    "path": str(artifact),
                    "sha256": file_sha256(artifact),
                    "ml_label": "Blip" if index % 2 == 0 else "Tomte",
                    "source_family": "BBH" if index % 2 == 0 else "NSBH",
                    "available_ifos": ["H1", "L1"] if index % 2 == 0 else ["H1"],
                    "gravityspy_corpus_audit_sha256": corpus_audit_sha256,
                }
            )
        manifest = tmp_path / f"{split}.jsonl"
        _write_jsonl(manifest, rows)
        manifests[split] = manifest

    result = freeze_physical_overlap_scaling_subsets(
        manifests["train"],
        manifests["val"],
        corpus_audit,
        [2, 4],
        tmp_path / "scales",
        seed=17,
        include_full=True,
    )
    assert result["passed"] is True
    assert result["scales"] == [2, 4, 6]
    assert result["test_rows_read"] == 0
    assert result["required_training_controls"] == [
        "fixed_epochs",
        "fixed_optimizer_updates",
    ]
    subsets = []
    for identity in result["subsets"]:
        rows = [
            json.loads(line)
            for line in Path(identity["manifest_path"]).read_text().splitlines()
        ]
        assert len(rows) == identity["scale"]
        assert identity["unique_physical_counts"]["glitch_id"] == len(rows)
        subsets.append({row["mixture_id"] for row in rows})
    assert subsets[0] < subsets[1] < subsets[2]

    tampered = tmp_path / "train-0.npz"
    tampered.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        freeze_physical_overlap_scaling_subsets(
            manifests["train"],
            manifests["val"],
            corpus_audit,
            [2],
            tmp_path / "tampered-scales",
        )


def test_overlap_scaling_hard_subset_freezes_four_score_blind_strata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GWYOLO_CODE_COMMIT", "a" * 40)
    corpus = tmp_path / "hard-corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "status": "verified_group_safe_gravityspy_aligned_network_corpus",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    corpus_sha = file_sha256(corpus)
    rows = []
    for index in range(100):
        artifact = tmp_path / f"hard-{index}.npz"
        artifact.write_bytes(f"hard-{index}".encode())
        rows.append(
            {
                "split": "val",
                "mixture_id": f"m-{index}",
                "injection_id": f"i-{index}",
                "waveform_id": f"w-{index}",
                "glitch_id": f"g-{index}",
                "injection_gps_block": f"ib-{index}",
                "network_gps_block": f"gb-{index}",
                "observing_run": "O3b",
                "ml_label": f"family-{index % 10}",
                "available_ifos": ["H1", "L1"],
                "optimal_snr_by_ifo": {"H1": 3.0, "L1": 4.0},
                "path": str(artifact),
                "sha256": file_sha256(artifact),
                "gravityspy_corpus_audit_sha256": corpus_sha,
            }
        )
    validation = tmp_path / "hard-validation.jsonl"
    _write_jsonl(validation, rows)
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "physical_overlap_scaling_hard_subset.yaml"
    )

    result = freeze_physical_overlap_scaling_hard_subset(
        validation,
        corpus,
        config,
        tmp_path / "hard-subset",
    )

    assert result["rows"] == 100
    assert result["candidate_scores_inspected"] is False
    assert result["model_outputs_inspected"] is False
    assert result["test_rows_read"] == 0
    assert set(result["strata"]) == {
        "low_network_snr",
        "missing_detector",
        "o3b_transfer",
        "rare_glitch_family",
    }
    assert all(value["rows"] == 100 for value in result["strata"].values())
    frozen_rows = [
        json.loads(line)
        for line in Path(result["hard_subset_manifest_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert frozen_rows[0]["hard_subset_network_snr"] == pytest.approx(5.0)
    assert len(frozen_rows[0]["hard_subset_strata"]) == 4
