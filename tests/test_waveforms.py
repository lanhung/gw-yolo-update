from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.waveforms import (
    annotate_detector_arrival_rows,
    evict_verified_background_bank_sources,
    load_materialized_context,
    materialize_background_bank,
    materialize_recipe,
    optimal_snr_stratum,
    pack_scaled_float16_signal,
    place_waveform_samples,
    run_injection_materialization,
    validate_recipe_identities,
    waveform_equivalence_metrics,
)


def test_detector_arrivals_use_geometric_delay_not_waveform_array_end() -> None:
    rows = [
        {
            "injection_id": "i1",
            "split": "val",
            "gps_time": 100.0,
            "right_ascension": 1.0,
            "declination": -0.5,
            "ifos": ["H1", "L1"],
            "signal_summary": {
                "H1": {"waveform_start_gps": 90.0, "waveform_samples": 30}
            },
        }
    ]
    delays = {"H1": 0.006, "L1": -0.004}
    annotated = annotate_detector_arrival_rows(
        rows, lambda ifo, _ra, _dec, _gps: delays[ifo]
    )[0]
    assert annotated["detector_arrival_gps"] == pytest.approx(
        {"H1": 100.006, "L1": 99.996}
    )
    assert annotated["geocenter_to_detector_delay_seconds"] == delays


def test_numeric_background_bank_survives_source_eviction(tmp_path) -> None:
    source = tmp_path / "strain.hdf5"
    with h5py.File(source, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("GPSstart", data=100)
        strain = handle.create_group("strain").create_dataset(
            "Strain", data=np.arange(32, dtype=np.float64)
        )
        strain.attrs["Xspacing"] = 0.25
    background = {
        "window_id": "bank-window",
        "gps_block": "bank-block",
        "split": "train",
        "gps_start": 103.0,
        "duration": 2.0,
        "ifos": ["H1"],
        "source_files": {"H1": {"path": str(source), "sha256": file_sha256(source)}},
    }
    manifest = tmp_path / "background.jsonl"
    manifest.write_text(json.dumps(background) + "\n", encoding="utf-8")
    bank_report = materialize_background_bank(
        manifest, tmp_path / "bank", target_sample_rate=4, context_duration=4.0
    )
    with open(bank_report["manifest_path"], encoding="utf-8") as handle:
        bank_background = json.loads(next(handle))
    assert bank_report["selected_windows"] == 1
    assert bank_report["unique_source_files"] == 1
    assert bank_report["splits"]["train"]["live_time_seconds"] == 2.0
    source.unlink()

    class FakeBackend:
        def generate(self, recipe, ifos, sample_rate):
            return {"H1": (103.0, np.asarray([1.0, 2.0]))}, {"H1": {"fake": True}}

    recipe = {
        "injection_id": "bank-injection",
        "background_window_id": "bank-window",
        "gps_block": "bank-block",
        "split": "train",
    }
    materialized = materialize_recipe(
        recipe,
        bank_background,
        FakeBackend(),
        4,
        tmp_path / "bank-injection.npz",
        context_duration=4.0,
        storage_mode="signal_scaled_float16",
    )
    loaded = load_materialized_context(materialized)
    assert loaded["noise"].shape == (1, 16)
    assert loaded["mixture"] == pytest.approx(loaded["noise"] + loaded["signal"])


def test_background_bank_source_eviction_requires_complete_hash_verification(tmp_path) -> None:
    cache = tmp_path / "bounded-cache"
    cache.mkdir()
    source = cache / "strain.hdf5"
    with h5py.File(source, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("GPSstart", data=100)
        strain = handle.create_group("strain").create_dataset(
            "Strain", data=np.arange(32, dtype=np.float64)
        )
        strain.attrs["Xspacing"] = 0.25
    background = {
        "window_id": "evict-window",
        "gps_block": "evict-block",
        "split": "train",
        "gps_start": 103.0,
        "duration": 2.0,
        "ifos": ["H1"],
        "source_files": {"H1": {"path": str(source), "sha256": file_sha256(source)}},
    }
    manifest = tmp_path / "evict-background.jsonl"
    manifest.write_text(json.dumps(background) + "\n", encoding="utf-8")
    bank_dir = tmp_path / "evict-bank"
    materialize_background_bank(manifest, bank_dir, 4, 4.0)
    eviction = evict_verified_background_bank_sources(
        bank_dir / "background_bank_report.json",
        cache,
        tmp_path / "eviction.json",
    )
    assert eviction["removed_files"] == 1
    assert eviction["recoverable"] is True
    assert not source.exists()

    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        evict_verified_background_bank_sources(
            bank_dir / "background_bank_report.json",
            cache,
            tmp_path / "must-not-write.json",
        )


def test_scaled_float16_signal_storage_preserves_physical_amplitude() -> None:
    signal = np.asarray([[0.0, 1e-24, -2e-24, 3e-24], [0.0, 0.0, 0.0, 0.0]])
    packed, peaks, metrics = pack_scaled_float16_signal(signal)
    reconstructed = packed.astype(np.float64) * peaks[:, None]
    assert packed.dtype == np.float16
    assert peaks.tolist() == pytest.approx([3e-24, 0.0])
    assert reconstructed == pytest.approx(signal, rel=1e-3, abs=1e-30)
    assert metrics["relative_l2_error"] < 1e-3
    assert metrics["normalized_overlap"] >= 0.999999


def test_optimal_snr_strata_boundaries() -> None:
    assert optimal_snr_stratum(3.999) == "snr_lt_4"
    assert optimal_snr_stratum(4.0) == "snr_4_8"
    assert optimal_snr_stratum(8.0) == "snr_8_15"
    assert optimal_snr_stratum(15.0) == "snr_15_30"
    assert optimal_snr_stratum(30.0) == "snr_ge_30"


def test_waveform_equivalence_metrics_exact_and_scaled() -> None:
    reference = np.asarray([0.0, 1.0 + 2.0j, -3.0j])
    exact = waveform_equivalence_metrics(reference, reference, -8.0, -8.0)
    assert exact["same_length"]
    assert exact["normalized_complex_overlap"] == pytest.approx(1.0)
    assert exact["relative_l2_error"] == 0.0
    assert exact["amplitude_norm_ratio"] == 1.0
    scaled = waveform_equivalence_metrics(reference * 2, reference, -7.5, -8.0)
    assert scaled["normalized_complex_overlap"] == pytest.approx(1.0)
    assert scaled["relative_l2_error"] == pytest.approx(1.0)
    assert scaled["amplitude_norm_ratio"] == pytest.approx(2.0)
    assert scaled["epoch_difference_seconds"] == pytest.approx(0.5)


def test_place_waveform_samples_clips_both_edges_by_hand() -> None:
    inside = place_waveform_samples(10.0, 2, 6, 11.0, np.asarray([1, 2, 3]))
    assert inside.tolist() == [0, 0, 1, 2, 3, 0]
    left = place_waveform_samples(10.0, 2, 6, 9.0, np.asarray([1, 2, 3, 4]))
    assert left.tolist() == [3, 4, 0, 0, 0, 0]
    right = place_waveform_samples(10.0, 2, 6, 12.5, np.asarray([7, 8, 9]))
    assert right.tolist() == [0, 0, 0, 0, 0, 7]


def test_place_waveform_interpolates_subsample_epoch() -> None:
    result = place_waveform_samples(10.0, 4, 16, 10.125, np.ones(32))
    assert np.isfinite(result).all()
    assert result[4:12] == pytest.approx(np.ones(8), abs=0.04)


def test_recipe_identity_audit_rejects_gps_leakage() -> None:
    rows = [
        {"injection_id": "i1", "waveform_id": "w1", "split": "val", "gps_block": "g"},
        {"injection_id": "i2", "waveform_id": "w2", "split": "test", "gps_block": "g"},
    ]
    with pytest.raises(ValueError, match="GPS-block leakage"):
        validate_recipe_identities(rows)


def test_materializer_rejects_internal_only_validation_as_external_evidence(tmp_path) -> None:
    recipes = tmp_path / "recipes.jsonl"
    backgrounds = tmp_path / "backgrounds.jsonl"
    validation = tmp_path / "validation.json"
    recipes.write_text(
        json.dumps(
            {
                "injection_id": "i1",
                "waveform_id": "w1",
                "split": "val",
                "gps_block": "g1",
                "background_window_id": "b1",
            }
        )
        + "\n"
    )
    backgrounds.write_text(json.dumps({"window_id": "b1"}) + "\n")
    validation.write_text(json.dumps({"passed": True, "validation_scope": "internal_smoke"}))
    with pytest.raises(ValueError, match="external_reference_waveform_equivalence"):
        run_injection_materialization(
            recipes,
            backgrounds,
            tmp_path / "output",
            backend_validation_report=validation,
        )


def test_signal_only_materialization_references_hashed_background(tmp_path) -> None:
    source = tmp_path / "strain.hdf5"
    with h5py.File(source, "w") as handle:
        meta = handle.create_group("meta")
        meta.create_dataset("GPSstart", data=100)
        strain = handle.create_group("strain").create_dataset(
            "Strain", data=np.arange(32, dtype=np.float64)
        )
        strain.attrs["Xspacing"] = 0.25

    class FakeBackend:
        def generate(self, recipe, ifos, sample_rate):
            assert ifos == ["H1"]
            assert sample_rate == 4
            return {"H1": (103.0, np.asarray([1.0, 2.0]))}, {"H1": {"fake": True}}

    recipe = {
        "injection_id": "i1",
        "background_window_id": "b1",
        "gps_block": "g1",
        "split": "val",
    }
    background = {
        "window_id": "b1",
        "gps_block": "g1",
        "split": "val",
        "gps_start": 103.0,
        "duration": 2.0,
        "ifos": ["H1"],
        "source_files": {"H1": {"path": str(source), "sha256": file_sha256(source)}},
    }
    output = tmp_path / "injection.npz"
    report = materialize_recipe(
        recipe, background, FakeBackend(), 4, output, context_duration=4.0
    )
    with np.load(output, allow_pickle=False) as arrays:
        assert "signal" in arrays
        assert "noise" not in arrays
        assert "strain" not in arrays
        assert arrays["signal"].shape == (1, 16)
    assert report["storage_mode"] == "signal_only"
    assert report["background_source_files"]["H1"]["sha256"] == file_sha256(source)
    loaded = load_materialized_context(report)
    assert loaded["noise"].shape == (1, 16)
    assert loaded["signal"].dtype == np.float64
    assert loaded["mixture"] == pytest.approx(loaded["noise"] + loaded["signal"])

    scaled_output = tmp_path / "injection-scaled.npz"
    scaled_report = materialize_recipe(
        recipe,
        background,
        FakeBackend(),
        4,
        scaled_output,
        context_duration=4.0,
        storage_mode="signal_scaled_float16",
    )
    with np.load(scaled_output, allow_pickle=False) as arrays:
        assert "signal" not in arrays
        assert arrays["signal_scaled"].dtype == np.float16
        assert arrays["signal_peak_scale"].dtype == np.float64
    scaled_loaded = load_materialized_context(scaled_report)
    assert scaled_loaded["signal"].dtype == np.float64
    assert scaled_loaded["signal"] == pytest.approx(loaded["signal"], rel=1e-3)
    assert scaled_report["signal_reconstruction"]["relative_l2_error"] <= 1e-3
