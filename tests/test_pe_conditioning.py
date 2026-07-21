from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from gwyolo.pe_conditioning import (
    materialize_native_pe_conditioning,
    read_dingo_event_settings,
)
from gwyolo.pe_inputs import materialize_common_pe_inputs
from test_pe_inputs import _paired_manifests


def _common_sources(tmp_path: Path) -> Path:
    clean, contaminated, masked, prior, model, policy = _paired_manifests(tmp_path)
    report = materialize_common_pe_inputs(
        clean,
        contaminated,
        masked,
        prior,
        model,
        policy,
        tmp_path / "common",
        "val",
        source_sample_rate_hz=16,
        source_duration_seconds=4.0,
        source_post_trigger_seconds=1.0,
        analysis_high_frequency_hz=4.0,
        asd_segment_seconds=1.0,
        asd_stride_seconds=0.5,
        asd_guard_seconds=0.5,
    )
    return Path(report["manifest_path"])


def test_dingo_native_conditioning_has_official_event_dataset_shape(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    source_manifest = _common_sources(tmp_path)
    config = tmp_path / "dingo.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "backend": "DINGO",
                "ifos": ["H1", "L1"],
                "source_sample_rate_hz": 16,
                "source_duration_seconds": 4,
                "source_post_trigger_seconds": 1,
                "window": {"type": "tukey", "roll_off_seconds": 0.5},
                "frequency_domain": {
                    "minimum_frequency_hz": 1,
                    "maximum_frequency_hz": 4,
                    "delta_frequency_hz": 0.25,
                    "fourier_convention": "numpy_rfft_times_delta_t",
                    "time_translation": "exp_minus_2pi_i_f_post_trigger",
                },
                "asd": {
                    "source": "common_source_artifact",
                    "condition_invariant_required": True,
                    "below_minimum_frequency_value": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "dingo"
    report = materialize_native_pe_conditioning(source_manifest, config, output, "val")
    assert report["backend"] == "DINGO"
    assert report["rows"] == 3
    rows = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert len({row["common_asd_sha256"] for row in rows}) == 1
    import h5py

    for row in rows:
        with h5py.File(row["native_conditioning_path"], "r") as handle:
            assert handle.attrs["dataset_type"] == "event_dataset"
            assert handle["data/waveform/H1"].shape == (17,)
            assert handle["data/asds/L1"].shape == (17,)
            assert np.all(handle["data/asds/H1"][:4] == 1.0)
        settings = read_dingo_event_settings(row["native_conditioning_path"])
        assert settings["time_buffer"] == 1
        assert settings["detectors"] == ["H1", "L1"]

    resumed = materialize_native_pe_conditioning(source_manifest, config, output, "val")
    assert resumed["manifest_sha256"] == report["manifest_sha256"]


def test_amplfi_native_conditioning_downsamples_and_retains_common_asd(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    source_manifest = _common_sources(tmp_path)
    config = tmp_path / "amplfi.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "backend": "AMPLFI",
                "ifos": ["H1", "L1"],
                "source_sample_rate_hz": 16,
                "source_duration_seconds": 4,
                "source_post_trigger_seconds": 1,
                "native_sample_rate_hz": 8,
                "native_kernel_seconds": 1,
                "native_whitening_duration_seconds": 1,
                "native_highpass_hz": 1,
                "native_right_pad_seconds": 0.25,
                "resampling": {
                    "method": "scipy_signal_resample_poly",
                    "window": ["kaiser", 8.6],
                },
                "asd": {
                    "source": "common_source_artifact",
                    "condition_invariant_required": True,
                    "runtime_whitening_must_not_reestimate_psd": True,
                },
            }
        ),
        encoding="utf-8",
    )
    report = materialize_native_pe_conditioning(
        source_manifest, config, tmp_path / "amplfi", "val"
    )
    rows = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert report["backend"] == "AMPLFI"
    assert report["condition_invariant_common_asd"] is True
    for row in rows:
        with h5py.File(row["native_conditioning_path"], "r") as handle:
            assert handle["strain"].shape == (2, 32)
            assert handle["asd"].shape == (2, 17)
            assert handle.attrs["common_asd_sha256"] == row["common_asd_sha256"]
        assert row["runtime_whitening_must_not_reestimate_psd"] is True
        assert row["native_right_pad_seconds"] == 0.25


def test_amplfi_native_conditioning_rejects_changed_event_position(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    source_manifest = _common_sources(tmp_path)
    config = tmp_path / "amplfi-invalid.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "backend": "AMPLFI",
                "ifos": ["H1", "L1"],
                "source_sample_rate_hz": 16,
                "source_duration_seconds": 4,
                "source_post_trigger_seconds": 2,
                "native_sample_rate_hz": 8,
                "native_kernel_seconds": 1,
                "native_whitening_duration_seconds": 1,
                "native_highpass_hz": 1,
                "native_right_pad_seconds": 0.25,
                "resampling": {
                    "method": "scipy_signal_resample_poly",
                    "window": ["kaiser", 8.6],
                },
                "asd": {
                    "source": "common_source_artifact",
                    "condition_invariant_required": True,
                    "runtime_whitening_must_not_reestimate_psd": True,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="post-trigger contract mismatch"):
        materialize_native_pe_conditioning(
            source_manifest, config, tmp_path / "amplfi-invalid", "val"
        )
