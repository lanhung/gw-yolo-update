from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwyolo.io import file_sha256
from gwyolo.pe_inputs import materialize_common_pe_inputs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _paired_manifests(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    materialized = tmp_path / "materialized.npz"
    rate = 8
    samples = 64
    signal = np.zeros((2, samples), dtype=np.float64)
    signal[:, 28:36] = np.asarray([[1.0], [0.5]])
    noise = np.vstack(
        [np.linspace(-0.2, 0.2, samples), np.linspace(0.1, -0.1, samples)]
    )
    np.savez(
        materialized,
        signal=signal,
        noise=noise,
        strain=noise + signal,
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(rate),
        context_gps_start=np.asarray(96.0),
        analysis_gps_start=np.asarray(99.0),
        analysis_start_index=np.asarray(24),
        analysis_stop_index=np.asarray(40),
    )
    base = {
        "injection_id": "injection-one",
        "waveform_id": "waveform-one",
        "source_family": "BBH",
        "split": "val",
        "gps_block": "injection-block",
        "gps_time": 100.0,
        "ifos": ["H1", "L1"],
        "mass_1_detector_msun": 40.0,
        "mass_2_detector_msun": 20.0,
        "luminosity_distance_mpc": 1000.0,
        "right_ascension": 1.0,
        "declination": 0.2,
        "polarization": 0.5,
        "inclination": 0.7,
        "coalescence_phase": 0.1,
        "waveform_approximant": "IMRPhenomXAS",
        "materialized_path": str(materialized),
        "materialized_sha256": file_sha256(materialized),
    }
    clean = tmp_path / "clean.jsonl"
    _write_jsonl(clean, [base])

    contaminated_array = tmp_path / "contaminated.npz"
    contaminated = noise[:, 24:40] + signal[:, 24:40] + 2.0
    np.savez(
        contaminated_array,
        analysis_strain=contaminated,
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(rate),
        analysis_gps_start=np.asarray(99.0),
    )
    contaminated_row = {
        **base,
        "analysis_override_path": str(contaminated_array),
        "analysis_override_sha256": file_sha256(contaminated_array),
        "analysis_override_kind": "real_glitch_contaminated",
        "glitch_id": "glitch-one",
        "glitch_gps_block": "glitch-block",
        "glitch_label": "Blip",
    }
    contaminated_manifest = tmp_path / "contaminated.jsonl"
    _write_jsonl(contaminated_manifest, [contaminated_row])

    masked_array = tmp_path / "masked.npz"
    np.savez(
        masked_array,
        analysis_strain=contaminated - 1.5,
        ifos=np.asarray(["H1", "L1"]),
        sample_rate=np.asarray(rate),
        analysis_gps_start=np.asarray(99.0),
    )
    probability = tmp_path / "probability.npz"
    np.savez(probability, chirp_probability=np.zeros((2, 2)), glitch_probability=np.ones((2, 2)))
    masked_row = {
        **contaminated_row,
        "analysis_override_path": str(masked_array),
        "analysis_override_sha256": file_sha256(masked_array),
        "analysis_override_kind": "mask_conditioned",
        "input_analysis_override_sha256": file_sha256(contaminated_array),
        "input_analysis_override_kind": "real_glitch_contaminated",
        "probability_path": str(probability),
        "probability_sha256": file_sha256(probability),
        "deglitch_strength": 0.9,
        "deglitch_algorithm": "hamming_stft_overlap_add",
    }
    masked_manifest = tmp_path / "masked.jsonl"
    _write_jsonl(masked_manifest, [masked_row])

    prior = tmp_path / "prior.yaml"
    prior.write_text(
        """population: BBH
distributions:
  chirp_mass: {minimum: 15, maximum: 100}
  mass_ratio: {minimum: 0.125, maximum: 0.999}
  luminosity_distance: {minimum: 100, maximum: 3100}
  theta_jn: {minimum: 0, maximum: 3.141592653589793}
  ra: {minimum: 0, maximum: 6.283185307179586}
  dec: {minimum: -1.5707963267948966, maximum: 1.5707963267948966}
  psi: {minimum: 0, maximum: 3.141592653589793}
""",
        encoding="utf-8",
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "mode: cleaned_strain\n"
        "algorithm: hamming_stft_overlap_add\n"
        "suppression_strength: 0.9\n"
        "auxiliary_channel_veto: false\n",
        encoding="utf-8",
    )
    return clean, contaminated_manifest, masked_manifest, prior, model, policy


def test_common_pe_inputs_are_paired_backend_neutral_and_resumable(tmp_path: Path) -> None:
    clean, contaminated, masked, prior, model, policy = _paired_manifests(tmp_path)
    output = tmp_path / "output"
    kwargs = dict(
        clean_manifest=clean,
        contaminated_manifest=contaminated,
        mask_conditioned_manifest=masked,
        common_prior_path=prior,
        mask_model_path=model,
        mask_policy_path=policy,
        output_dir=output,
        required_split="val",
        source_sample_rate_hz=16,
        source_duration_seconds=4.0,
        source_post_trigger_seconds=1.0,
        analysis_high_frequency_hz=4.0,
        asd_segment_seconds=1.0,
        asd_stride_seconds=0.5,
        asd_guard_seconds=0.5,
    )
    report = materialize_common_pe_inputs(**kwargs)
    assert report["paired_injections"] == 1
    assert report["rows"] == 3
    assert report["bandlimited_upsampling_used"] is True
    assert report["input_population_matches_analysis_prior_distribution"] is False

    rows = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert {row["condition"] for row in rows} == {
        "clean",
        "contaminated",
        "mask_conditioned",
    }
    assert len({row["analysis_input_sha256"] for row in rows}) == 3
    for row in rows:
        assert row["input_ifos"] == ["H1", "L1"]
        assert row["input_sample_rate_hz"] == 16
        with np.load(row["analysis_input_path"], allow_pickle=False) as arrays:
            assert arrays["strain"].shape == (2, 64)
            assert arrays["strain"].dtype == np.float32
            assert arrays["asd"].shape == (2, 17)
            assert arrays["asd_frequencies"][-1] == 4.0
            assert np.all(arrays["asd"] > 0)
    assert len({row["common_asd_sha256"] for row in rows}) == 1
    masked_row = next(row for row in rows if row["condition"] == "mask_conditioned")
    assert masked_row["mask_artifact_sha256"] == file_sha256(
        masked_row["mask_artifact_path"]
    )

    resumed = materialize_common_pe_inputs(**kwargs)
    assert resumed["manifest_sha256"] == report["manifest_sha256"]


def test_common_pe_inputs_reject_mismatched_mask_lineage(tmp_path: Path) -> None:
    clean, contaminated, masked, prior, model, policy = _paired_manifests(tmp_path)
    row = json.loads(masked.read_text().strip())
    row["input_analysis_override_sha256"] = "0" * 64
    _write_jsonl(masked, [row])
    with pytest.raises(ValueError, match="not derived"):
        materialize_common_pe_inputs(
            clean,
            contaminated,
            masked,
            prior,
            model,
            policy,
            tmp_path / "bad-output",
            "val",
            source_sample_rate_hz=16,
            source_duration_seconds=4.0,
            source_post_trigger_seconds=1.0,
            analysis_high_frequency_hz=4.0,
            asd_segment_seconds=1.0,
            asd_stride_seconds=0.5,
            asd_guard_seconds=0.5,
        )
