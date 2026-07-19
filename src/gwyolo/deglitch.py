from __future__ import annotations

import os
import tempfile
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_write_json, file_sha256


def _frame_stft(signal: np.ndarray, window_size: int, hop_size: int) -> tuple[np.ndarray, int]:
    if signal.ndim != 1:
        raise ValueError("STFT input must be one-dimensional")
    if window_size <= 1 or hop_size <= 0 or hop_size > window_size:
        raise ValueError("invalid STFT window/hop")
    frame_count = max(1, int(np.ceil(max(signal.size - window_size, 0) / hop_size)) + 1)
    padded_size = (frame_count - 1) * hop_size + window_size
    padded = np.pad(signal.astype(np.float64), (0, padded_size - signal.size))
    window = np.hamming(window_size)
    frames = np.stack(
        [padded[index * hop_size : index * hop_size + window_size] * window for index in range(frame_count)]
    )
    return np.fft.rfft(frames, axis=1).T, signal.size


def _inverse_stft(
    spectrum: np.ndarray, original_size: int, window_size: int, hop_size: int
) -> np.ndarray:
    frames = np.fft.irfft(spectrum.T, n=window_size, axis=1)
    output_size = (frames.shape[0] - 1) * hop_size + window_size
    output = np.zeros(output_size, dtype=np.float64)
    normalization = np.zeros(output_size, dtype=np.float64)
    window = np.hamming(window_size)
    for index, frame in enumerate(frames):
        start = index * hop_size
        output[start : start + window_size] += frame * window
        normalization[start : start + window_size] += window**2
    valid = normalization > 1e-10
    output[valid] /= normalization[valid]
    return output[:original_size]


def _resize_mask(mask: np.ndarray, frequency_bins: int, time_bins: int) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be frequency x time")
    source_frequency = np.linspace(0.0, 1.0, mask.shape[0])
    target_frequency = np.linspace(0.0, 1.0, frequency_bins)
    frequency_resized = np.stack(
        [np.interp(target_frequency, source_frequency, mask[:, column]) for column in range(mask.shape[1])],
        axis=1,
    )
    source_time = np.linspace(0.0, 1.0, mask.shape[1])
    target_time = np.linspace(0.0, 1.0, time_bins)
    return np.stack(
        [np.interp(target_time, source_time, row) for row in frequency_resized], axis=0
    )


def mask_deglitch(
    strain: np.ndarray,
    sample_rate: int,
    chirp_probability: np.ndarray,
    glitch_probability: np.ndarray,
    strength: float = 0.9,
    window_size: int = 256,
    hop_size: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Suppress glitch-like STFT coefficients while explicitly protecting chirp pixels."""

    if strain.ndim != 2:
        raise ValueError("strain must have shape [ifo, time]")
    if chirp_probability.shape != glitch_probability.shape or chirp_probability.ndim != 4:
        raise ValueError("probabilities must share [ifo, q, frequency, time] shape")
    if chirp_probability.shape[0] != strain.shape[0]:
        raise ValueError("probability IFO dimension must match strain")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between zero and one")
    cleaned = []
    removed_fractions = []
    gains = []
    for ifo_index, signal in enumerate(strain):
        spectrum, original_size = _frame_stft(signal, window_size, hop_size)
        chirp = np.max(chirp_probability[ifo_index], axis=0)
        glitch = np.max(glitch_probability[ifo_index], axis=0)
        suppress = np.clip(glitch, 0.0, 1.0) * (1.0 - np.clip(chirp, 0.0, 1.0))
        suppress = _resize_mask(suppress, spectrum.shape[0], spectrum.shape[1])
        gain = np.clip(1.0 - strength * suppress, 0.0, 1.0)
        modified = spectrum * gain
        cleaned_signal = _inverse_stft(modified, original_size, window_size, hop_size)
        energy_before = float(np.sum(np.abs(spectrum) ** 2))
        energy_after = float(np.sum(np.abs(modified) ** 2))
        removed_fraction = (
            (energy_before - energy_after) / energy_before if energy_before > 0 else 0.0
        )
        removed_fractions.append(float(np.clip(removed_fraction, 0.0, 1.0)))
        gains.append(float(np.mean(gain)))
        cleaned.append(cleaned_signal)
    result = np.stack(cleaned).astype(np.float32)
    return result, {
        "sample_rate": sample_rate,
        "strength": strength,
        "window_size": window_size,
        "hop_size": hop_size,
        "removed_tf_energy_fraction_by_ifo": removed_fractions,
        "mean_gain_by_ifo": gains,
    }


def deglitch_metrics(
    mixture: np.ndarray,
    cleaned: np.ndarray,
    clean_reference: np.ndarray,
    chirp_reference: np.ndarray,
) -> dict[str, float]:
    if not (mixture.shape == cleaned.shape == clean_reference.shape == chirp_reference.shape):
        raise ValueError("all deglitch metric arrays must share shape")

    def mse(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))

    def projection(value: np.ndarray, target: np.ndarray) -> float:
        denominator = float(np.linalg.norm(target))
        return float(np.sum(value * target) / denominator) if denominator > 0 else 0.0

    before_mse = mse(mixture, clean_reference)
    after_mse = mse(cleaned, clean_reference)
    before_projection = projection(mixture, chirp_reference)
    after_projection = projection(cleaned, chirp_reference)
    return {
        "mse_to_clean_before": before_mse,
        "mse_to_clean_after": after_mse,
        "mse_reduction_fraction": (before_mse - after_mse) / before_mse if before_mse else 0.0,
        "chirp_projection_before": before_projection,
        "chirp_projection_after": after_projection,
        "chirp_projection_retention": (
            after_projection / before_projection if before_projection else 1.0
        ),
        "waveform_change_rms": float(np.sqrt(np.mean((cleaned - mixture) ** 2))),
    }


def run_oracle_deglitch(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    strength: float = 0.9,
) -> dict[str, Any]:
    with np.load(input_path, allow_pickle=False) as arrays:
        required = {"strain", "clean_strain", "chirp_strain", "chirp_mask", "glitch_mask", "sample_rate"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"Input NPZ is missing: {missing}")
        mixture = arrays["strain"].astype(np.float32)
        clean_reference = arrays["clean_strain"].astype(np.float32)
        chirp_reference = arrays["chirp_strain"].astype(np.float32)
        cleaned, suppression = mask_deglitch(
            mixture,
            int(arrays["sample_rate"]),
            arrays["chirp_mask"].astype(np.float32),
            arrays["glitch_mask"].astype(np.float32),
            strength,
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npz", dir=target.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, cleaned_strain=cleaned)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    report = {
        "status": "oracle_mask_upper_bound",
        "scientific_claim_allowed": False,
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "suppression": suppression,
        "metrics": deglitch_metrics(mixture, cleaned, clean_reference, chirp_reference),
    }
    atomic_write_json(report_path, report)
    return report


def summarize_deglitch_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one deglitch row is required")
    output: dict[str, Any] = {}
    for scene_type in sorted({str(row["scene_type"]) for row in rows}):
        selected = [row for row in rows if row["scene_type"] == scene_type]
        summary = {"scenes": len(selected)}
        for key in (
            "mse_reduction_fraction",
            "chirp_projection_retention",
            "waveform_change_rms",
        ):
            values = np.asarray([float(row["metrics"][key]) for row in selected])
            summary[key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
        output[scene_type] = summary
    return output


def run_oracle_deglitch_benchmark(
    factory_report_path: str | Path,
    output_path: str | Path,
    strength: float = 0.9,
) -> dict[str, Any]:
    with Path(factory_report_path).open("r", encoding="utf-8") as handle:
        factory_report = json.load(handle)
    rows = []
    for record in factory_report["records"]:
        if record["scene_type"] not in {"overlap", "chirp_only"}:
            continue
        with np.load(record["path"], allow_pickle=False) as arrays:
            mixture = arrays["strain"].astype(np.float32)
            cleaned, suppression = mask_deglitch(
                mixture,
                int(arrays["sample_rate"]),
                arrays["chirp_mask"].astype(np.float32),
                arrays["glitch_mask"].astype(np.float32),
                strength,
            )
            metrics = deglitch_metrics(
                mixture,
                cleaned,
                arrays["clean_strain"].astype(np.float32),
                arrays["chirp_strain"].astype(np.float32),
            )
        rows.append(
            {
                "scene_id": record["scene_id"],
                "scene_type": record["scene_type"],
                "metrics": metrics,
                "suppression": suppression,
            }
        )
    report = {
        "status": "oracle_mask_upper_bound",
        "scientific_claim_allowed": False,
        "factory_report_path": str(factory_report_path),
        "factory_report_sha256": file_sha256(factory_report_path),
        "strength": strength,
        "summary": summarize_deglitch_rows(rows),
        "rows": rows,
    }
    atomic_write_json(output_path, report)
    return report
