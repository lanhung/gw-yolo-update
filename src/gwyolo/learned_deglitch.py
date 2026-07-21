from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .deglitch import mask_deglitch
from .gwosc import _fft_downsample, read_hdf5_segment
from .io import atomic_write_json, atomic_write_text, file_sha256
from .injection_score import apply_analysis_override
from .runtime import execution_provenance
from .waveforms import _atomic_save_npz, load_materialized_context


def signal_retention_metrics(
    mixture: np.ndarray,
    cleaned: np.ndarray,
    noise: np.ndarray,
    signal: np.ndarray,
) -> dict[str, Any]:
    if not (mixture.shape == cleaned.shape == noise.shape == signal.shape):
        raise ValueError("mixture, cleaned, noise and signal must share shape")
    if mixture.ndim != 2:
        raise ValueError("deglitch retention arrays must have [IFO, time] shape")

    def retention(indices: Any) -> float | None:
        expected = signal[indices].astype(np.float64)
        denominator = float(np.sum(expected**2))
        if denominator <= 0:
            return None
        residual = (cleaned[indices] - noise[indices]).astype(np.float64)
        return float(np.sum(residual * expected) / denominator)

    change = cleaned.astype(np.float64) - mixture.astype(np.float64)
    error = cleaned.astype(np.float64) - noise.astype(np.float64) - signal.astype(np.float64)
    return {
        "network_signal_projection_retention": retention(np.s_[:]),
        "signal_projection_retention_by_ifo": [retention(index) for index in range(signal.shape[0])],
        "waveform_change_rms": float(np.sqrt(np.mean(change**2))),
        "postclean_signal_error_rms": float(np.sqrt(np.mean(error**2))),
        "injected_signal_rms": float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_family"])].append(row)
    output = {}
    for family, selected in sorted(grouped.items()):
        output[family] = {"injections": len(selected)}
        for key in (
            "network_signal_projection_retention",
            "waveform_change_rms",
            "postclean_signal_error_rms",
        ):
            values = np.asarray([float(row["metrics"][key]) for row in selected])
            output[family][key] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
    return output


def run_learned_deglitch(
    materialized_manifest: str | Path,
    scored_manifest: str | Path,
    output_dir: str | Path,
    strength: float = 0.9,
) -> dict[str, Any]:
    with Path(materialized_manifest).open("r", encoding="utf-8") as handle:
        materialized = [json.loads(line) for line in handle if line.strip()]
    with Path(scored_manifest).open("r", encoding="utf-8") as handle:
        scores = [json.loads(line) for line in handle if line.strip()]
    if not materialized or not scores:
        raise ValueError("materialized and scored manifests must be non-empty")
    score_by_id = {str(row["injection_id"]): row for row in scores}
    if len(score_by_id) != len(scores):
        raise ValueError("scored manifest contains duplicate injection IDs")
    missing = sorted(
        str(row["injection_id"])
        for row in materialized
        if str(row["injection_id"]) not in score_by_id
    )
    if missing:
        raise ValueError(f"Materialized injections lack scored masks: {missing[:10]}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    verified_background_hashes: dict[str, str] = {}
    result_rows = []
    for row in materialized:
        scored = score_by_id[str(row["injection_id"])]
        probability_path = Path(scored["probability_path"])
        if file_sha256(probability_path) != str(scored["probability_sha256"]):
            raise ValueError(f"Probability hash mismatch for {row['injection_id']}")
        with np.load(probability_path, allow_pickle=False) as probabilities:
            probability_ifos = [str(value) for value in probabilities["ifos"].tolist()]
            chirp = np.asarray(probabilities["chirp_probability"], dtype=np.float32)
            glitch = np.asarray(probabilities["glitch_probability"], dtype=np.float32)
        context = load_materialized_context(row, verified_background_hashes)
        context, input_override = apply_analysis_override(row, context)
        start = int(context["analysis_start_index"])
        stop = int(context["analysis_stop_index"])
        ifos = list(context["ifos"])
        indices = [probability_ifos.index(ifo) for ifo in ifos]
        mixture = np.asarray(context["mixture"][:, start:stop], dtype=np.float64)
        noise = np.asarray(context["noise"][:, start:stop], dtype=np.float64)
        signal = np.asarray(context["signal"][:, start:stop], dtype=np.float64)
        cleaned, suppression = mask_deglitch(
            mixture,
            int(context["sample_rate"]),
            chirp[indices],
            glitch[indices],
            strength,
        )
        metrics = signal_retention_metrics(mixture, cleaned, noise, signal)
        cleaned_path = output / "arrays" / f"{row['injection_id']}.npz"
        _atomic_save_npz(
            cleaned_path,
            analysis_strain=cleaned,
            cleaned_strain=cleaned,
            ifos=np.asarray(ifos),
            sample_rate=np.asarray(context["sample_rate"], dtype=np.int64),
            analysis_gps_start=np.asarray(context["analysis_gps_start"], dtype=np.float64),
        )
        result_rows.append(
            {
                **row,
                "cleaned_path": str(cleaned_path),
                "cleaned_sha256": file_sha256(cleaned_path),
                "analysis_override_path": str(cleaned_path),
                "analysis_override_sha256": file_sha256(cleaned_path),
                "analysis_override_kind": "mask_conditioned",
                "input_analysis_override_sha256": input_override.get(
                    "analysis_override_sha256"
                ),
                "input_analysis_override_kind": input_override.get(
                    "analysis_override_kind"
                ),
                "probability_path": str(probability_path),
                "probability_sha256": scored["probability_sha256"],
                "deglitch_strength": strength,
                "deglitch_algorithm": "hamming_stft_overlap_add",
                "metrics": metrics,
                "suppression": suppression,
            }
        )
    manifest_path = output / "learned_deglitch.jsonl"
    atomic_write_text(
        manifest_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows),
    )
    report = {
        "status": "learned_mask_signal_retention_diagnostic",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "real background has no counterfactual glitch-free reference; fixed-FAR paired search "
            "and targeted glitch-overlap injections remain required"
        ),
        "strength": strength,
        "injections": len(result_rows),
        "materialized_manifest_sha256": file_sha256(materialized_manifest),
        "scored_manifest_sha256": file_sha256(scored_manifest),
        "summary": _summarize(result_rows),
        "signal_retention_interpretation_valid": all(
            row.get("input_analysis_override_kind")
            in {None, "clean", "clean_reference"}
            for row in result_rows
        ),
        "unique_injection_ids": len({row["injection_id"] for row in result_rows}),
        "unique_waveform_ids": len({row["waveform_id"] for row in result_rows}),
        "unique_gps_blocks": len({row["gps_block"] for row in result_rows}),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        **execution_provenance(),
    }
    atomic_write_json(output / "learned_deglitch_report.json", report)
    return report


def run_learned_background_deglitch(
    background_manifest: str | Path,
    scored_manifest: str | Path,
    output_dir: str | Path,
    strength: float = 0.9,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
    required_split: str | None = None,
) -> dict[str, Any]:
    """Write central cleaned overrides; trigger scoring retains the original PSD context."""

    if not 0 <= strength <= 1 or target_sample_rate <= 0 or context_duration <= 0:
        raise ValueError("Invalid learned background deglitch settings")
    with Path(background_manifest).open("r", encoding="utf-8") as handle:
        background = [json.loads(line) for line in handle if line.strip()]
    with Path(scored_manifest).open("r", encoding="utf-8") as handle:
        scores = [json.loads(line) for line in handle if line.strip()]
    if not background or not scores:
        raise ValueError("Background and scored manifests must be non-empty")
    observed_splits = sorted({str(row.get("split")) for row in background})
    if required_split is not None and observed_splits != [required_split]:
        raise ValueError(
            f"Background deglitch required split {required_split}, observed {observed_splits}"
        )
    score_by_id = {str(row["window_id"]): row for row in scores}
    if len(score_by_id) != len(scores):
        raise ValueError("Scored background manifest contains duplicate window IDs")
    background_ids = [str(row["window_id"]) for row in background]
    if len(background_ids) != len(set(background_ids)):
        raise ValueError("Background manifest contains duplicate window IDs")
    missing = sorted(set(background_ids) - set(score_by_id))
    if missing:
        raise ValueError(f"Background windows lack scored masks: {missing[:10]}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    verified_source_hashes: dict[str, str] = {}
    result_rows = []
    for row in background:
        scored = score_by_id[str(row["window_id"])]
        probability_path = Path(scored["probability_path"])
        if file_sha256(probability_path) != str(scored["probability_sha256"]):
            raise ValueError(f"Probability hash mismatch for {row['window_id']}")
        with np.load(probability_path, allow_pickle=False) as probabilities:
            probability_ifos = tuple(str(value) for value in probabilities["ifos"].tolist())
            chirp = np.asarray(probabilities["chirp_probability"], dtype=np.float32)
            glitch = np.asarray(probabilities["glitch_probability"], dtype=np.float32)
        if probability_ifos != model_ifos:
            raise ValueError("Background probability detector order differs from model_ifos")
        center = (float(row["gps_start"]) + float(row["gps_end"])) / 2.0
        output_samples = int(round(float(row["duration"]) * target_sample_rate))
        raw = np.zeros((len(model_ifos), output_samples), dtype=np.float64)
        source_ifos = [str(value) for value in row["ifos"]]
        for ifo in source_ifos:
            if ifo not in model_ifos:
                raise ValueError(f"Background source detector {ifo} is not configured")
            source = row["source_files"][ifo]
            source_path = str(source["path"])
            observed_hash = verified_source_hashes.get(source_path)
            if observed_hash is None:
                observed_hash = file_sha256(source_path)
                verified_source_hashes[source_path] = observed_hash
            if observed_hash != str(source["sha256"]):
                raise ValueError(f"Background source hash mismatch for {ifo}")
            segment = read_hdf5_segment(source_path, center, context_duration)
            values = _fft_downsample(
                np.asarray(segment["strain"], dtype=np.float64),
                int(segment["sample_rate"]),
                target_sample_rate,
            )
            start = values.size // 2 - output_samples // 2
            selected = values[start : start + output_samples]
            if selected.shape != (output_samples,) or not np.isfinite(selected).all():
                raise ValueError(f"Background analysis crop is invalid for {ifo}")
            raw[model_ifos.index(ifo)] = selected
        cleaned, suppression = mask_deglitch(
            raw, target_sample_rate, chirp, glitch, strength
        )
        cleaned_path = output / "arrays" / f"{row['window_id']}.npz"
        _atomic_save_npz(
            cleaned_path,
            analysis_strain=cleaned,
            cleaned_strain=cleaned,
            ifos=np.asarray(model_ifos),
            sample_rate=np.asarray(target_sample_rate, dtype=np.int64),
            analysis_gps_start=np.asarray(row["gps_start"], dtype=np.float64),
        )
        result_rows.append(
            {
                **row,
                "analysis_override_path": str(cleaned_path),
                "analysis_override_sha256": file_sha256(cleaned_path),
                "analysis_override_kind": "mask_conditioned",
                "probability_sha256": scored["probability_sha256"],
                "suppression": suppression,
            }
        )
    manifest = output / "learned_background_deglitch.jsonl"
    atomic_write_text(
        manifest, "".join(json.dumps(row, sort_keys=True) + "\n" for row in result_rows)
    )
    removed = np.asarray(
        [
            value
            for row in result_rows
            for value in row["suppression"]["removed_tf_energy_fraction_by_ifo"]
        ],
        dtype=np.float64,
    )
    report = {
        "status": "learned_mask_background_analysis_overrides",
        "scientific_claim_allowed": False,
        "scientific_blocker": (
            "cleaned background must be rescored and compared at independently calibrated "
            "validation thresholds before continuous time-slide evaluation"
        ),
        "strength": strength,
        "model_ifos": list(model_ifos),
        "target_sample_rate": target_sample_rate,
        "context_duration": context_duration,
        "required_split": required_split,
        "observed_splits": observed_splits,
        "windows": len(result_rows),
        "unique_gps_blocks": len({str(row["gps_block"]) for row in result_rows}),
        "removed_tf_energy_fraction": {
            "mean": float(np.mean(removed)),
            "median": float(np.median(removed)),
            "p95": float(np.percentile(removed, 95)),
        },
        "background_manifest_sha256": file_sha256(background_manifest),
        "scored_manifest_sha256": file_sha256(scored_manifest),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        **execution_provenance(),
    }
    atomic_write_json(output / "learned_background_deglitch_report.json", report)
    return report
