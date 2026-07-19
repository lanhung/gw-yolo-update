from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .deglitch import mask_deglitch
from .io import atomic_write_json, atomic_write_text, file_sha256
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
            cleaned_strain=cleaned,
            ifos=np.asarray(ifos),
            sample_rate=np.asarray(context["sample_rate"], dtype=np.int64),
            analysis_gps_start=np.asarray(context["analysis_gps_start"], dtype=np.float64),
        )
        result_rows.append(
            {
                "injection_id": row["injection_id"],
                "source_family": row["source_family"],
                "gps_block": row["gps_block"],
                "cleaned_path": str(cleaned_path),
                "cleaned_sha256": file_sha256(cleaned_path),
                "probability_sha256": scored["probability_sha256"],
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
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_write_json(output / "learned_deglitch_report.json", report)
    return report
