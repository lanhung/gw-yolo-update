from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .factory import _normalize_power, multiresolution_power
from .gwosc import _fft_downsample, _whiten, read_hdf5_segment
from .io import atomic_write_json, atomic_write_text, file_sha256, load_yaml


def network_ranking(
    chirp_scores: dict[str, float],
    glitch_scores: dict[str, float],
    valid_ifos: list[str],
) -> dict[str, Any]:
    if not valid_ifos:
        raise ValueError("at least one valid IFO is required")
    missing = [ifo for ifo in valid_ifos if ifo not in chirp_scores or ifo not in glitch_scores]
    if missing:
        raise ValueError(f"Missing scores for valid IFOs: {missing}")
    ordered_chirp = sorted((float(chirp_scores[ifo]) for ifo in valid_ifos), reverse=True)
    coherent_score = ordered_chirp[1] if len(ordered_chirp) >= 2 else ordered_chirp[0]
    maximum_glitch = max(float(glitch_scores[ifo]) for ifo in valid_ifos)
    return {
        "ranking_score": coherent_score,
        "ranking_definition": "second-highest valid-IFO chirp maximum; single-IFO maximum if needed",
        "network_mode": "coincident" if len(valid_ifos) >= 2 else "single_ifo_diagnostic",
        "valid_ifos": valid_ifos,
        "maximum_chirp_score": ordered_chirp[0],
        "coherent_chirp_score": coherent_score,
        "maximum_glitch_score": maximum_glitch,
        "chirp_glitch_margin": coherent_score - maximum_glitch,
    }


def _window_strain(
    row: dict[str, Any],
    model_ifos: tuple[str, ...],
    target_sample_rate: int,
    context_duration: float,
) -> tuple[np.ndarray, list[str]]:
    center = (float(row["gps_start"]) + float(row["gps_end"])) / 2.0
    window_duration = float(row["duration"])
    valid_ifos = [str(item) for item in row["ifos"]]
    context_by_ifo = {}
    for ifo in valid_ifos:
        source = row["source_files"][ifo]["path"]
        segment = read_hdf5_segment(source, center, context_duration)
        context_by_ifo[ifo] = _fft_downsample(
            segment["strain"], segment["sample_rate"], target_sample_rate
        )
    output_samples = int(round(window_duration * target_sample_rate))
    strains = []
    for ifo in model_ifos:
        if ifo not in context_by_ifo:
            strains.append(np.zeros(output_samples, dtype=np.float32))
            continue
        whitened = _whiten(context_by_ifo[ifo])
        center_index = whitened.size // 2
        start = center_index - output_samples // 2
        strains.append(whitened[start : start + output_samples])
    return np.stack(strains), valid_ifos


def score_background_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    model_ifos: tuple[str, ...] = ("H1", "L1", "V1"),
    q_values: tuple[float, ...] = (4.0, 8.0, 16.0),
    target_sample_rate: int = 1024,
    context_duration: float = 64.0,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Trigger scoring requires torch") from exc
    from .numeric import MultiIFOQNet

    config = load_yaml(config_path)
    tensor_config = config["numeric_training"]["tensor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_channels = len(model_ifos) * len(q_values)
    if int(checkpoint["input_channels"]) != expected_channels:
        raise ValueError(
            f"Checkpoint has {checkpoint['input_channels']} channels; scorer requires {expected_channels}"
        )
    model = MultiIFOQNet(expected_channels, int(checkpoint["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    rows = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest_rows = [json.loads(line) for line in handle if line.strip()]
    started = time.time()
    failures = []
    for row in manifest_rows:
        try:
            strain, valid_ifos = _window_strain(
                row, model_ifos, target_sample_rate, context_duration
            )
            power = multiresolution_power(
                strain,
                target_sample_rate,
                q_values,
                int(tensor_config["frequency_bins"]),
                int(tensor_config["time_bins"]),
                float(tensor_config["fmin"]),
                float(tensor_config["fmax"]),
            )
            features = _normalize_power(power).reshape(
                1, expected_channels, power.shape[-2], power.shape[-1]
            )
            with torch.no_grad():
                logits = model(torch.from_numpy(features).to(device))
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            probabilities = probabilities.reshape(
                2, len(model_ifos), len(q_values), power.shape[-2], power.shape[-1]
            )
            chirp_scores = {
                ifo: float(np.max(probabilities[0, index]))
                for index, ifo in enumerate(model_ifos)
            }
            glitch_scores = {
                ifo: float(np.max(probabilities[1, index]))
                for index, ifo in enumerate(model_ifos)
            }
            ranking = network_ranking(chirp_scores, glitch_scores, valid_ifos)
            rows.append(
                {
                    "window_id": row["window_id"],
                    "split": row["split"],
                    "gps_start": row["gps_start"],
                    "gps_end": row["gps_end"],
                    "gps_block": row["gps_block"],
                    "chirp_scores": chirp_scores,
                    "glitch_scores": glitch_scores,
                    "padded_ifos": [ifo for ifo in model_ifos if ifo not in valid_ifos],
                    **ranking,
                }
            )
        except (ValueError, OSError, KeyError) as exc:
            failures.append({"window_id": row.get("window_id"), "error": str(exc)})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    triggers_path = output / "background_triggers.jsonl"
    atomic_write_text(
        triggers_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    report = {
        "status": "real_o4a_domain_transfer_diagnostic",
        "scientific_claim_allowed": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "config_path": str(config_path),
        "model_ifos": list(model_ifos),
        "q_values": list(q_values),
        "target_sample_rate": target_sample_rate,
        "context_duration": context_duration,
        "input_windows": len(manifest_rows),
        "scored_windows": len(rows),
        "failed_windows": len(failures),
        "failures": failures,
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "network_mode_counts": dict(
            sorted(Counter(row["network_mode"] for row in rows).items())
        ),
        "triggers_path": str(triggers_path),
        "triggers_sha256": file_sha256(triggers_path),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output / "trigger_score_report.json", report)
    return report
